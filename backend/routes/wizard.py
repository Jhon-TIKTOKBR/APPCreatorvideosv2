import os
import re
import urllib.parse
import sqlite3
import base64
try:
    import httpx
except ImportError:
    httpx = None
try:
    from google.genai import types
except ImportError:
    types = None
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from backend.database import get_db, DB_PATH, save_generation_to_supabase
from backend.services.tiktok_collector import (
    extract_tiktok_ids,
    fetch_tiktok_oembed,
    fetch_public_metadata_ytdlp,
    extract_tiktok_metadata_fast,
    is_blacklisted_or_invalid,
    BLACKLIST_PHRASES
)
from backend.services.ai_analyzer import (
    run_llm_json,
    run_llm_multimodal,
    has_configured_api_key,
    generate_gemini_image,
    generate_gemini_tts_wav
)

router = APIRouter(prefix="/api/wizard", tags=["Video Creation Wizard"])

def safe_b64decode(data_uri: str) -> bytes:
    """Decodifica string base64 pura ou data URI (ex: data:image/jpeg;base64,...)."""
    if not data_uri:
        return b""
    try:
        clean = data_uri.strip()
        if "," in clean:
            clean = clean.split(",", 1)[1]
        return base64.b64decode(clean)
    except Exception:
        return b""


def balance_script_scenes(scripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Garante matematicamente que o roteiro da Cena 2 tenha a mesma quantidade de palavras
    da Cena 1 (divisão equilibrada 50/50 do áudio total respeitando pontuação natural).
    """
    if not scripts:
        return scripts

    for sc in scripts:
        s1 = (sc.get("scene_1_text") or "").strip()
        s2 = (sc.get("scene_2_text") or "").strip()
        full = (sc.get("full_audio") or f"{s1} {s2}").strip()
        if not full:
            continue

        words = full.split()
        total_w = len(words)
        if total_w < 6:
            continue

        w1_count = len(s1.split())
        w2_count = len(s2.split())

        # Se a diferença for maior que 2 palavras, divide exatamente no meio (50% / 50%)
        if abs(w1_count - w2_count) > 2:
            mid = total_w // 2
            # Procura pontuação próxima ao meio (janela de ±3 palavras) para corte natural da frase
            best_split = mid
            best_dist = 999
            for idx in range(max(2, mid - 3), min(total_w - 2, mid + 4)):
                token = words[idx - 1]
                if any(token.endswith(p) for p in [".", "!", "?", ",", ";", ":"]):
                    dist = abs(idx - mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_split = idx

            new_s1 = " ".join(words[:best_split]).strip()
            new_s2 = " ".join(words[best_split:]).strip()
            sc["scene_1_text"] = new_s1
            sc["scene_2_text"] = new_s2
            sc["full_audio"] = f"{new_s1} {new_s2}".strip()

        sc["scene_1_word_count"] = len(sc["scene_1_text"].split())
        sc["scene_2_word_count"] = len(sc["scene_2_text"].split())

    return scripts

class VerifyKeyRequest(BaseModel):
    api_key: str

class StartWizardRequest(BaseModel):
    mode: Optional[str] = "photos" # "photos", "video" ou "link"
    video_format: str = "UGC-POV" # UGC, POV, UGC-POV
    product_input: Optional[str] = None # Nome do Produto
    description: Optional[str] = None # Detalhes do produto, tecido, bolsos, zíper, oferta
    images: Optional[List[Dict[str, Any]]] = None # [{"data": "base64...", "mime_type": "image/jpeg"}]
    video_data: Optional[Dict[str, Any]] = None # {"data": "base64...", "mime_type": "video/mp4" ou "audio/wav"}
    frames: Optional[List[Dict[str, Any]]] = None # [{"data": "base64...", "mime_type": "image/jpeg"}]
    api_key: Optional[str] = None # Chave Gemini enviada pelo navegador do usuário
    agent_instruction: Optional[str] = None # Instrução extra para o Agente Diretor
    # Retrocompatibilidade com links antigos
    competitor_url: Optional[str] = None
    competitor_speech_or_caption: Optional[str] = None


class AgentRefineRequest(BaseModel):
    api_key: Optional[str] = None
    product_name: str
    current_scripts: List[Dict[str, Any]]
    agent_xray: Optional[Dict[str, Any]] = None
    user_command: str
    video_format: str = "UGC-POV"


class ExtractProductInfoRequest(BaseModel):
    url: str
    context_text: Optional[str] = None
    api_key: Optional[str] = None


class ScriptVariation(BaseModel):
    id: int
    angle_title: str
    target_emotion: str
    scene_1_text: str
    scene_2_text: str
    full_audio: str
    movements_summary: str


class GenerateFlowPromptRequest(BaseModel):
    selected_script_id: int
    environment_id: str # ambiente_1, ambiente_2, ambiente_3
    script_data: Dict[str, Any]
    product_name: str
    video_format: str = "UGC-POV"
    narration_mode: Optional[str] = "separate" # "separate" (Outro site/CapCut) ou "embedded" (Direto no Flow)
    agent_xray: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None


class GenerateAngleImageRequest(BaseModel):
    angle_id: str
    angle_title: str
    angle_prompt: str
    environment_id: str = "ambiente_1"
    product_images: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[str] = None


def _load_saved_model_reference_b64(environment_id: str = "ambiente_1") -> List[Dict[str, str]]:
    """Carrega a foto original salva da modelo ruiva e do ambiente em Base64."""
    refs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "03_REFERENCIAS")
    files_to_load = ["Modelo e ambiente 1.jpeg"]
    if environment_id == "ambiente_2":
        files_to_load.append("Ambiente quarto.jpeg")
    elif environment_id == "ambiente_3":
        files_to_load.append("Modelo e ambiente 1.2.jpeg")

    loaded: List[Dict[str, str]] = []
    for fname in files_to_load:
        fpath = os.path.join(refs_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                    loaded.append({"data": b64, "mime_type": "image/jpeg"})
            except Exception:
                pass
    return loaded


@router.get("/agent-status")
def get_agent_status():
    """Verifica se o servidor na Vercel já possui GEMINI_API_KEY configurada."""
    server_has_key = has_configured_api_key()
    return {
        "server_has_key": server_has_key,
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    }


@router.post("/verify-key")
def verify_gemini_api_key(req: VerifyKeyRequest):
    """Testa em tempo real se a chave Gemini colada pelo usuário é válida."""
    key = (req.api_key or "").strip()
    if len(key) < 15:
        return {"valid": False, "message": "A chave parece curta demais. Copie a chave completa do Google AI Studio (começa com AIza...)."}
    res = run_llm_json(
        "Responda apenas JSON válido.",
        'Retorne exatamente: {"status": "online"}',
        api_key=key
    )
    if res and res.get("status") == "online":
        return {"valid": True, "message": "Agente Gemini conectado e funcionando 100%!"}
    return {"valid": False, "message": "Não foi possível autenticar essa chave no Google AI Studio. Verifique se copiou corretamente."}


def extract_product_with_regex(text: str) -> Optional[str]:
    """Heurística de identificação de categorias de produto via regex."""
    if not text:
        return None
    fashion_patterns = [
        r"((?:conjunto|vestido|blusa|calça|macaquinho|cropped|body|saia|blazer|short|kimono|camisa|jaqueta|regata|moletom|legging|pijama|macacão)\s+[a-zA-Zá-úÁ-Ú]+(?:\s+[a-zA-Zá-úÁ-Ú]+){0,3})",
        r"((?:tênis|sandália|rasteirinha|papete|bota|salto|scarpin|mocassim|chinelo|mule)\s+[a-zA-Zá-úÁ-Ú]+(?:\s+[a-zA-Zá-úÁ-Ú]+){0,3})",
        r"((?:bolsa|carteira|mala|mochila|cinto|colar|brinco|pulseira|óculos|relógio)\s+[a-zA-Zá-úÁ-Ú]+(?:\s+[a-zA-Zá-úÁ-Ú]+){0,3})",
        r"((?:sérum|batom|gloss|rímel|base|corretivo|protetor|creme|perfume|modelador|escova|babyliss|shampoo|kit)\s+[a-zA-Zá-úÁ-Ú]+(?:\s+[a-zA-Zá-úÁ-Ú]+){0,3})",
        r"((?:seladora|organizador|luminária|mini\s+\w+|suporte|massageador|aparador)\s+[a-zA-Zá-úÁ-Ú]+(?:\s+[a-zA-Zá-úÁ-Ú]+){0,3})"
    ]
    for pat in fashion_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1).title().strip()
            if not is_blacklisted_or_invalid(cand):
                return cand
    return None


def extract_product_with_gemini(text: str) -> Optional[str]:
    """Extrai nome limpo do produto usando o Gemini com base em roteiro, legenda e título."""
    if not text or is_blacklisted_or_invalid(text):
        return None
    system_prompt = """Você é um especialista em catálogo de e-commerce e TikTok Shop.
Sua tarefa é extrair e identificar o NOME COMERCIAL EXATO do produto que está sendo demonstrado ou vendido neste conteúdo (roteiro, legenda ou título).
Regras estritas:
1. Retorne APENAS o nome comercial objetivo do produto (ex: "Bolsa Transversal Couro", "Tênis Plataforma Casual", "Vestido Canelado Midi", "Sérum Facial Clareador", "Mini Seladora Térmica").
2. Remova nomes de contas (@usuario), hashtags, marcas irrelevantes, emojis e frases de clique ("compre aqui", "olha isso", "link na bio").
3. Se não houver nenhum produto detectável, retorne null.
4. Responda SEMPRE em JSON rigoroso: {"product_name": "Nome do Produto ou null"}"""

    llm_res = run_llm_json(system_prompt, f"Identifique o produto neste conteúdo:\n{text}")
    if llm_res and llm_res.get("product_name"):
        cand = str(llm_res["product_name"]).strip()
        if cand and cand.lower() != "null" and not is_blacklisted_or_invalid(cand):
            return cand
    return None


async def resolve_product_name_from_url(raw_url: str, context_text: Optional[str] = None) -> Optional[str]:
    """Extrai e normaliza o nome do produto a partir de uma URL ou texto bruto de contexto SEM usar cache antigo."""
    clean_url = raw_url.strip() if raw_url else ""
    caption = (context_text or "").strip()

    # 1. Se o usuário passou texto de contexto (fala ou legenda), tentar extrair com Gemini primeiro
    if caption:
        gemini_name = extract_product_with_gemini(caption)
        if gemini_name and not is_blacklisted_or_invalid(gemini_name):
            return gemini_name
        regex_name = extract_product_with_regex(caption)
        if regex_name and not is_blacklisted_or_invalid(regex_name):
            return regex_name

    # 2. Se for link do TikTok, buscar metadados frescos (sem cache de banco)
    if "tiktok.com" in clean_url.lower():
        meta_res = await extract_tiktok_metadata_fast(clean_url)
        if meta_res.get("success") and meta_res.get("text"):
            caption = meta_res["text"]
            gemini_name = extract_product_with_gemini(caption)
            if gemini_name and not is_blacklisted_or_invalid(gemini_name):
                return gemini_name
            regex_name = extract_product_with_regex(caption)
            if regex_name and not is_blacklisted_or_invalid(regex_name):
                return regex_name

    # 3. Checar query params na URL (ex: ?product_name=vestido-midi ou ?title=...)
    if not caption and ("?" in clean_url):
        try:
            parsed = urllib.parse.urlparse(clean_url)
            qs = urllib.parse.parse_qs(parsed.query)
            for k in ['product_name', 'title', 'name', 'item_name', 'p', 'product']:
                if k in qs and qs[k]:
                    val = qs[k][0]
                    words = [w for w in re.split(r'[-_+\s]', val) if len(w) > 1 and not w.isdigit()]
                    if len(words) >= 2:
                        cand = " ".join(words)
                        if not is_blacklisted_or_invalid(cand):
                            caption = cand
                            break
        except Exception:
            pass

    # 4. Checar slug do caminho da URL (ex: /vestido-midi-canelado/)
    if not caption and ("/" in clean_url):
        try:
            parsed = urllib.parse.urlparse(clean_url)
            parts = [p for p in parsed.path.split('/') if p and not p.isdigit() and len(p) > 2]
            ignored = {'view', 'product', 'item', 'goods', 'produtos', 'produto', 'video', 'share', 'p'}
            candidates = [p for p in parts if p.lower() not in ignored and not p.startswith('@')]
            if candidates:
                last = candidates[-1].replace('.html', '').replace('.htm', '')
                words = [w for w in re.split(r'[-_]', last) if len(w) > 2 and not w.isdigit() and not w.lower().startswith('mlb')]
                if len(words) >= 2:
                    cand = " ".join(words)
                    if not is_blacklisted_or_invalid(cand):
                        caption = cand
        except Exception:
            pass

    text_to_analyze = caption or (clean_url if not clean_url.startswith("http") else "")
    if not text_to_analyze or is_blacklisted_or_invalid(text_to_analyze):
        return None

    # Tenta Gemini com todo o contexto
    gemini_cand = extract_product_with_gemini(text_to_analyze)
    if gemini_cand and not is_blacklisted_or_invalid(gemini_cand):
        return gemini_cand

    # Tenta regex
    regex_cand = extract_product_with_regex(text_to_analyze)
    if regex_cand and not is_blacklisted_or_invalid(regex_cand):
        return regex_cand

    # Limpeza básica de texto
    cleaned = re.sub(r'#\w+', '', text_to_analyze)
    cleaned = re.sub(r'https?://\S+', '', cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1 and not w.isdigit()][:4]
    if len(words) >= 2:
        cand = " ".join(words).title()
        if not is_blacklisted_or_invalid(cand):
            return cand

    return None


@router.post("/extract-product-info")
async def extract_product_info(req: ExtractProductInfoRequest):
    """
    Identifica e extrai o nome do produto automaticamente a partir de um link ou texto.
    """
    raw_url = req.url.strip() if req.url else ""
    context_text = req.context_text.strip() if req.context_text else ""

    if not raw_url and not context_text:
        return {"success": False, "product_name": None}

    clean_name = await resolve_product_name_from_url(raw_url, context_text=context_text)
    if clean_name and not is_blacklisted_or_invalid(clean_name):
        return {
            "success": True,
            "product_name": clean_name
        }

    return {
        "success": False,
        "product_name": None,
        "message": "Não foi possível extrair automaticamente. Digite o nome do produto ou cole a fala/legenda do vídeo."
    }


@router.get("/environments")
def list_environments():
    """Retorna os 3 ambientes configurados com base nas referências do usuário."""
    return {
        "environments": [
            {
                "id": "ambiente_1",
                "title": "Ambiente 1: Provador de Boutique",
                "subtitle": "Fiel à foto de referência enviada",
                "image_url": "/references/Modelo%20e%20ambiente%201.jpeg",
                "lighting": "Luz difusa de teto de boutique, piso porcelanato escuro com reflexo nítido, cortina preta e cabides de madeira.",
                "vibe": "Comercial, estilo loja de shopping, alta conversão para looks femininos."
            },
            {
                "id": "ambiente_2",
                "title": "Ambiente 2: Quarto Acolhedor",
                "subtitle": "Fiel à foto 'Ambiente quarto.jpeg'",
                "image_url": "/references/Ambiente%20quarto.jpeg",
                "lighting": "Luz suave e acolhedora de abajur de cabeceira e claridade do quarto, cama com colcha de girassol, tapete felpudo claro e piso de madeira.",
                "vibe": "Intimista, vida real, provador em casa, conforto e alta conexão emocional."
            },
            {
                "id": "ambiente_3",
                "title": "Ambiente 3: Hall com Espelho Orgânico LED",
                "subtitle": "Estilo moderno e sofisticado",
                "image_url": "/references/Modelo%20e%20ambiente%201.2.jpeg",
                "lighting": "LED indireto halo quente atrás do espelho orgânico assimétrico, hall contemporâneo com piso brilhante.",
                "vibe": "Arrumada para sair à noite, alto valor percebido."
            }
        ]
    }


def build_category_scripts(product_name: str, video_format: str, raw_context: str = "", visual_details: str = "") -> List[Dict[str, Any]]:
    prod = product_name.strip()
    low = (prod + " " + (raw_context or "") + " " + (visual_details or "")).lower()

    # Detectar características visuais e de produto para enriquecer dinamicamente os movimentos
    has_pocket = any(k in low for k in ["bolso", "bolsos", "pockets"])
    has_zipper = any(k in low for k in ["zíper", "ziper", "fecho"])
    has_stretch = any(k in low for k in ["elasticidade", "elastano", "estica", "canelado", "lycra"])
    has_lining = any(k in low for k in ["duplo", "não marca", "nao marca", "forro", "transparência", "transparencia"])

    extra_movement = ""
    if has_stretch:
        extra_movement += "puxa suavemente o tecido com ambas as mãos demonstrando elasticidade e retorno sem deformar; "
    if has_pocket:
        extra_movement += "aproxima a mão em close-up deslizando e puxando o bolso mostrando costura reforçada; "
    if has_zipper:
        extra_movement += "destaca o fechamento com zíper de boutique em ângulo próximo; "

    # 1. Calçados (tênis, sandália, bota, salto, etc.)
    if any(k in low for k in ["tênis", "tenis", "sandália", "sandalia", "bota", "salto", "scarpin", "mocassim", "chinelo", "papete", "rasteira", "mule", "calçado", "calcado"]):
        return [
            {
                "id": 1,
                "angle_title": "Desejo & Conforto Absurdo (Cores)",
                "target_emotion": "Desejo incontrolável de ter todas as cores sem cansar os pés",
                "scene_1_text": f"Comprei uma, duas, três cores desse {prod} de tanto que me apaixonei! Eu estava cansada de calçado duro que machuca o calcanhar e cansa a perna. Esse modelo tem palmilha ultra macia e acabamento de shopping,",
                "scene_2_text": "combina com absolutamente tudo, do jeans ao vestido, e é super leve! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante a sua numeração antes que acabe!",
                "full_audio": f"Comprei uma, duas, três cores desse {prod} de tanto que me apaixonei! Eu estava cansada de calçado duro que machuca o calcanhar e cansa a perna. Esse modelo tem palmilha ultra macia e acabamento de shopping, combina com absolutamente tudo, do jeans ao vestido, e é super leve! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante a sua numeração antes que acabe!",
                "movements_summary": "Modelo desfila em direção ao espelho, vira os pés mostrando sola e detalhes da lateral em 360 graus, dá passos leves e flexiona para mostrar maciez, finalizando aos 9-10s apontando o dedo indicador firme para o carrinho."
            },
            {
                "id": 2,
                "angle_title": "Quebra de Ceticismo & Durabilidade",
                "target_emotion": "Alívio e surpresa positiva com o material de alta qualidade",
                "scene_1_text": f"Eu tinha certeza que por esse valor esse {prod} ia ser duro ou de material fraco. Mas quando chegou no correio e calcei... olha a perfeição dessa costura e o conforto!",
                "scene_2_text": "Não aperta os dedos, não faz bolha e dá para passar o dia inteiro batendo perna sem dor nenhuma. Clica no carrinho aqui embaixo antes que as numerações mais pedidas esgotem!",
                "full_audio": f"Eu tinha certeza que por esse valor esse {prod} ia ser duro ou de material fraco. Mas quando chegou no correio e calcei... olha a perfeição dessa costura e o conforto! Não aperta os dedos, não faz bolha e dá para passar o dia inteiro batendo perna sem dor nenhuma. Clica no carrinho aqui embaixo antes que as numerações mais pedidas esgotem!",
                "movements_summary": "Aproximação rápida no espelho, passo firme à frente, exibe o calçado de perfil para um lado e para o outro, joga o cabelo ruivo e aponta o dedo indicador para baixo no carrinho aos 9-10s."
            },
            {
                "id": 3,
                "angle_title": "Versatilidade & Look Caro",
                "target_emotion": "Sensação de estar usando calçado de marca cara",
                "scene_1_text": f"A única peça que eu consigo usar o dia todo no trabalho e ir direto pro restaurante à noite sem cansar nada! Esse {prod} deixa qualquer look básico chique em segundos.",
                "scene_2_text": "Parece aqueles modelos importados de vitrine que custam 400 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "full_audio": f"A única peça que eu consigo usar o dia todo no trabalho e ir direto pro restaurante à noite sem cansar nada! Esse {prod} deixa qualquer look básico chique em segundos. Parece aqueles modelos importados de vitrine que custam 400 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "movements_summary": "Giro confiante de corpo inteiro no espelho, passo elegante à frente destacando os pés, vira de perfil de ambos os lados e finaliza com dedo indicador apontando direto para o botão do carrinho."
            }
        ]

    # 2. Bolsas e Acessórios
    if any(k in low for k in ["bolsa", "carteira", "mala", "mochila", "cinto", "colar", "brinco", "semijoia", "óculos", "oculos", "relógio", "relogio", "acessório", "acessorio"]):
        return [
            {
                "id": 1,
                "angle_title": "Desejo & Elegância Imediata",
                "target_emotion": "Desejo de elevar qualquer visual em segundos",
                "scene_1_text": f"Comprei uma, duas cores dessa {prod} de tão maravilhosa! Eu estava cansada de peça que descasca na primeira semana ou tem zíper que emperra. Olha o peso e o acabamento dos metais,",
                "scene_2_text": "tem divisórias perfeitas que cabem tudo e deixa qualquer roupa básica com cara de look milionário! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante a sua!",
                "full_audio": f"Comprei uma, duas cores dessa {prod} de tão maravilhosa! Eu estava cansada de peça que descasca na primeira semana ou tem zíper que emperra. Olha o peso e o acabamento dos metais, tem divisórias perfeitas que cabem tudo e deixa qualquer roupa básica com cara de look milionário! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante a sua!",
                "movements_summary": "Modelo segura o acessório em destaque em frente ao espelho, abre e exibe os detalhes internos e metais reluzentes, coloca no ombro/braço, desfila de perfil e aponta firme para o carrinho aos 9-10s."
            },
            {
                "id": 2,
                "angle_title": "Quebra de Ceticismo & Acabamento Premium",
                "target_emotion": "Surpresa com a qualidade das costuras e material",
                "scene_1_text": f"Eu tinha certeza que por esse preço essa {prod} ia ter costura torta ou material grosseiro. Mas quando abri a caixa... olha o luxo dessa peça!",
                "scene_2_text": "O forro é reforçado, o toque é impecável e não risca fácil. Clica no carrinho aqui embaixo antes que as cores mais elegantes esgotem!",
                "full_audio": f"Eu tinha certeza que por esse preço essa {prod} ia ter costura torta ou material grosseiro. Mas quando abri a caixa... olha o luxo dessa peça! O forro é reforçado, o toque é impecável e não risca fácil. Clica no carrinho aqui embaixo antes que as cores mais elegantes esgotem!",
                "movements_summary": "Aproximação dinâmica no espelho, passa as mãos suavemente no material demonstrando textura, vira de lado ajustando no corpo, e aponta o dedo indicador para o carrinho no final."
            },
            {
                "id": 3,
                "angle_title": "Versatilidade (Dia a Dia à Festa)",
                "target_emotion": "Praticidade e elegância de boutique de luxo",
                "scene_1_text": f"A única peça que eu consigo usar de manhã no trabalho com alfaiataria e sair à noite sem precisar trocar! Essa {prod} é um absurdo de linda.",
                "scene_2_text": "Parece aquelas peças de grife de shopping que custam 500 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "full_audio": f"A única peça que eu consigo usar de manhã no trabalho com alfaiataria e sair à noite sem precisar trocar! Essa {prod} é um absurdo de linda. Parece aquelas peças de grife de shopping que custam 500 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "movements_summary": "Giro confiante de 360 graus exibindo o caimento com a bolsa/acessório, passo firme à frente, joga o cabelo ruivo e aponta direto para o botão do carrinho."
            }
        ]

    # 3. Beleza, Skincare e Cabelo
    if any(k in low for k in ["sérum", "serum", "batom", "gloss", "rímel", "rimel", "base", "corretivo", "creme", "perfume", "modelador", "escova", "babyliss", "shampoo", "cabelo", "maquiag", "skincare"]):
        return [
            {
                "id": 1,
                "angle_title": "Resultado Imediato & Desejo",
                "target_emotion": "Encanto com o efeito visível nos primeiros segundos",
                "scene_1_text": f"Comprei logo dois frascos desse {prod} antes que esgotasse de vez! Eu estava cansada de produto caro que promete milagre e não muda nada. Olha o acabamento e a textura logo na primeira passada,",
                "scene_2_text": "deixa aquele efeito glow natural de salão e dura o dia inteiro sem craquelar ou ressecar! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
                "full_audio": f"Comprei logo dois frascos desse {prod} antes que esgotasse de vez! Eu estava cansada de produto caro que promete milagre e não muda nada. Olha o acabamento e a textura logo na primeira passada, deixa aquele efeito glow natural de salão e dura o dia inteiro sem craquelar ou ressecar! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
                "movements_summary": "Modelo segura o produto em destaque próximo ao espelho, exibe o efeito visual impecável de pele/cabelo bem cuidado, joga os fios ruivos para trás com brilho, vira de perfil e aponta firme para o carrinho no final."
            },
            {
                "id": 2,
                "angle_title": "Quebra de Ceticismo & Fórmula Real",
                "target_emotion": "Alívio com a eficácia comprovada no espelho",
                "scene_1_text": f"Eu tinha certeza que esse {prod} era só mais uma modinha de internet que não funciona. Mas quando testei no espelho e vi a transformação...",
                "scene_2_text": "A fórmula rende muito, tem toque seco e o resultado é imediato sem precisar de filtro nenhum! Clica no carrinho aqui embaixo antes que o lote promocional esgote!",
                "full_audio": f"Eu tinha certeza que esse {prod} era só mais uma modinha de internet que não funciona. Mas quando testei no espelho e vi a transformação... A fórmula rende muito, tem toque seco e o resultado é imediato sem precisar de filtro nenhum! Clica no carrinho aqui embaixo antes que o lote promocional esgote!",
                "movements_summary": "Aproximação surpresa no espelho, toque delicado no rosto/cabelo demonstrando suavidade e maciez, vira de perfil sorrindo com confiança e finaliza estendendo o dedo indicador apontando para o carrinho aos 9-10s."
            },
            {
                "id": 3,
                "angle_title": "Efeito Salão / Tratamento Caro",
                "target_emotion": "Sensação de economia inteligente com padrão profissional",
                "scene_1_text": f"O único produto da minha rotina que me economizou centenas de reais em clínica e salão! Esse {prod} entrega um resultado profissional em menos de 2 minutos.",
                "scene_2_text": "Parece aqueles importados caríssimos que custam 300 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "full_audio": f"O único produto da minha rotina que me economizou centenas de reais em clínica e salão! Esse {prod} entrega um resultado profissional em menos de 2 minutos. Parece aqueles importados caríssimos que custam 300 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "movements_summary": "Giro dinâmico no espelho exibindo o visual impecável de perto, passo à frente com postura confiante, joga o cabelo ruivo e aponta direto para o botão do carrinho."
            }
        ]

    # 4. Moda Feminina (Padrão e Roupas) com injeção de detalhes (bolso, zíper, elasticidade)
    mov_1 = f"Movimentação contínua em alta velocidade: {extra_movement}a modelo estica o tecido mostrando elasticidade; passa as mãos firmemente na cintura mostrando caimento liso sem marcas; dá passos à frente em direção ao espelho; vira de perfil para a esquerda, gira e vira para a direita; joga o cabelo ruivo para trás do ombro; cortes rápidos a cada 1.2s trocando de cor; e finaliza aos 9s-10s estendendo o dedo indicador apontando firme para baixo no carrinho do TikTok Shop."
    mov_2 = f"Movimentação contínua em alta velocidade: aproximação surpresa no espelho dando passos à frente; {extra_movement}puxão firme com as duas mãos esticando o tecido para provar elasticidade e zero transparência; passa as mãos na lateral da cintura mostrando caimento reto; vira de perfil; joga o cabelo ruivo para trás; finaliza aos 9s-10s apontando o dedo indicador para baixo no carrinho."
    mov_3 = f"Movimentação contínua em alta velocidade: giro confiante de 360 graus de corpo inteiro; {extra_movement}passa as mãos deslizando pelo tecido ajustando a peça; caminha 2 passos à frente; joga o cabelo ruivo para trás; vira de perfil de ambos os lados e finaliza estendendo o dedo indicador apontando direto para o botão do carrinho."

    return [
        {
            "id": 1,
            "angle_title": "Desejo & Contagem de Cores",
            "target_emotion": "Desejo incontrolável de ter todas as cores",
            "scene_1_text": f"Comprei uma, duas, três, quatro cores desse {prod} de tanto que me apaixonei! Eu estava cansada de roupa que fica transparente e marca a barriga. Esse modelo tem o caimento perfeito que modela o corpo,",
            "scene_2_text": "esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
            "full_audio": f"Comprei uma, duas, três, quatro cores desse {prod} de tanto que me apaixonei! Eu estava cansada de roupa que fica transparente e marca a barriga. Esse modelo tem o caimento perfeito que modela o corpo, esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
            "movements_summary": mov_1
        },
        {
            "id": 2,
            "angle_title": "Quebra de Ceticismo & Caimento Real",
            "target_emotion": "Alívio e surpresa positiva com o tecido sem transparência",
            "scene_1_text": f"Eu tinha certeza que por esse preço esse {prod} ia marcar toda a barriga ou ficar transparente. Mas quando chegou no correio e provei... olha o caimento dessa modelagem!",
            "scene_2_text": "O tecido é duplo, não tem transparência nenhuma e disfarça qualquer imperfeição na hora. Veste do 38 ao 46 perfeitamente! Clica no carrinho aqui embaixo antes que as cores mais pedidas esgotem!",
            "full_audio": f"Eu tinha certeza que por esse preço esse {prod} ia marcar toda a barriga ou ficar transparente. Mas quando chegou no correio e provei... olha o caimento dessa modelagem! O tecido é duplo, não tem transparência nenhuma e disfarça qualquer imperfeição na hora. Veste do 38 ao 46 perfeitamente! Clica no carrinho aqui embaixo antes que as cores mais pedidas esgotem!",
            "movements_summary": mov_2
        },
        {
            "id": 3,
            "angle_title": "Versatilidade (Trabalho ao Jantar)",
            "target_emotion": "Praticidade e sensação de estar elegante com roupa de shopping cara",
            "scene_1_text": f"A única peça do meu guarda-roupa que eu consigo usar no trabalho de manhã com blazer e ir direto pro restaurante à noite sem amassar nada! Esse {prod} é um absurdo de chique.",
            "scene_2_text": "Parece aquelas peças de shopping que custam 300 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
            "full_audio": f"A única peça do meu guarda-roupa que eu consigo usar no trabalho de manhã com blazer e ir direto pro restaurante à noite sem amassar nada! Esse {prod} é um absurdo de chique. Parece aquelas peças de shopping que custam 300 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
            "movements_summary": mov_3
        }
    ]


@router.post("/analyze-and-generate-scripts")
async def analyze_and_generate_scripts(req: StartWizardRequest):
    """
    Agente Diretor de Criação TikTok Shop:
    - Modo 'photos': Inspeciona 1 a 4 fotos/prints + descrição via Gemini Vision, gera Raio-X Visual e 3 roteiros 100% inéditos.
    - Modo 'video': Transcreve o áudio/vídeo (.mp4) + analisa os frames do concorrente via Gemini Multimodal e modela 3 roteiros exclusivos.
    """
    mode = req.mode or "photos"
    user_api_key = (req.api_key or "").strip() or None
    extra_instruction = (req.agent_instruction or "").strip()

    # Verifica se há chave Gemini ativa (no servidor Vercel ou enviada pelo navegador do usuário)
    if not has_configured_api_key(user_api_key):
        return {
            "success": False,
            "error": "API_KEY_REQUIRED",
            "message": "O Agente Diretor precisa da sua Chave Gratuita do Gemini para enxergar os prints, transcrever o vídeo e criar roteiros 100% inéditos. Clique para conectar sua chave em 30 segundos!"
        }

    # =========================================================================
    # ESTRATÉGIA 1: FOTOS / PRINTS + TEXTO / DESCRIÇÃO (AGENTE VISION)
    # =========================================================================
    if mode == "photos":
        images_list = req.images or []
        desc = (req.description or "").strip()
        prod_input = (req.product_input or "").strip()

        if not images_list and not desc and not prod_input:
            return {
                "success": False,
                "error": "INPUT_REQUIRED",
                "message": "Por favor, adicione pelo menos uma foto/print do produto ou preencha o nome e a descrição do produto."
            }

        product_name = prod_input if (prod_input and not prod_input.startswith("http")) else ""

        system_prompt = """Você é um Agente Diretor Sênior de Criação de Vídeos Virais para TikTok Shop Brasil (especialista em UGC-POV de alta conversão).
Sua missão é agir como um verdadeiro Diretor de Criação autônomo:
1. RAIO-X VISUAL E TÉCNICO: Inspecione minuciosamente cada foto/print enviado (inclusive textos presentes em prints de anúncios) e a descrição do usuário. Identifique:
   - O nome comercial exato e modelo do produto.
   - O tecido/material aparente, textura, elasticidade, acabamento de costura.
   - Todos os detalhes específicos da peça: parte da frente, parte de trás, bolsos, zíperes, botões, cós, fendas, caimento.
2. ESTILOS DE GANCHOS VIRAIS + STORYTELLING REAL ENVOLVENTE (OBRIGATÓRIO):
   Cada um dos 3 roteiros DEVE iniciar a Cena 1 com um gancho viral brasileiro de choque de preço ou dor real E desenvolver um MICRO-STORYTELLING humano e autêntico (expectativa vs. realidade, situação real de uso, reação da mãe/amigas ou superação de uma frustração com roupas), conectando a história aos detalhes reais vistos nas fotos:
   - Roteiro 1 (Choque de Preço + Storytelling de Expectativa vs. Realidade): Iniciar com "Tava tão barato que eu comprei foi logo 3...", "Nem minha mãe acreditou no preço que paguei nesse [produto]..." ou "Tá tão barato que se eu te falar o preço você não acredita...", contando que comprou achando que pelo preço seria simples, mas quando abriu o pacote e vestiu ficou chocada com o tecido, bolso e acabamento.
   - Roteiro 2 (Dor Real + Storytelling de Autoestima/Alívio): Iniciar com "Eu tava cansada de roupa que aperta e marca tudo..." ou "Eu já não aguentava mais passar raiva com roupa transparente...", contando uma situação real do dia a dia e como essa peça vestiu perfeitamente sem marcar nada.
   - Roteiro 3 (Storytelling de Elogio / Loja Cara de Shopping): Iniciar com "Se eu te falar quanto paguei nesse [produto] você acha que é mentira...", contando que usou para sair/trabalhar e todo mundo achou que era peça de 300 reais de boutique.
3. CENAS MAIS RÁPIDAS ESTILO TIME-LAPSE, APENAS MEIO GIRO (SEM CELULAR FLUTUANDO) E TELA 100% LIMPA:
   Em 'movements_summary', dirija a modelo em ritmo acelerado estilo TIME-LAPSE com cortes rápidos (fast jump cuts a cada 0.8s a 1.2s):
   - REGRA FÍSICA DO CELULAR NA MÃO (PROIBIDO CELULAR FLUTUANDO): Os dedos da mão da modelo envolvem e seguram firmemente o celular branco na frente do rosto o tempo todo (conectado fisicamente à mão, punho e braço — NUNCA solto ou flutuando no ar), mantendo a mesma posição mesmo quando ela se aproxima em close-up para mostrar a roupa mais de perto com a mão livre.
   - REGRA DE APENAS MEIO GIRO LATERAL (PROIBIDO GIRO 360°): A modelo faz APENAS MEIO GIRO lateral suave de quadril e tronco (45° a 75° de um lado para o outro) para mostrar o caimento lateral e a parte de trás da roupa sem nunca girar 360° completos (para o celular nunca desgrudar da mão).
   - REGRA DE TELA LIMPA: NÃO deve aparecer nenhum texto, legenda ou ícone na tela do vídeo.
   - A mão livre puxa suavemente o bolso e a costura do tecido de perto demonstrando elasticidade e acabamento reforçado, mantendo sempre 100% as características exatas da roupa; finalização aos 9-10s apontando o dedo indicador para o carrinho.
4. REGRA OBRIGATÓRIA DE EQUILÍBRIO DE PALAVRAS (CENA 1 = CENA 2):
   O roteiro da Cena 2 ('scene_2_text') DEVE ter EXATAMENTE A MESMA QUANTIDADE DE PALAVRAS que o roteiro da Cena 1 ('scene_1_text') — cerca de 28 a 32 palavras na Cena 1 e 28 a 32 palavras na Cena 2.

Responda SEMPRE em JSON rigoroso com esta estrutura exata:
{
  "product_name": "Nome Comercial Exato do Produto",
  "visual_details": "Descrição técnica em inglês das características exatas da peça (cor, corte, bolsos, parte de trás, tecido, costuras) para preservar 100% a fidelidade visual no gerador de imagem e vídeo",
  "agent_xray": {
    "product_identified": "Nome e modelo detalhado identificado pelo Agente",
    "fabric_and_finish": "Análise do tecido, elasticidade, gramatura e acabamento",
    "detected_features": [
      "Detalhe real 1 detectado nas fotos/print",
      "Detalhe real 2 detectado (ex: bolso, costura, parte de trás)",
      "Detalhe real 3 detectado",
      "Detalhe real 4 detectado"
    ],
    "competitor_transcript": "Texto/descrição lido dos prints ou pontos-chave extraídos da oferta",
    "agent_strategy": "Explicação direta do Agente Diretor sobre o storytelling com ganchos de choque de preço/dor e o ritmo time-lapse com meio giro criado para esta peça"
  },
  "scripts": [
    {
      "id": 1,
      "angle_title": "Choque de Preço & Storytelling ('Tava tão barato que comprei logo 3 / Nem minha mãe acreditou')",
      "target_emotion": "Euforia de achado imperdível + surpresa real",
      "scene_1_text": "Fala natural em PT-BR para Cena 1 (0-10s) com ~30 palavras unindo gancho de choque de preço + micro-storytelling e detalhes reais da peça",
      "scene_2_text": "Fala natural em PT-BR para Cena 2 (10-20s) com EXATAMENTE a mesma quantidade de palavras da Cena 1 (~30 palavras) concluindo a história com prova de qualidade + CTA urgente para o carrinho",
      "full_audio": "Junção fluida de scene_1_text + scene_2_text",
      "movements_summary": "Estilo Time-Lapse com cortes rápidos (0.8s-1.2s) e ZERO texto/ícone na tela: mão segurando firmemente o celular fixo na mesma posição na frente do rosto (sem celular flutuando, mesmo de perto); aproximação rápida em close-up com a mão livre puxando suavemente o bolso e a costura mostrando elasticidade; apenas MEIO GIRO lateral (45° a 75°) mostrando perfil e parte de trás mantendo as características exatas da roupa; dedo apontando para o carrinho aos 9-10s"
    },
    {
      "id": 2,
      "angle_title": "Fim da Frustração ('Eu tava cansada de roupa que aperta / transparente')",
      "target_emotion": "Alívio e identificação imediata",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    },
    {
      "id": 3,
      "angle_title": "Qualidade de Boutique ('Tá tão barato que se eu falar o preço você não acredita')",
      "target_emotion": "Certeza de compra inteligente",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    }
  ]
}"""

        user_text = f"""DADOS ENVIADOS PELO CRIADOR:
- Nome do produto informado: {product_name or 'Identifique automaticamente pelos prints/fotos e descrição'}
- Descrição / Detalhes fornecidos: {desc or 'Extraia todos os detalhes diretamente das imagens/prints anexados'}
- Formato de vídeo: {req.video_format}
- Quantidade de fotos/prints anexados: {len(images_list)}
{f'- Instrução especial do criador: {extra_instruction}' if extra_instruction else ''}

Faça o Raio-X completo das imagens e do texto e crie 3 roteiros 100% exclusivos e ultra persuasivos para este produto."""

        gemini_res = None
        if images_list:
            multimodal_contents = []
            for img in images_list[:4]:
                raw_data = img.get("data", "")
                if raw_data:
                    multimodal_contents.append({
                        "data": raw_data,
                        "mime_type": img.get("mime_type", "image/jpeg")
                    })
            multimodal_contents.append(user_text)
            gemini_res = run_llm_multimodal(system_prompt, multimodal_contents, api_key=user_api_key)
        else:
            gemini_res = run_llm_json(system_prompt, user_text, api_key=user_api_key)

        if not gemini_res or "scripts" not in gemini_res or len(gemini_res.get("scripts", [])) < 1:
            return {
                "success": False,
                "error": "AGENT_EXECUTION_ERROR",
                "message": "O Agente Gemini não conseguiu processar a resposta com a chave atual. Verifique se sua Chave Gemini está ativa e tente novamente."
            }

        scripts = balance_script_scenes(gemini_res["scripts"])
        if not product_name and gemini_res.get("product_name"):
            product_name = str(gemini_res["product_name"]).strip()
        if not product_name:
            product_name = "Produto Identificado"

        return {
            "success": True,
            "ai_engine_active": True,
            "mode": "photos",
            "product_name": product_name,
            "is_generic_product": False,
            "visual_details": gemini_res.get("visual_details", ""),
            "agent_xray": gemini_res.get("agent_xray", {}),
            "video_format": req.video_format,
            "scripts": scripts
        }

    # =========================================================================
    # ESTRATÉGIA 2: SUBIR VÍDEO (.MP4) / ÁUDIO DO CONCORRENTE (AGENTE MULTIMODAL)
    # =========================================================================
    elif mode == "video":
        video_info = req.video_data
        frames_list = req.frames or []
        desc = (req.description or "").strip()
        prod_input = (req.product_input or "").strip()
        user_caption = (req.competitor_speech_or_caption or "").strip()

        if not video_info and not frames_list and not user_caption and not desc:
            return {
                "success": False,
                "error": "VIDEO_REQUIRED",
                "message": "Por favor, selecione e suba um arquivo de vídeo (.mp4, .mov) ou áudio do concorrente."
            }

        product_name = prod_input if (prod_input and not prod_input.startswith("http")) else ""

        system_prompt = """Você é um Agente Diretor Sênior de Engenharia Reversa de Vídeos Virais para TikTok Shop Brasil.
Sua missão:
1. TRANSCRIÇÃO REAL E ANÁLISE VISUAL: Ouça o áudio/vídeo anexado e inspecione os frames capturados do vídeo do concorrente.
   - Transcreva palavra por palavra o que é dito no vídeo em 'competitor_transcript' (se o vídeo tiver apenas música/texto na tela, transcreva os textos visíveis nos frames e descreva a ação exata da cena).
   - Identifique o produto exato, tecido, detalhes de frente/costas, bolsos, costura e caimento mostrados no vídeo.
2. ESTILOS DE GANCHOS VIRAIS + STORYTELLING REAL ENVOLVENTE (OBRIGATÓRIO):
   Crie 3 roteiros 100% inéditos modelados no produto do vídeo, onde cada roteiro inicia a Cena 1 com um gancho viral brasileiro E desenvolve um MICRO-STORYTELLING humano e autêntico (expectativa vs. realidade, situação real de uso, reação da mãe/amigas ou superação de frustração com roupas):
   - Roteiro 1 (Choque de Preço + Storytelling de Expectativa vs. Realidade): Iniciar com "Tava tão barato que eu comprei foi logo 3...", "Nem minha mãe acreditou no preço que paguei nesse [produto]..." ou "Tá tão barato que se eu te falar o preço você não acredita...".
   - Roteiro 2 (Dor Real + Storytelling de Autoestima/Alívio): Iniciar com "Eu tava cansada de roupa que aperta e marca tudo..." ou "Eu já não aguentava mais roupa transparente...".
   - Roteiro 3 (Storytelling de Elogio / Loja Cara de Shopping): Iniciar com "Se eu te falar quanto paguei nesse [produto] com acabamento de shopping você acha que é mentira...".
3. CENAS MAIS RÁPIDAS ESTILO TIME-LAPSE, APENAS MEIO GIRO (SEM CELULAR FLUTUANDO) E TELA 100% LIMPA:
   Em 'movements_summary', inclua movimentação super rápida e dinâmica estilo TIME-LAPSE com cortes rápidos (fast jump cuts a cada 0.8s–1.2s):
   - REGRA FÍSICA DO CELULAR NA MÃO (PROIBIDO CELULAR FLUTUANDO): Os dedos da mão da modelo envolvem e seguram firmemente o celular branco na frente do rosto o tempo todo (conectado fisicamente à mão, punho e braço — NUNCA solto ou flutuando no ar), mantendo a mesma posição mesmo quando ela se aproxima em close-up para mostrar a roupa mais de perto com a mão livre.
   - REGRA DE APENAS MEIO GIRO LATERAL (PROIBIDO GIRO 360°): A modelo faz APENAS MEIO GIRO lateral suave de quadril e tronco (45° a 75° de um lado para o outro) para mostrar o caimento lateral e a parte de trás da roupa sem nunca girar 360° completos.
   - REGRA DE TELA LIMPA: NÃO deve aparecer nenhum texto, legenda ou ícone na tela do vídeo.
   - Aproximação rápida em close-up com a mão livre puxando suavemente o bolso e a costura do tecido demonstrando elasticidade e acabamento reforçado, mantendo sempre as características exatas da roupa; e apontando para o carrinho aos 9-10s.
4. REGRA OBRIGATÓRIA DE EQUILÍBRIO DE PALAVRAS (CENA 1 = CENA 2):
   O roteiro da Cena 2 ('scene_2_text') DEVE ter EXATAMENTE A MESMA QUANTIDADE DE PALAVRAS que o roteiro da Cena 1 ('scene_1_text') — cerca de 28 a 32 palavras na Cena 1 e 28 a 32 palavras na Cena 2.

Responda SEMPRE em JSON rigoroso com esta estrutura exata:
{
  "product_name": "Nome Comercial Exato do Produto",
  "visual_details": "Descrição técnica em inglês das características exatas da peça vista no vídeo (cor, corte, bolsos, parte de trás, tecido, costuras)",
  "agent_xray": {
    "product_identified": "Produto e modelo identificados no vídeo",
    "fabric_and_finish": "Tecido, elasticidade e acabamento observados nas cenas",
    "detected_features": [
      "Gancho/Detalhe 1 extraído do vídeo",
      "Detalhe visual 2 da peça (bolso, costura, parte de trás)",
      "Gatilho de conversão 3 identificado",
      "Movimento-chave 4 observado"
    ],
    "competitor_transcript": "Transcrição completa do áudio falado no vídeo e/ou textos na tela",
    "agent_strategy": "Análise do Agente Diretor: o que o concorrente fez no vídeo e como aplicamos o storytelling com ganchos de choque de preço e ritmo time-lapse com meio giro"
  },
  "scripts": [
    {
      "id": 1,
      "angle_title": "Choque de Preço & Storytelling ('Tava tão barato que comprei logo 3 / Nem minha mãe acreditou')",
      "target_emotion": "Euforia de achado imperdível + surpresa real",
      "scene_1_text": "Fala em PT-BR Cena 1 (0-10s) com ~30 palavras unindo gancho de choque de preço + micro-storytelling",
      "scene_2_text": "Fala em PT-BR Cena 2 (10-20s) com EXATAMENTE a mesma quantidade de palavras da Cena 1 (~30 palavras) + CTA para o carrinho",
      "full_audio": "Áudio completo unido",
      "movements_summary": "Estilo Time-Lapse com cortes rápidos (0.8s-1.2s) e ZERO texto ou ícone na tela: mão segurando firmemente o celular fixo na mesma posição na frente do rosto (sem celular flutuando, mesmo mostrando a roupa de perto); close-up rápido puxando bolso e costura com a mão livre mostrando elasticidade; apenas MEIO GIRO lateral (45° a 75°) mostrando perfil e parte de trás mantendo características exatas da roupa; e dedo apontando para o carrinho aos 9-10s"
    },
    {
      "id": 2,
      "angle_title": "Fim da Frustração ('Eu tava cansada de roupa que aperta / transparente')",
      "target_emotion": "Alívio e confiança",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    },
    {
      "id": 3,
      "angle_title": "Qualidade de Boutique ('Tá tão barato que se eu te falar o preço você não acredita')",
      "target_emotion": "Urgência de compra",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    }
  ]
}"""

        user_text = f"""DADOS DO VÍDEO ENVIADO:
- Produto informado pelo vendedor: {product_name or 'Identifique automaticamente pelo vídeo e frames'}
- Observações / Instruções do vendedor: {desc or user_caption or 'Transcreva o vídeo e modele os 3 roteiros com base no produto real do vídeo'}
- Formato desejado: {req.video_format}
{f'- Comando extra ao Agente: {extra_instruction}' if extra_instruction else ''}

Transcreva o vídeo/cenas, faça o Raio-X completo e gere os 3 roteiros exclusivos com mesma quantidade de palavras na Cena 1 e Cena 2."""

        gemini_res = None
        if video_info or frames_list:
            multimodal_contents = []
            if video_info and video_info.get("data"):
                multimodal_contents.append({
                    "data": video_info.get("data"),
                    "mime_type": video_info.get("mime_type", "video/mp4")
                })

            for frame in frames_list[:4]:
                if frame.get("data"):
                    multimodal_contents.append({
                        "data": frame.get("data"),
                        "mime_type": frame.get("mime_type", "image/jpeg")
                    })

            multimodal_contents.append(user_text)
            gemini_res = run_llm_multimodal(system_prompt, multimodal_contents, api_key=user_api_key)
        else:
            gemini_res = run_llm_json(system_prompt, user_text, api_key=user_api_key)

        if not gemini_res or "scripts" not in gemini_res or len(gemini_res.get("scripts", [])) < 1:
            return {
                "success": False,
                "error": "AGENT_EXECUTION_ERROR",
                "message": "Não foi possível transcrever ou processar o vídeo com a chave atual. Se o vídeo for muito longo, tente enviar os prints na aba 'Fotos / Prints' ou adicione uma observação sobre o produto."
            }

        scripts = balance_script_scenes(gemini_res["scripts"])
        if not product_name and gemini_res.get("product_name"):
            product_name = str(gemini_res["product_name"]).strip()
        if not product_name:
            product_name = "Produto Identificado"

        return {
            "success": True,
            "ai_engine_active": True,
            "mode": "video",
            "product_name": product_name,
            "is_generic_product": False,
            "visual_details": gemini_res.get("visual_details", ""),
            "agent_xray": gemini_res.get("agent_xray", {}),
            "video_format": req.video_format,
            "scripts": scripts
        }

    return {
        "success": False,
        "error": "INVALID_MODE",
        "message": "Use o modo 'photos' (Fotos/Prints + Descrição) ou 'video' (Subir Vídeo .mp4)."
    }


@router.post("/refine-with-agent")
async def refine_with_agent(req: AgentRefineRequest):
    """
    Chat Conversacional com o Agente Diretor:
    Permite ao usuário pedir qualquer alteração em linguagem natural (ex: 'mostre mais a parte de trás e o bolso',
    'deixe os movimentos mais rápidos', 'mude o gancho para promoção') e o Agente reescreve os roteiros na hora!
    """
    user_api_key = (req.api_key or "").strip() or None
    if not has_configured_api_key(user_api_key):
        return {
            "success": False,
            "error": "API_KEY_REQUIRED",
            "message": "Conecte sua Chave Gemini no topo para conversar e pedir ajustes ao Agente!"
        }

    system_prompt = """Você é o Agente Diretor Sênior de Criação de Vídeos para TikTok Shop Brasil.
O usuário está revisando os 3 roteiros gerados e enviou um COMANDO DE AJUSTE para você.
Sua missão:
1. Obedeça 100% ao pedido do usuário (seja sobre mostrar detalhes da parte de trás, puxar o bolso e a costura em close-up rápido com uma das mãos, mudar o tom da fala, destacar elasticidade ou alterar a oferta).
2. REGRA DO CELULAR FIXO E TELA LIMPA: Em 'movements_summary', garanta que o celular permaneça 100% fixo e imóvel na mesma posição cobrindo o rosto durante todo o vídeo (mesmo quando aproximar para mostrar a roupa mais de perto), e que NÃO apareça nenhum texto ou ícone na tela.
3. REGRA DE EQUILÍBRIO DE PALAVRAS: O texto da Cena 2 ('scene_2_text') DEVE ter EXATAMENTE a mesma quantidade de palavras da Cena 1 ('scene_1_text').
4. Escreva uma resposta curta e amigável em 'agent_reply' confirmando o que você alterou.

Responda SEMPRE em JSON rigoroso:
{
  "agent_reply": "Pronto! Ajustei os 3 roteiros para...",
  "visual_details": "Descrição técnica atualizada em inglês dos detalhes visuais e movimentos solicitados para o Google Flow",
  "scripts": [
    {
      "id": 1,
      "angle_title": "...",
      "target_emotion": "...",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    },
    {
      "id": 2,
      "angle_title": "...",
      "target_emotion": "...",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    },
    {
      "id": 3,
      "angle_title": "...",
      "target_emotion": "...",
      "scene_1_text": "...",
      "scene_2_text": "...",
      "full_audio": "...",
      "movements_summary": "..."
    }
  ]
}"""

    user_prompt = f"""Produto: {req.product_name}
Formato: {req.video_format}
Raio-X Atual do Produto: {req.agent_xray}
Roteiros Atuais: {req.current_scripts}

COMANDO DO USUÁRIO PARA O AGENTE:
"{req.user_command}"

Aplique o ajuste solicitado agora mesmo nos 3 roteiros e nos movimentos, mantendo a mesma quantidade de palavras na Cena 1 e Cena 2."""

    res = run_llm_json(system_prompt, user_prompt, api_key=user_api_key)
    if not res or "scripts" not in res:
        return {
            "success": False,
            "message": "O Agente não conseguiu processar o ajuste no momento. Tente novamente."
        }

    balanced = balance_script_scenes(res["scripts"])
    return {
        "success": True,
        "agent_reply": res.get("agent_reply", "Roteiros e movimentos atualizados conforme seu pedido!"),
        "visual_details": res.get("visual_details", ""),
        "scripts": balanced
    }


@router.post("/generate-flow-prompt")
def generate_flow_prompt(req: GenerateFlowPromptRequest):
    """
    Passo 4: Gera os prompts masters prontos para enviar no Flow:
    1. Prompt de Imagem: Coloca a modelo ruiva de referência no ambiente selecionado vestindo o produto.
    2. Prompts de Vídeo: Animação Cena 1 (0-10s) e Cena 2 (10-20s com CTA apontando para o carrinho).
    3. Áudio da Locução em PT-BR para narração.
    """
    env_id = req.environment_id
    script = req.script_data
    prod = (req.product_name or "Look Feminino Elegante").strip()
    xray = req.agent_xray or {}
    features_list = xray.get("detected_features") or []
    fabric_info = xray.get("fabric_and_finish") or ""
    raw_movements = script.get("movements_summary") or ""
    custom_movements = re.sub(r"360\s*(?:graus|°|degrees)?", "meio giro lateral (45° a 75°)", raw_movements, flags=re.IGNORECASE)
    custom_movements = re.sub(r"giro\s+completo", "meio giro lateral", custom_movements, flags=re.IGNORECASE)
    garment_spec = f"{prod} ({fabric_info}; {', '.join(features_list[:4])})" if (fabric_info or features_list) else prod

    # Especificação exata da iluminação e fundo na reflexão do espelho
    if env_id == "ambiente_1":
        env_title = "Ambiente 1: Provador de Boutique (Fiel à Foto de Referência)"
        env_bg_en = "Behind her in the reflection is the interior of luxury boutique fitting room stalls: solid white divider stall partitions, matte black privacy curtain, and clothes neatly hung on wooden wall racks. Ultra-glossy polished black porcelain tile floor with realistic floor reflections beneath her white sneakers. Clean diffuse boutique overhead lighting, zero harsh shadows. Solid boutique room walls behind her back, with absolutely NO mirror behind the model."
    elif env_id == "ambiente_2":
        env_title = "Ambiente 2: Quarto Acolhedor (Fiel à Foto 'Ambiente quarto.jpeg')"
        env_bg_en = "Behind her in the reflection is the cozy modern bedroom interior: bed dressed with a bright sunflower patterned quilt and sunflower pillows, plush shaggy cream rug on warm wooden plank floor, bedside nightstand with a warm glowing lamp, solid room wall behind the bed with NO mirror behind the model. Warm cozy ambient bedroom lighting."
    else:
        env_title = "Ambiente 3: Hall de Entrada com Espelho Orgânico LED"
        env_bg_en = "Behind her in the reflection is a modern apartment entryway with sleek console table, contemporary solid accent wall, soft warm indirect ambient glow with NO mirror behind the model."

    fixed_phone_and_clean_screen_rule = "STRICT HALF-TURN & NO FLOATING PHONE RULE: The model's fingers and hand MUST remain physically gripping and wrapped tightly around the white smartphone at all times, naturally connected to her wrist, forearm, and shoulder — the smartphone MUST NEVER float in mid-air and MUST NEVER detach from her fingers. She performs ONLY a natural half-turn / side-to-side hip and torso pivot (45 to 75 degrees) to show the side and back of the outfit — NEVER a 360-degree spin and NEVER turning her head/phone away from the mirror. Even when stepping closer to the mirror to show garment details up close, her hand holds the smartphone fixed in the exact same position in front of her upper face/eyes. ABSOLUTELY NO text, NO captions, NO subtitles, NO UI icons, and NO graphics on the screen (100% clean raw camera frame)."

    # Prompt 1: Imagem da Modelo no Ambiente Selecionado Vestindo o Produto (Ângulo 1: Frente Corpo Inteiro)
    model_image_prompt = f"""Vertical 9:16 point-of-view mirror selfie photograph (FRONT FULL-BODY ANGLE). The entire image frame is the direct front reflection inside a clean flat full-length wall mirror. We see the direct front reflection of a curvy copper-ginger red-haired woman with soft natural shoulder-length waves. She is looking directly forward into the mirror, her hand and fingers firmly gripping a white smartphone fixed and stationary at face level so the white phone body directly covers and occludes her upper face and eyes (phone physically held in her hand, never floating). She is wearing {garment_spec}, delicate thin gold chain necklace with small pendant, classic gold link watch on right wrist, thin gold bracelet on left wrist, clean white sneakers. {env_bg_en}. {fixed_phone_and_clean_screen_rule}. Strict composition rule: The camera view is the mirror reflection itself; behind her back is the solid room interior only, absolutely NO secondary mirror behind her. Ultra-detailed 8K photorealism, high-end commercial fashion lookbook, pristine fabric texture and exact garment characteristics of {garment_spec}."""

    # 4 Ângulos da Modelo Vestindo o Look (Prontos para Gerar & Baixar ou Colar no Google Flow)
    angle_closeup_prompt = f"""Vertical 9:16 point-of-view mirror selfie photograph (HIGH-DEFINITION CLOSE-UP DETAIL ANGLE). Direct mirror reflection POV of the curvy copper-ginger red-haired woman stepped closer to the mirror glass to show the outfit up close. Her hand and fingers are firmly gripping the white smartphone completely fixed in the exact same position covering her upper face/eyes (phone physically connected to her hand and arm, never floating, never lowered). Wearing {garment_spec}, gold link watch on right wrist. Her free hand is gently pulling the pocket opening and reinforced fabric seam of {garment_spec} outward to clearly demonstrate fabric elasticity, high-density weave, and impeccable stitching up close. {env_bg_en}. {fixed_phone_and_clean_screen_rule}. Strictly preserve 100% the exact garment details, pocket design, and fabric color of {garment_spec}. Ultra-detailed 8K photorealism."""

    angle_back_prompt = f"""Vertical 9:16 point-of-view mirror selfie photograph (NATURAL HALF-TURN 75-DEGREE HIP & BACK ANGLE IN MIRROR). Direct mirror reflection POV showing the curvy copper-ginger red-haired woman in a natural half-turn (75-degree hip and torso pivot): her upper torso and head remain angled toward the mirror so her hand and fingers physically grip the white smartphone fixed in front of her upper face (natural arm anatomy, phone NEVER floating), while her hips and lower back pivot in a half-turn to clearly showcase the back design, rear pockets, back waistband fit, rear seams, and flattering silhouette of {garment_spec} without any distortion. Wearing {garment_spec}, white sneakers, gold watch. {env_bg_en}. {fixed_phone_and_clean_screen_rule}. Strictly NO secondary mirror behind her. Ultra-detailed 8K photorealism."""

    angle_side_prompt = f"""Vertical 9:16 point-of-view mirror selfie photograph (45-DEGREE SIDE PROFILE HALF-TURN ANGLE). Direct mirror reflection POV of the curvy copper-ginger red-haired woman pivoted in a dynamic 45-degree side profile half-turn in front of the full-length mirror, her hand and fingers firmly gripping the white smartphone fixed in the exact same position covering her upper face (never floating), free hand gently pulling the side pocket/seam of {garment_spec}. Highlighting the lateral drape, side seam, pocket depth, and smooth non-marking fit of {garment_spec}. {env_bg_en}. {fixed_phone_and_clean_screen_rule}. Ultra-detailed 8K photorealism."""

    model_angles = [
        {
            "id": "front",
            "title": "Ângulo 1: Frente (Corpo Inteiro no Espelho)",
            "badge": "Cena Inicial",
            "subtitle": "Modelo ruiva de frente com mão firme no celular fixo vestindo o look completo",
            "prompt_en": model_image_prompt
        },
        {
            "id": "closeup",
            "title": "Ângulo 2: Close-up Detalhes (Mão Puxando Bolso e Costura)",
            "badge": "Detalhe de Perto",
            "subtitle": "Mais de perto com mão firme no celular fixo e mão livre puxando bolso/costura",
            "prompt_en": angle_closeup_prompt
        },
        {
            "id": "back",
            "title": "Ângulo 3: Meio Giro Costas (75° Mostrando Parte de Trás & Bolso)",
            "badge": "Meio Giro Traseiro",
            "subtitle": "Meio giro natural (sem celular flutuando) mostrando parte de trás e bolso traseiro",
            "prompt_en": angle_back_prompt
        },
        {
            "id": "side",
            "title": "Ângulo 4: Meio Giro Perfil 45° (Silhueta & Modelagem)",
            "badge": "Lateral 45°",
            "subtitle": "Meio giro lateral com mão firme no celular destacando caimento que não marca",
            "prompt_en": angle_side_prompt
        }
    ]

    scene_1_audio = script.get("scene_1_text") or ""
    scene_2_audio = script.get("scene_2_text") or ""
    full_audio = script.get("full_audio") or ""
    narration_mode = (req.narration_mode or "separate").lower()

    # Prompts de Vídeo - Modo 1: Narração Separada em Outro Site (CapCut / ElevenLabs)
    # Estilo TIME-LAPSE ACELERADO com cortes rápidos (0.8s-1.2s), APENAS MEIO GIRO (sem celular flutuando) e zero texto/ícones na tela
    scene_1_prompt_separate = f"""Vertical 9:16 mirror selfie POV video animation, Scene 1 (0 to 10 seconds). {fixed_phone_and_clean_screen_rule} Fast-paced time-lapse style speed-ramped fashion try-on (1.3x accelerated time-lapse feel) with ultra-fast jump cuts every 0.8s to 1.2s showcasing multiple angles of the outfit. Direct reflection in a full-length wall mirror (solid room interior behind model, strictly NO mirror behind her back). Curvy copper-ginger red-haired model with natural shoulder waves, her hand and fingers firmly gripping the white smartphone completely fixed and stationary in the exact same position in front of her upper face/eyes (phone physically connected to her hand and arm, never floating, even when showing the outfit closer up). Wearing delicate gold chain necklace, gold link watch on right wrist, thin gold bracelet on left wrist, white sneakers. Wearing {garment_spec}. {env_bg_en}. Rapid time-lapse sequence of dynamic movements ({custom_movements}): Cut 1 (Front Angle): model steps briskly forward toward mirror keeping her hand firmly gripping the phone in place; Cut 2 (High-Definition Close-Up Shot): model moves closer to mirror keeping her hand holding the smartphone in the exact same fixed position on her face while her free hand gently pulls the pocket and fabric seam outward demonstrating extreme elasticity and reinforced stitching; Cut 3 (45-Degree Side Half-Turn): swift 45-degree hip pivot gliding free hand along waistline; Cut 4 (75-Degree Back Half-Turn ONLY — NO 360 spin): energetic half-turn of hips and torso showing rear pockets, back seams, and flattering fit while her hand keeps holding the phone fixed at her face. Strict visual consistency: always preserve 100% the exact garment characteristics, pockets, stitching, and cut of {garment_spec} across all fast cuts. Ultra-realistic 8K photorealism, zero text or icons on screen."""

    scene_2_prompt_separate = f"""Vertical 9:16 mirror selfie POV video animation, Scene 2 (10 to 20 seconds). {fixed_phone_and_clean_screen_rule} Fast-paced time-lapse style speed-ramped editing (1.3x accelerated time-lapse feel) with rapid jump cuts every 0.8s to 1.2s showing multiple angles. Direct mirror reflection POV (solid room behind the model, NO mirror behind her back). Curvy copper-ginger red-haired model with her hand and fingers always firmly gripping the white smartphone completely fixed and stationary in the exact same position in front of her face (never floating, even when showing garment details up close). Wearing gold watch on right wrist, delicate gold jewelry, white sneakers. Wearing {garment_spec}. {env_bg_en}. Rapid continuous multi-angle actions ({custom_movements}): model steps close to mirror in high-definition macro detail keeping her hand holding the phone fixed in the exact same position, uses her free hand to gently pull and snap the pocket and side seam proving elasticity and reinforced finish; performs ONLY a brisk half-turn (45 to 75 degrees side-to-side hip pivot, NO 360 spin) to exhibit rear details, pockets, and silhouette; sweeps copper hair back over shoulder. At the 9s to 10s mark, extends index finger of her free hand pointing straight down firmly toward the bottom of the screen (cart CTA button). Strict garment consistency: always retains exact clothing design, pockets, back details, and seams of {garment_spec}. Ultra-realistic 8K photorealism, zero text or icons on screen."""

    # Prompts de Vídeo - Modo 2: Narração Embutida Diretamente no Vídeo Gerado pelo Flow
    scene_1_prompt_embedded = f"""Vertical 9:16 mirror selfie POV video animation with synchronized dialogue audio, Scene 1 (0 to 10 seconds). {fixed_phone_and_clean_screen_rule} Fast-paced time-lapse style speed-ramped fashion try-on (1.3x accelerated time-lapse feel) with ultra-fast jump cuts every 0.8s to 1.2s showcasing multiple angles of the outfit. Direct mirror reflection POV (solid room interior behind model, strictly NO mirror behind her back). Curvy copper-ginger red-haired model with her hand and fingers always firmly gripping the white smartphone completely fixed and stationary in the exact same position in front of her upper face/eyes (phone physically connected to her hand and arm, never floating, even when showing the outfit closer up). Wearing gold jewelry, gold watch on right wrist, white sneakers. Wearing {garment_spec}. {env_bg_en}. Rapid time-lapse sequence of dynamic movements ({custom_movements}): Cut 1 (Front Angle): model steps briskly forward keeping her hand holding the phone locked in place; Cut 2 (High-Definition Close-Up Shot): model moves closer to mirror keeping her hand gripping the smartphone in the exact same fixed position on her face while her free hand gently pulls the pocket and fabric seam demonstrating elasticity and reinforced stitching; Cut 3 (Side & Back Half-Turn ONLY — NO 360 spin): swift 45-to-75 degree half-turn hip pivot showcasing rear pockets, back details, and fit while keeping the phone firmly held in her hand. Spoken dialogue in natural conversational Brazilian Portuguese (PT-BR) audio track: "{scene_1_audio}". Her lips and face remain occluded behind the stationary white smartphone. Strict garment consistency: always preserves exact clothing details, pockets, back details, and seams of {garment_spec}. Ultra-realistic 8K photorealism, zero text or icons on screen."""

    scene_2_prompt_embedded = f"""Vertical 9:16 mirror selfie POV video animation with synchronized dialogue audio, Scene 2 (10 to 20 seconds). {fixed_phone_and_clean_screen_rule} Fast-paced time-lapse style speed-ramped editing (1.3x accelerated time-lapse feel) with rapid jump cuts every 0.8s to 1.2s showing multiple angles. Direct mirror reflection POV (solid room behind model, strictly NO mirror behind her back). Curvy copper-ginger red-haired model with her hand and fingers always firmly gripping the white smartphone completely fixed and stationary in the exact same position in front of her face (never floating, even when showing garment details up close). Wearing gold watch on right wrist, delicate gold jewelry, white sneakers. Wearing {garment_spec}. {env_bg_en}. Rapid continuous multi-angle actions ({custom_movements}): model steps close to mirror in high-definition macro detail keeping her hand gripping the phone fixed in the exact same position, uses her free hand to gently pull and snap the pocket and side seam proving elasticity and reinforced finish; performs ONLY a brisk half-turn (45 to 75 degrees side-to-side hip pivot, NO 360 spin) to exhibit rear details, pockets, and silhouette; sweeps hair back over shoulder. At the 9s to 10s mark, extends index finger of her free hand pointing straight down firmly toward the bottom of the screen (cart CTA button). Spoken dialogue in natural conversational Brazilian Portuguese (PT-BR) audio track: "{scene_2_audio}". Strict garment consistency: always retains exact clothing design, pockets, back details, and seams of {garment_spec}. Ultra-realistic 8K photorealism, zero text or icons on screen."""

    # Seleção dos prompts ativos de acordo com a opção escolhida
    if narration_mode == "embedded":
        active_scene_1 = scene_1_prompt_embedded
        active_scene_2 = scene_2_prompt_embedded
        flow_instructions = f"""INSTRUÇÕES PARA O GOOGLE FLOW (MODO NARRAÇÃO EMBUTIDA + MÚLTIPLOS ÂNGULOS):
1. Gere e baixe os 4 Ângulos da sua Modelo Ruiva vestindo o look no 'Estúdio de Imagens Multi-Ângulos' abaixo (ou copie os prompts dos 4 ângulos).
2. Suba as imagens dos ângulos (Frente, Close-up Bolso/Costura, Costas e Perfil) no Google Flow.
3. Use o Prompt da Cena 1 e da Cena 2 para animar os ângulos em ritmo Time-Lapse com cortes rápidos e fala em português!"""
    else:
        active_scene_1 = scene_1_prompt_separate
        active_scene_2 = scene_2_prompt_separate
        flow_instructions = f"""INSTRUÇÕES PARA O GOOGLE FLOW (MODO NARRAÇÃO SEPARADA + MÚLTIPLOS ÂNGULOS):
1. Gere e baixe os 4 Ângulos da sua Modelo Ruiva vestindo o look no 'Estúdio de Imagens Multi-Ângulos' abaixo (Frente, Close-up Bolso/Costura, Costas e Perfil).
2. Suba as imagens geradas dos ângulos como frames/referências no Google Flow junto com os Prompts das Cenas 1 e 2 (estilo Time-Lapse rápido).
3. No CapCut (ou ElevenLabs), gere a locução em PT-BR com o 'Áudio da Locução' abaixo (Voz Jéssica a 1.15x-1.20x) e sincronize!"""

    supabase_saved = save_generation_to_supabase({
        "product_name": prod,
        "video_format": req.video_format,
        "environment_id": env_id,
        "narration_mode": narration_mode,
        "model_image_prompt": model_image_prompt,
        "scene_1_prompt": active_scene_1,
        "scene_2_prompt": active_scene_2,
        "audio_script": full_audio
    })

    return {
        "success": True,
        "supabase_saved": supabase_saved,
        "product_name": prod,
        "environment_title": env_title,
        "narration_mode": narration_mode,
        "model_image_prompt": model_image_prompt,
        "model_angles": model_angles,
        "scene_1_prompt": active_scene_1,
        "scene_2_prompt": active_scene_2,
        "scene_1_prompt_separate": scene_1_prompt_separate,
        "scene_2_prompt_separate": scene_2_prompt_separate,
        "scene_1_prompt_embedded": scene_1_prompt_embedded,
        "scene_2_prompt_embedded": scene_2_prompt_embedded,
        "audio_script": full_audio,
        "scene_1_audio": scene_1_audio,
        "scene_2_audio": scene_2_audio,
        "movements_direction": script.get("movements_summary") or "",
        "flow_instructions": flow_instructions
    }


@router.post("/generate-angle-image")
def generate_angle_image_endpoint(req: GenerateAngleImageRequest):
    """
    Gera a imagem real da modelo ruiva salva vestindo a roupa escaneada no ângulo escolhido
    (Frente, Close-up Bolso/Costura, Costas ou Perfil Lateral) usando o Gemini Image Generator.
    """
    ref_images = _load_saved_model_reference_b64(req.environment_id)
    if req.product_images:
        for p_img in req.product_images[:2]:
            if p_img.get("data"):
                ref_images.append({
                    "data": p_img["data"],
                    "mime_type": p_img.get("mime_type", "image/jpeg")
                })

    full_instruction = f"""Using the first reference image (the curvy copper-ginger red-haired woman holding a white smartphone in a mirror selfie) and the attached product reference image(s), generate a photorealistic 9:16 vertical mirror selfie image of this exact same red-haired woman wearing this exact garment with 100% fidelity to its color, fabric, pockets, and seams. Angle requested: {req.angle_title}. Detailed scene description: {req.angle_prompt}"""

    res = generate_gemini_image(
        prompt_text=full_instruction,
        reference_images=ref_images,
        api_key=req.api_key
    )
    return res


class GenerateTTSRequest(BaseModel):
    text: str
    voice_name: str = "Aoede" # Aoede, Kore, Leda, Zephyr
    tone_style: str = "animada" # animada, amiga, urgente
    api_key: Optional[str] = None


class GenerateColorVariationsRequest(BaseModel):
    colors_text: str # Ex: "Preto, Vinho, Bege"
    product_name: str
    environment_id: str = "ambiente_1"
    agent_xray: Optional[Dict[str, Any]] = None


@router.post("/generate-tts-audio")
def generate_tts_audio_endpoint(req: GenerateTTSRequest):
    """
    Item 1: Gera a locução em Português Brasileiro (PT-BR) direto pelo Gemini TTS
    e retorna um arquivo .WAV pronto para ouvir no player e baixar.
    """
    return generate_gemini_tts_wav(
        text_pt=req.text,
        voice_name=req.voice_name,
        tone_style=req.tone_style,
        api_key=req.api_key
    )


@router.post("/generate-color-variations")
def generate_color_variations_endpoint(req: GenerateColorVariationsRequest):
    """
    Item 2: Gera prompts de imagem da mesma modelo ruiva com celular 100% fixo
    vestindo exatamente a mesma roupa em múltiplas variações de cores (para o corte 'Comprei logo 3 cores').
    """
    raw_colors = [c.strip() for c in re.split(r"[,;/\n]+", req.colors_text or "") if c.strip()]
    if not raw_colors:
        raw_colors = ["Preto", "Vinho", "Bege"]

    prod = (req.product_name or "Look Feminino").strip()
    xray = req.agent_xray or {}
    features_list = xray.get("detected_features") or []
    fabric_info = xray.get("fabric_and_finish") or ""
    garment_spec = f"{prod} ({fabric_info}; {', '.join(features_list[:4])})" if (fabric_info or features_list) else prod

    env_id = req.environment_id
    if env_id == "ambiente_1":
        env_bg_en = "Behind her in the reflection is the interior of luxury boutique fitting room stalls: solid white divider stall partitions, matte black privacy curtain, and clothes neatly hung on wooden wall racks. Ultra-glossy polished black porcelain tile floor with realistic floor reflections beneath her white sneakers. Clean diffuse boutique overhead lighting, zero harsh shadows. Solid boutique room walls behind her back, with absolutely NO mirror behind the model."
    elif env_id == "ambiente_2":
        env_bg_en = "Behind her in the reflection is the cozy modern bedroom interior: bed dressed with a bright sunflower patterned quilt and sunflower pillows, plush shaggy cream rug on warm wooden plank floor, bedside nightstand with a warm glowing lamp, solid room wall behind the bed with NO mirror behind the model. Warm cozy ambient bedroom lighting."
    else:
        env_bg_en = "Behind her in the reflection is a modern apartment entryway with sleek console table, contemporary solid accent wall, soft warm indirect ambient glow with NO mirror behind the model."

    fixed_rule = "STRICT FIXED SMARTPHONE & CLEAN SCREEN RULE: The model MUST hold the white smartphone completely fixed, rigid, and stationary in the exact same position in front of her upper face/eyes with one hand. ABSOLUTELY NO text, NO captions, NO UI icons, and NO graphics on the screen."

    color_angles = []
    for idx, color_name in enumerate(raw_colors[:6], start=1):
        c_id = f"color_{idx}"
        prompt_en = f"""Vertical 9:16 point-of-view mirror selfie photograph (COLOR VARIATION #{idx}: {color_name.upper()}). Direct front reflection inside a clean flat full-length wall mirror of the curvy copper-ginger red-haired woman with soft natural shoulder-length waves. She is holding the white smartphone completely fixed and stationary in the exact same position covering her upper face and eyes with one hand. She is wearing {garment_spec} specifically in the color {color_name.upper()}, preserving 100% the exact same garment design, pockets, seams, fabric texture, and silhouette — ONLY the garment color changes to {color_name.upper()}. Wearing delicate gold necklace, gold watch on right wrist, thin gold bracelet on left wrist, white sneakers. {env_bg_en}. {fixed_rule}. Ultra-detailed 8K photorealism."""
        color_angles.append({
            "id": c_id,
            "title": f"Cor {idx}: {color_name.title()} (Mesma Pose & Celular Fixo)",
            "badge": f"Cor {idx}",
            "subtitle": f"Modelo ruiva na mesma posição vestindo {prod} na cor {color_name.title()} para corte rápido",
            "prompt_en": prompt_en
        })

    return {
        "success": True,
        "colors": raw_colors[:6],
        "color_angles": color_angles
    }


@router.get("/history")
def get_saved_history():
    """
    Item 5: Retorna os últimos criativos salvos no Supabase para a Biblioteca de Produtos Campeões.
    """
    from backend.database import SUPABASE_URL, SUPABASE_KEY
    if not SUPABASE_URL or not SUPABASE_KEY or not httpx:
        return {"success": True, "items": []}
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/wizard_generations?select=*&order=created_at.desc&limit=20",
            headers=headers,
            timeout=5.0
        )
        if resp.status_code == 200:
            return {"success": True, "items": resp.json()}
    except Exception:
        pass
    return {"success": True, "items": []}
