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
from backend.services.ai_analyzer import run_llm_json, run_llm_multimodal

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

class StartWizardRequest(BaseModel):
    mode: Optional[str] = "photos" # "photos", "video" ou "link"
    video_format: str = "UGC-POV" # UGC, POV, UGC-POV
    product_input: Optional[str] = None # Nome do Produto
    description: Optional[str] = None # Detalhes do produto, tecido, bolsos, zíper, oferta
    images: Optional[List[Dict[str, Any]]] = None # [{"data": "base64...", "mime_type": "image/jpeg"}]
    video_data: Optional[Dict[str, Any]] = None # {"data": "base64...", "mime_type": "video/mp4" ou "audio/wav"}
    frames: Optional[List[Dict[str, Any]]] = None # [{"data": "base64...", "mime_type": "image/jpeg"}]
    # Retrocompatibilidade com links antigos
    competitor_url: Optional[str] = None
    competitor_speech_or_caption: Optional[str] = None


class ExtractProductInfoRequest(BaseModel):
    url: str
    context_text: Optional[str] = None # Texto adicional opcional (legenda ou fala)


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
    Passo 1 & 2: Analisa a entrada escolhida pelo criador e gera 3 variações de roteiros:
    - Modo 'photos': Analisa 1 a 4 fotos/prints do produto + descrição via Gemini Vision.
    - Modo 'video': Analisa o arquivo de vídeo (.mp4) ou áudio do concorrente + frames.
    - Modo 'link' (legado): Fallback para link caso o usuário use retrocompatibilidade.
    """
    mode = req.mode or "photos"

    # =========================================================================
    # ESTRATÉGIA 1: FOTOS / PRINTS + TEXTO / DESCRIÇÃO (100% à prova de bloqueios)
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
        is_generic_product = False
        product_warning = None
        visual_details = ""
        scripts = None

        # Se temos imagens e o Gemini está disponível:
        if types and images_list:
            system_prompt = """Você é um diretor sênior de roteiros para TikTok Shop especializado em vídeos de alta conversão (formato Provador UGC-POV).
Sua missão:
1. Inspecione visualmente as fotos/prints do produto enviadas: analise tecido (canelado, linho, algodão, etc.), modelagem (caimento, fendas, corte), detalhes especiais (bolsos, zíperes, costuras reforçadas, botões, decote, caimento na cintura/quadril) e cores disponíveis.
2. Identifique ou confirme o nome comercial do produto com precisão.
3. Crie 3 variações de roteiros persuasivos adaptados para esse produto no TikTok Shop.
Cada roteiro deve conter:
- Cena 1 (0-10s): Gancho ultra dinâmico citando o produto e destacando os detalhes visuais reais vistos nas fotos.
- Cena 2 (10-20s): Quebra de objeção (não marca, tecido encorpado, não é transparente) e chamada para ação com a modelo estendendo o dedo indicador apontando firme para o carrinho aos 9-10s.
- movements_summary: Instruções detalhadas de movimentos rápidos da modelo no espelho (cobrindo o rosto com o celular), puxando suavemente o bolso, a costura e o tecido para provar elasticidade e acabamento.
Responda SEMPRE em JSON rigoroso:
{
  "product_name": "Nome Comercial do Produto",
  "visual_details": "Resumo dos detalhes reais vistos nas fotos (bolso, zíper, tecido, cores)",
  "scripts": [
    {
      "id": 1,
      "angle_title": "Desejo & Contagem de Cores",
      "target_emotion": "Desejo incontrolável de ter todas as cores",
      "scene_1_text": "texto da cena 1 citando o produto",
      "scene_2_text": "texto da cena 2 com CTA para o carrinho",
      "full_audio": "texto completo do áudio",
      "movements_summary": "direção de movimentos rápidos da modelo"
    }
  ]
}"""
            multimodal_contents = []
            for img in images_list[:4]:
                raw = safe_b64decode(img.get("data", ""))
                if raw:
                    mime = img.get("mime_type", "image/jpeg")
                    multimodal_contents.append(types.Part.from_bytes(data=raw, mime_type=mime))

            user_text = f"""Produto informado pelo criador: {product_name or 'Identifique detalhadamente pelas fotos'}
Descrição e especificações fornecidas: {desc or 'Observe as fotos anexadas'}
Formato do vídeo: {req.video_format}

Analise os detalhes visuais das fotos e gere os 3 roteiros completos com ritmo acelerado."""
            multimodal_contents.append(user_text)

            gemini_res = run_llm_multimodal(system_prompt, multimodal_contents)
            if gemini_res and "scripts" in gemini_res and len(gemini_res["scripts"]) >= 3:
                scripts = gemini_res["scripts"]
                if not product_name and gemini_res.get("product_name"):
                    product_name = str(gemini_res["product_name"]).strip()
                visual_details = gemini_res.get("visual_details", "")

        # Fallback inteligente se Gemini offline ou sem imagens
        if not scripts:
            if not product_name:
                product_name = extract_product_with_gemini(desc) or extract_product_with_regex(desc) or "Produto a Definir"
                if product_name == "Produto a Definir":
                    is_generic_product = True
                    product_warning = "Aviso: Nome do produto definido como 'Produto a Definir'. Recomendamos digitar o nome exato no campo de produto."
            scripts = build_category_scripts(product_name, req.video_format, raw_context=desc, visual_details=desc)

        return {
            "success": True,
            "mode": "photos",
            "product_name": product_name,
            "is_generic_product": is_generic_product,
            "product_warning": product_warning,
            "visual_details": visual_details,
            "video_format": req.video_format,
            "scripts": scripts
        }

    # =========================================================================
    # ESTRATÉGIA 2: SUBIR VÍDEO (.MP4) / ÁUDIO DO CONCORRENTE
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
        is_generic_product = False
        product_warning = None
        scripts = None

        if types and (video_info or frames_list):
            system_prompt = """Você é um diretor sênior de criação de vídeos de alta conversão para TikTok Shop.
Sua missão:
1. Analise o vídeo/áudio do concorrente e as cenas anexadas.
2. Transcreva a fala e extraia os ganchos mentais, ritmo, quebra de objeções e CTA usados no vídeo.
3. Identifique o produto demonstrado.
4. Modele esse ritmo e gere 3 novas variações de roteiros otimizadas para o produto do vendedor.
Cada roteiro deve conter:
- Cena 1 (0-10s): Gancho forte de alta retenção.
- Cena 2 (10-20s): Quebra de objeção e CTA com a modelo apontando para o carrinho aos 9-10s.
- movements_summary: Instruções de movimentos rápidos no espelho (estilo UGC-POV).
Responda SEMPRE em JSON rigoroso:
{
  "product_name": "Nome Comercial do Produto",
  "competitor_transcript": "Transcrição do que é falado no vídeo",
  "scripts": [
    {
      "id": 1,
      "angle_title": "Desejo & Contagem de Cores",
      "target_emotion": "Desejo incontrolável de ter todas as cores",
      "scene_1_text": "texto cena 1",
      "scene_2_text": "texto cena 2 com CTA para o carrinho",
      "full_audio": "áudio completo",
      "movements_summary": "direção de movimentos rápidos"
    }
  ]
}"""
            multimodal_contents = []
            if video_info:
                v_raw = safe_b64decode(video_info.get("data", ""))
                v_mime = video_info.get("mime_type", "video/mp4")
                if v_raw:
                    multimodal_contents.append(types.Part.from_bytes(data=v_raw, mime_type=v_mime))

            for frame in frames_list[:3]:
                f_raw = safe_b64decode(frame.get("data", ""))
                if f_raw:
                    multimodal_contents.append(types.Part.from_bytes(data=f_raw, mime_type=frame.get("mime_type", "image/jpeg")))

            user_text = f"""Produto do vendedor: {product_name or 'Identifique pelo vídeo'}
Observações e detalhes adicionais: {desc or user_caption or 'Nenhuma'}
Formato do vídeo: {req.video_format}

Analise o concorrente e gere 3 roteiros modelados com perfeição."""
            multimodal_contents.append(user_text)

            gemini_res = run_llm_multimodal(system_prompt, multimodal_contents)
            if gemini_res and "scripts" in gemini_res and len(gemini_res["scripts"]) >= 3:
                scripts = gemini_res["scripts"]
                if not product_name and gemini_res.get("product_name"):
                    product_name = str(gemini_res["product_name"]).strip()

        if not scripts:
            if not product_name:
                product_name = extract_product_with_gemini(user_caption or desc) or extract_product_with_regex(user_caption or desc) or "Produto a Definir"
                if product_name == "Produto a Definir":
                    is_generic_product = True
                    product_warning = "Aviso: Nome do produto definido provisoriamente como 'Produto a Definir'."
            scripts = build_category_scripts(product_name, req.video_format, raw_context=user_caption or desc, visual_details=desc)

        return {
            "success": True,
            "mode": "video",
            "product_name": product_name,
            "is_generic_product": is_generic_product,
            "product_warning": product_warning,
            "video_format": req.video_format,
            "scripts": scripts
        }

    # =========================================================================
    # ESTRATÉGIA LEGADA: LINK DO TIKTOK
    # =========================================================================
    comp_url = req.competitor_url.strip() if req.competitor_url else ""
    user_speech_caption = (req.competitor_speech_or_caption or "").strip()

    metadata_caption = ""
    if comp_url:
        meta_res = await extract_tiktok_metadata_fast(comp_url)
        if meta_res.get("success") and meta_res.get("text"):
            metadata_caption = meta_res["text"]

    caption = user_speech_caption or metadata_caption or ""
    if not caption:
        return {
            "success": False,
            "error": "TRANSCRIPTION_FAILED",
            "message": "O TikTok bloqueou a extração automática de áudio e legenda deste link. Recomendamos usar a aba '📸 Fotos / Prints + Texto' ou '📤 Subir Vídeo (.mp4)' para gerar instantaneamente sem bloqueios!"
        }

    product_name = ""
    is_generic_product = False
    product_warning = None

    if req.product_input and req.product_input.strip() and not req.product_input.strip().startswith("http"):
        product_name = req.product_input.strip()

    if not product_name or is_blacklisted_or_invalid(product_name):
        combined_context = f"Legenda/Título do Vídeo: {metadata_caption}\nFala/Roteiro: {user_speech_caption or caption}"
        gemini_prod = extract_product_with_gemini(combined_context)
        if gemini_prod and not is_blacklisted_or_invalid(gemini_prod):
            product_name = gemini_prod

    if not product_name or is_blacklisted_or_invalid(product_name):
        regex_prod = extract_product_with_regex(caption)
        if regex_prod and not is_blacklisted_or_invalid(regex_prod):
            product_name = regex_prod

    if not product_name or is_blacklisted_or_invalid(product_name):
        product_name = "Produto a Definir"
        is_generic_product = True
        product_warning = "Aviso: Nome definido provisoriamente como 'Produto a Definir'."

    scripts = build_category_scripts(product_name, req.video_format, raw_context=caption)

    return {
        "success": True,
        "mode": "link",
        "product_name": product_name,
        "is_generic_product": is_generic_product,
        "product_warning": product_warning,
        "caption_used": caption,
        "video_format": req.video_format,
        "scripts": scripts
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

    # Prompt 1: Imagem da Modelo no Ambiente Selecionado Vestindo o Produto (Keyframe Base)
    model_image_prompt = f"""Vertical 9:16 point-of-view mirror selfie photograph. The entire image frame is the direct front reflection inside a clean flat full-length wall mirror. We see the direct front reflection of a curvy copper-ginger red-haired woman with soft natural shoulder-length waves. She is looking directly forward into the mirror, holding a white smartphone with both hands at chest level, pointing its rear camera at the mirror glass so the white phone body directly covers and occludes her upper face and eyes. She is wearing {prod}, delicate thin gold chain necklace with small pendant, classic gold link watch on right wrist, thin gold bracelet on left wrist, clean white sneakers. {env_bg_en}. Strict composition rule: The camera view is the mirror reflection itself; the model is facing forward into the mirror; behind her back is the solid room interior only, absolutely NO secondary mirror behind her, NO dual mirrors, NO reflection of her back. Ultra-detailed 8K photorealism, high-end commercial fashion lookbook, pristine fabric texture and silhouette of {prod}, authentic floor reflections, clean diffuse lighting matching reference photo without harsh shadows."""

    scene_1_audio = script.get("scene_1_text") or ""
    scene_2_audio = script.get("scene_2_text") or ""
    full_audio = script.get("full_audio") or ""
    narration_mode = (req.narration_mode or "separate").lower()

    # Prompts de Vídeo - Modo 1: Narração Separada em Outro Site (CapCut / ElevenLabs)
    # Foco 100% no visual hiper-rápido de movimentos, aproximação em close-up, puxão de bolso/costura com uma das mãos e cortes sem ruído de voz no Flow
    scene_1_prompt_separate = f"""Vertical 9:16 mirror selfie POV video animation, Scene 1 (0 to 10 seconds). The camera viewpoint is the reflection in a full-length wall mirror, with the solid room interior behind the model (strictly NO mirror behind her back). High energy, ultra-fast dynamic commercial pacing. Curvy copper-ginger red-haired model with natural shoulder waves, holding white smartphone locked with one hand in front of face covering upper face and eyes. Wearing delicate gold chain necklace, gold link watch on right wrist, thin gold bracelet on left wrist, white sneakers. Wearing {prod}. {env_bg_en}. Continuous high-speed agile micro-movements: model immediately steps forward into a sharp close-up view near the mirror; with her free hand, she firmly pulls and stretches the pocket opening, waistband, and reinforced fabric seam outward demonstrating extreme elasticity and rapid bounce-back; rubs fingers across the fabric weave to highlight texture and high-density finish with zero transparency; glides her hand across the waist and hips; swiftly pivots 45 degrees and turns to showcase rear pockets, back seams, and 360-degree flattering fit. Ultra-fast jump cuts (every 1.2s to 1.5s) switching outfit color variations in rapid cadence. Strict visual consistency: perfectly preserve the exact garment details, pockets, stitching, and cut of {prod} without distortion. Ultra-realistic 8K photorealism, exact reference lighting with authentic floor reflections."""

    scene_2_prompt_separate = f"""Vertical 9:16 mirror selfie POV video animation, Scene 2 (10 to 20 seconds). Direct mirror reflection POV (solid room behind the model, NO mirror behind her back). Fast energetic commercial fashion TikTok. Curvy copper-ginger red-haired model holding white smartphone locked with one hand in front of face. Wearing gold watch on right wrist, delicate gold jewelry, white sneakers. Wearing {prod}. {env_bg_en}. Rapid continuous actions: model steps close to mirror in macro detail, uses her free hand to pull and snap the fabric at the pocket and side seam proving heavy fabric weight, elasticity, and flawless reinforced stitching; rubs hand down thigh and hip; briskly turns from front to profile and back to exhibit rear pockets and silhouette; sweeps hair back over shoulder. At the 9s to 10s mark, extends index finger pointing straight down firmly toward the bottom of the screen (cart CTA button). Strict garment consistency: retains exact clothing design, pockets, and seams of {prod}. Ultra-realistic 8K photorealism, exact diffuse boutique overhead lighting and dark floor reflections."""

    # Prompts de Vídeo - Modo 2: Narração Embutida Diretamente no Vídeo Gerado pelo Flow
    # Inclui o diálogo em áudio nativo em português brasileiro sincronizado com as ações visuais
    scene_1_prompt_embedded = f"""Vertical 9:16 mirror selfie POV video animation with synchronized dialogue audio, Scene 1 (0 to 10 seconds). Direct mirror reflection POV (solid room interior behind model, strictly NO mirror behind her back). High energy, ultra-fast dynamic commercial pacing. Curvy copper-ginger red-haired model holding white smartphone locked with one hand in front of face covering upper face and eyes. Wearing gold jewelry, gold watch on right wrist, white sneakers. Wearing {prod}. {env_bg_en}. Continuous high-speed agile micro-movements: model immediately steps forward into a sharp close-up view near the mirror; with her free hand, she firmly pulls and stretches the pocket opening, waistband, and reinforced fabric seam outward demonstrating extreme elasticity and rapid bounce-back; rubs fingers across the fabric weave to highlight texture and high-density finish with zero transparency; glides her hand across the waist and hips; swiftly pivots 45 degrees and turns to showcase rear pockets, back seams, and 360-degree flattering fit. Ultra-fast jump cuts (every 1.2s to 1.5s) switching outfit color variations. Spoken dialogue in natural conversational Brazilian Portuguese (PT-BR) audio track: "{scene_1_audio}". The model speaks with vibrant high-converting TikTok Shop cadence; her lips and face remain naturally occluded behind the white smartphone held in front of her face, ensuring natural acoustic delivery without lip-sync distortion. Strict garment consistency: preserves exact clothing details, pockets, and seams of {prod}. Ultra-realistic 8K photorealism, exact reference lighting with authentic floor reflections."""

    scene_2_prompt_embedded = f"""Vertical 9:16 mirror selfie POV video animation with synchronized dialogue audio, Scene 2 (10 to 20 seconds). Direct mirror reflection POV (solid room behind model, strictly NO mirror behind her back). Fast energetic commercial fashion TikTok. Curvy copper-ginger red-haired model holding white smartphone locked with one hand in front of face. Wearing gold watch on right wrist, delicate gold jewelry, white sneakers. Wearing {prod}. {env_bg_en}. Rapid continuous actions: model steps close to mirror in macro detail, uses her free hand to pull and snap the fabric at the pocket and side seam proving heavy fabric weight, elasticity, and flawless reinforced stitching; rubs hand down thigh and hip; briskly turns from front to profile and back to exhibit rear pockets and silhouette; sweeps hair back over shoulder. At the 9s to 10s mark, extends index finger pointing straight down firmly toward the bottom of the screen (cart CTA button). Spoken dialogue in natural conversational Brazilian Portuguese (PT-BR) audio track: "{scene_2_audio}". Persuasive call-to-action tone delivered in clear Brazilian Portuguese, concluding right as the finger points to the cart at 9-10 seconds. Strict garment consistency: retains exact clothing design, pockets, and seams of {prod}. Ultra-realistic 8K photorealism, exact diffuse boutique overhead lighting and dark floor reflections."""

    # Seleção dos prompts ativos de acordo com a opção escolhida
    if narration_mode == "embedded":
        active_scene_1 = scene_1_prompt_embedded
        active_scene_2 = scene_2_prompt_embedded
        flow_instructions = f"""INSTRUÇÕES PARA O GOOGLE FLOW (MODO NARRAÇÃO EMBUTIDA):
1. Faça o upload da imagem da sua modelo ruiva ('03_REFERENCIAS/Modelo e ambiente 1.jpeg') e do ambiente ({env_title}).
2. Gere o look inicial usando o 'Prompt de Imagem (Modelo Vestindo o Produto)'.
3. Use o Prompt da Cena 1 com Narração Embutida para gerar o primeiro clipe com áudio e fala nativa em português.
4. Use o Prompt da Cena 2 com Narração Embutida para gerar o segundo clipe com áudio e CTA apontando para o carrinho.
5. O Flow gerará vídeo + áudio diretamente em um único arquivo!"""
    else:
        active_scene_1 = scene_1_prompt_separate
        active_scene_2 = scene_2_prompt_separate
        flow_instructions = f"""INSTRUÇÕES PARA O GOOGLE FLOW (MODO NARRAÇÃO SEPARADA - RECOMENDADO):
1. Faça o upload da imagem da sua modelo ruiva ('03_REFERENCIAS/Modelo e ambiente 1.jpeg') e do ambiente ({env_title}).
2. Gere o look inicial usando o 'Prompt de Imagem (Modelo Vestindo o Produto)'.
3. Use os Prompts Visuais das Cenas 1 e 2 no Flow para gerar clipes com foco total em velocidade de movimentos e detalhes do tecido.
4. No CapCut (ou ElevenLabs / Fish Audio), gere a locução em PT-BR usando o 'Áudio da Locução' abaixo (Voz Jéssica a 1.15x).
5. Junte os vídeos com o áudio no CapCut para sincronia perfeita e máxima conversão de vendas."""

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
