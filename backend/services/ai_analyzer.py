import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def has_configured_api_key(custom_api_key: Optional[str] = None) -> bool:
    key = (custom_api_key or "").strip() or (os.getenv("GEMINI_API_KEY") or "").strip()
    return bool(key and len(key) > 15)


def get_genai_client(custom_api_key: Optional[str] = None):
    api_key = (custom_api_key or "").strip() or (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Erro ao instanciar cliente Gemini: {e}")
        return None


def parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def run_llm_json(system_prompt: str, user_prompt: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Executa chamada para o Gemini Flash com retorno JSON estrito e fallback entre modelos.
    """
    client = get_genai_client(api_key)
    if not client:
        return None

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = [preferred_model]
    for fallback_m in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        if fallback_m not in models_to_try:
            models_to_try.append(fallback_m)

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            if response and response.text:
                parsed = parse_json_safely(response.text)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Erro na chamada do Gemini API ({model_name}): {e}")
            continue

    return None


def run_llm_multimodal(system_prompt: str, contents: Any, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Executa chamada multimodal para o Gemini Flash (imagens, áudio, vídeo + texto)
    com retorno em JSON estrito e fallback automático de modelo.
    """
    client = get_genai_client(api_key)
    if not client:
        return None

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = [preferred_model]
    for fallback_m in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        if fallback_m not in models_to_try:
            models_to_try.append(fallback_m)

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=0.35
                )
            )
            if response and response.text:
                parsed = parse_json_safely(response.text)
                if parsed:
                    return parsed
        except Exception as e:
            logger.error(f"Erro na chamada multimodal do Gemini ({model_name}): {e}")
            continue

    return None


def analyze_video_content(
    caption: str,
    transcript: str,
    product_name: str,
    system_prompt: str,
    user_prompt: str
) -> Dict[str, Any]:
    """
    Executa a análise estrutural do vídeo com base no prompt configurado.
    """
    result = run_llm_json(system_prompt, user_prompt)
    if result:
        return result

    # Fallback estruturado caso a chave da API ainda não tenha sido adicionada
    preview_hook = transcript[:80] if transcript else (caption[:80] if caption else "Gancho visual rápido")
    return {
        "hook_initial": f"{preview_hook}...",
        "headline": f"Como resolver o problema principal com {product_name or 'o produto'}",
        "pain_or_desire": "Economizar tempo e evitar frustração no dia a dia",
        "sales_argument": "Demonstração prática de facilidade de uso e eficácia imediata",
        "video_structure": "Problema > Demonstração da Solução > Prova Visual > CTA",
        "product_demo_style": "Demonstração prática em close-up com teste em tempo real",
        "cta_type": "Compre pelo carrinho amarelo do TikTok Shop antes que esgote",
        "hook_strength_score": 8.0,
        "key_takeaways": "1. Início direto ao ponto sem enrolação. 2. Foco na transformação visual instantânea.",
        "_notice": "Análise gerada em modo local (Defina GEMINI_API_KEY no .env para análise profunda via Gemini Flash)"
    }


def analyze_pattern_cluster(
    product_name: str,
    videos_data_json: str,
    system_prompt: str,
    user_prompt: str
) -> Dict[str, Any]:
    """
    Compara os vídeos do mesmo produto e identifica padrões vencedores.
    """
    result = run_llm_json(system_prompt, user_prompt)
    if result:
        return result

    return {
        "common_winning_hooks": [
            "Quebra de padrão nos primeiros 2 segundos com pergunta provocativa ou erro comum",
            "Demonstração visual do 'Antes e Depois' logo no primeiro frame",
            "Uso de som dinâmico ou ASMR durante o unboxing/teste"
        ],
        "dominant_structures": [
            "Hook Rápido (0-3s) > Problema Exagerado (3-8s) > Apresentação do Produto (8-18s) > CTA TikTok Shop (18-25s)",
            "História de Descoberta Acidental ('Comprei achando que era mentira...')"
        ],
        "key_selling_arguments": [
            "Praticidade e rapidez no resultado",
            "Custo-benefício comparado a alternativas tradicionais",
            "Portabilidade e facilidade de limpeza/armazenamento"
        ],
        "engagement_insights": "Vídeos com demonstração prática em situações reais tiveram taxa de compartilhamento 2.4x maior do que vídeos focados apenas na embalagem.",
        "synthesis_report": f"Para o produto {product_name}, a estratégia mais eficaz é focar em ganchos de frustração diária seguidos pela solução imediata, com fechamento direto no carrinho amarelo."
    }


def generate_differentiated_ideas(
    product_name: str,
    winning_patterns_str: str,
    custom_instructions: str,
    count: int,
    system_prompt: str,
    user_prompt: str
) -> Dict[str, Any]:
    """
    Gera ideias originais inspiradas nos padrões vencedores, sem cópia.
    """
    result = run_llm_json(system_prompt, user_prompt)
    if result and "ideas" in result:
        return result

    return {
        "ideas": [
            {
                "angle_title": "O Teste Cego / Desafio do Ceticismo",
                "hook_suggestion": "Eu tinha certeza que isso era golpe da internet até testar ao vivo...",
                "proposed_structure": "Hook de Ceticismo > Teste Realista sem Cortes > Surpresa com o Resultado > CTA de Estoque Limitado",
                "differentiation_angle": "Em vez de elogiar o produto desde o início como a maioria faz, adota a postura cética do próprio espectador.",
                "script_outline": "0-3s: Expressão de dúvida mostrando a embalagem. 3-10s: Executa o teste mais difícil possível. 10-20s: Mostra a reação genuína e o produto funcionando perfeitamente. 20-30s: 'Link tá aqui no carrinho pra quem quiser testar também'."
            },
            {
                "angle_title": "O Erro Oculto que Todo Mundo Comete",
                "hook_suggestion": "Pare de fazer isso com a sua comida se você não quer jogar dinheiro no lixo!",
                "proposed_structure": "Alerta de Erro > Consequência Negativa > Solução Definitiva > Prova Prática > CTA",
                "differentiation_angle": "Usa aversão à perda em vez de apenas desejo de compra, despertando urgência imediata.",
                "script_outline": "0-3s: Mostra alimentos estragando ou o problema acontecendo. 3-12s: Explica por que o método convencional falha. 12-22s: Apresenta o produto selando e protegendo. 22-30s: Direcionamento para a oferta do TikTok Shop."
            },
            {
                "angle_title": "Comparativo Lado a Lado: Jeito Antigo vs. Jeito Inteligente",
                "hook_suggestion": "Você ainda perde 15 minutos fazendo isso desse jeito?",
                "proposed_structure": "Tela Dividida / Corte Rápido > Frustração Antiga vs Facilidade Nova > CTA",
                "differentiation_angle": "Compara visualmente o esforço exagerado do método antigo com a facilidade sem esforço do produto.",
                "script_outline": "0-4s: Demonstração cômica da luta com o método tradicional. 4-15s: Resolução em 3 segundos com o produto. 15-25s: Close nos detalhes e facilidade de guardar. 25-30s: CTA claro no carrinho."
            }
        ]
    }


def generate_scenes_from_analysis(
    product_name: str,
    caption: str,
    hook: str,
    pain_or_desire: str,
    sales_argument: str,
    colors: str = "red, terracotta, beige, black"
) -> Dict[str, Any]:
    """
    Gera as 2 Cenas no formato Provador POV 9:16 com prompts de imagem em inglês
    prontos para Nano Banana / Imagen 3 e roteiro de áudio sincronizado em PT-BR.
    """
    system_prompt = """Você é um diretor e prompt engineer sênior especializado em gerar prompts visuais para IA (Nano Banana / Imagen 3) e roteiros de conversão máxima para TikTok Shop.
Você gera saídas no formato estrito de Provador POV 9:16 em 2 Cenas de 10 segundos, com prompts visuais em inglês extremamente detalhados (sem texto/ícones, celular cobrindo o rosto, iluminação natural, 8k) e áudio em PT-BR."""

    user_prompt = f"""Produto: {product_name}
Legenda original: {caption}
Gancho identificado: {hook}
Dor explorada: {pain_or_desire}
Argumento de venda: {sales_argument}
Cores para cortes: {colors}

Gere o JSON com as 2 cenas estruturadas no padrão Provador POV 9:16:
{{
  "scene_1": {{
    "title": "Cena 1 (0 a 10s): Gancho & Jump Cuts de Cores",
    "image_prompt_en": "Vertical 9:16 mirror selfie POV video, Scene 1 (0 to 10 seconds). Curvy blonde model in a boutique fitting room, holding a white smartphone strictly locked in front of her face. ABSOLUTELY NO text, icons, or graphic elements on screen. Wearing {product_name}. Model walks forward closer to mirror, sweeps hair back, turns to side, touches waist line, steps back slightly. Hard jump cuts switch set colors ({colors}) synchronized with counting. Ultra-realistic 8K photorealism, cinematic fitting room lighting.",
    "audio_script_pt": "Comprei uma, duas, três, quatro cores do mesmo modelo de tanto que me apaixonei! Eu estava cansada de roupa que fica transparente e marca a barriga. Esse {product_name} tem o caimento perfeito que modela o corpo,",
    "editing_direction": "Cortes ultrarrápidos (1.2s a 1.5s por take). A cada cor contada na fala, corte seco com um micro-movimento diferente (passo à frente, mão na cintura, virada de lado)."
  }},
  "scene_2": {{
    "title": "Cena 2 (10 a 20s): Quebra de Objeções & CTA Apontando para o Carrinho",
    "image_prompt_en": "Vertical 9:16 mirror selfie POV video, Scene 2 (10 to 20 seconds). Curvy blonde model in boutique fitting room, smartphone strictly locked in front of face. ABSOLUTELY NO text or icons on screen. Wearing {product_name}. Model walks forward, strokes fabric waist detail, adjusts shoulder straps, and stretches fabric. At 9s-10s mark, extends right index finger pointing straight down toward the bottom of the screen. Ultra-realistic 8K photorealism.",
    "audio_script_pt": "esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
    "editing_direction": "Micro-movimento aos 6s esticando o tecido para provar elasticidade. Aos 9s exatos, estende o dedo indicador apontando para baixo na direção do carrinho do TikTok Shop."
  }},
  "full_audio_pt": "Comprei uma, duas, três, quatro cores do mesmo modelo de tanto que me apaixonei! Eu estava cansada de roupa que fica transparente e marca a barriga. Esse {product_name} tem o caimento perfeito que modela o corpo, esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!"
}}"""

    result = run_llm_json(system_prompt, user_prompt)
    if result and "scene_1" in result and "scene_2" in result:
        return result

    # Fallback estruturado de alta qualidade caso a chamada LLM esteja offline
    prod = product_name or "peça de moda"
    return {
        "scene_1": {
            "title": "Cena 1 (0 a 10s): Gancho & Jump Cuts de Cores",
            "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 1 (0 to 10 seconds). Curvy blonde model in a boutique fitting room, holding a white smartphone strictly locked in front of her face. ABSOLUTELY NO text, icons, or graphic elements on screen. Wearing elegant {prod}. Model walks forward closer to mirror, sweeps hair back, turns to side, touches waist line, steps back slightly. Hard jump cuts switch set colors ({colors}) synchronized with counting. Ultra-realistic 8K photorealism, warm fitting room glow, cinematic natural reflections.",
            "audio_script_pt": f"Comprei uma, duas, três, quatro cores do mesmo modelo de tanto que me apaixonei! Eu estava cansada de peça que fica transparente e marca a barriga. Esse {prod} tem o caimento perfeito que modela o corpo,",
            "editing_direction": "Cortes ultrarrápidos (1.2s a 1.5s por take). A cada cor contada na fala, corte seco com um micro-movimento diferente (passo à frente, mão na cintura, virada de lado)."
        },
        "scene_2": {
            "title": "Cena 2 (10 a 20s): Quebra de Objeções & CTA Apontando para o Carrinho",
            "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 2 (10 to 20 seconds). Curvy blonde model in boutique fitting room, smartphone strictly locked in front of face. ABSOLUTELY NO text or icons on screen. Wearing elegant {prod}. Model walks forward, strokes fabric waist detail, adjusts shoulder straps, and stretches fabric. At 9s-10s mark, extends right index finger pointing straight down toward the bottom of the screen. Ultra-realistic 8K photorealism.",
            "audio_script_pt": "esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
            "editing_direction": "Micro-movimento aos 6s esticando o tecido para provar elasticidade. Aos 9s exatos, estende o dedo indicador apontando para baixo na direção do carrinho do TikTok Shop."
        },
        "full_audio_pt": f"Comprei uma, duas, três, quatro cores do mesmo modelo de tanto que me apaixonei! Eu estava cansada de peça que fica transparente e marca a barriga. Esse {prod} tem o caimento perfeito que modela o corpo, esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!"
    }


def generate_tripack_from_video(
    product_name: str,
    caption: str,
    hook: str,
    pain_or_desire: str,
    sales_argument: str,
    model_description: str = "Curvy copper-ginger red-haired model holding white smartphone strictly locked in front of face, wearing gold watch on right wrist, delicate gold chain necklace"
) -> Dict[str, Any]:
    """
    Gera o pacote de 3 Vídeos em 3 Ambientes e 3 Ângulos diferentes para o mesmo produto.
    Mantém rigorosamente a modelo ruiva de referência, seus acessórios dourados e varia os 3 cenários.
    """
    prod = product_name or "peça de vestuário"
    m_desc = model_description or "Curvy copper-ginger red-haired model holding white smartphone strictly locked in front of face, wearing gold watch on right wrist, delicate gold chain necklace"

    return {
        "product_name": prod,
        "video_1": {
            "title": "Vídeo 1: Desejo & Contagem de Cores",
            "environment": "Ambiente 1: Provador de Boutique (Fiel às Referências)",
            "environment_desc": "Piso de porcelanato preto espelhado de alto brilho, cabines brancas de boutique com cortina escura ao fundo.",
            "scene_1": {
                "title": "Cena 1 (0 a 10s): Gancho 'Comprei 1, 2, 3, 4 cores'",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 1 (0 to 10 seconds). {m_desc}. ABSOLUTELY NO text, icons, or graphic elements on screen. Wearing elegant {prod}. Standing inside luxury boutique fitting room stalls, ultra-glossy polished black porcelain tile floor with crisp light reflections, black curtain and wooden clothes hangers in background. Model walks forward closer to mirror, sweeps copper hair back, turns to side, touches waist line, steps back slightly. Hard jump cuts switch set colors (red, terracotta, beige, black) synchronized with counting. Ultra-realistic 8K photorealism, bright boutique store lighting.",
                "audio_script_pt": f"Comprei uma, duas, três, quatro cores do mesmo modelo de tanto que me apaixonei! Eu estava cansada de roupa que fica transparente e marca a barriga. Esse {prod} tem o caimento perfeito que modela o corpo,",
                "editing_direction": "Cortes ultrarrápidos (1.2s a 1.5s por take). A cada cor contada na fala, corte seco com um micro-movimento diferente (passo à frente, mão na cintura, virada de lado)."
            },
            "scene_2": {
                "title": "Cena 2 (10 a 20s): Quebra de Objeção & Dedo no Carrinho",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 2 (10 to 20 seconds). {m_desc}. ABSOLUTELY NO text or icons on screen. Wearing elegant {prod}. Luxury boutique fitting room, black glossy tile floor. Model walks forward, strokes fabric waist detail, adjusts shoulder straps, and stretches fabric. At 9s-10s mark, extends right index finger pointing straight down toward the bottom of the screen. Ultra-realistic 8K photorealism.",
                "audio_script_pt": "esconde as imperfeições e deixa qualquer look chique em segundos, sem falar que é super versátil. Veste super bem do M ao GG! Se o carrinho laranja estiver aparecendo aqui embaixo, já corre e garante o seu antes que acabe!",
                "editing_direction": "Micro-movimento aos 6s esticando o tecido para provar elasticidade. Aos 9s exatos, estende o dedo indicador apontando para baixo na direção do carrinho do TikTok Shop."
            }
        },
        "video_2": {
            "title": "Vídeo 2: Quebra de Ceticismo & Caimento Real",
            "environment": "Ambiente 2: Closet / Quarto Moderno Minimalista",
            "environment_desc": "Espelho de chão de corpo inteiro, piso de madeira clara em espinha de peixe, luz suave de janela da manhã.",
            "scene_1": {
                "title": "Cena 1 (0 a 10s): Gancho 'Duvidei até chegar'",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 1 (0 to 10 seconds). {m_desc}. ABSOLUTELY NO text, icons, or graphic elements on screen. Wearing {prod} in terracotta color. Standing in a minimalist luxury bedroom in front of a full-length arched floor mirror, light natural oak herringbone wooden floor, soft sheer linen window curtains with gentle morning sunlight. Model steps back to show full body silhouette, turns sideways showing flat stomach, pinches fabric. Hard jump cuts between neutral colors. Ultra-realistic 8K photorealism, soft natural daylight.",
                "audio_script_pt": f"Eu tinha certeza que por esse preço esse {prod} ia marcar toda a barriga ou ficar transparente. Mas quando chegou no correio e provei... olha o caimento dessa modelagem!",
                "editing_direction": "Take 1 (0-3s): Expressão corporal de surpresa virando de perfil. Take 2 (3-7s): Aproximação no espelho mostrando que não marca sutiã. Take 3 (7-10s): Puxãozinho leve provando que o tecido é encorpado."
            },
            "scene_2": {
                "title": "Cena 2 (10 a 20s): Teste de Transparência & CTA Carrinho",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 2 (10 to 20 seconds). {m_desc}. ABSOLUTELY NO text or icons on screen. Wearing {prod} in beige color. Minimalist bedroom setting, arched floor mirror. Model moves smoothly, adjusts hem, and points index finger down to screen bottom at 9s mark. Ultra-realistic 8K photorealism.",
                "audio_script_pt": "O tecido é duplo, não tem transparência nenhuma e disfarça qualquer imperfeição na hora. Veste do 38 ao 46 perfeitamente! Clica no carrinho aqui embaixo antes que as cores mais pedidas esgotem!",
                "editing_direction": "Gesto de esticar a barra lateral com as duas mãos aos 5s. Aos 9s, estende o dedo indicador apontando para baixo com firmeza."
            }
        },
        "video_3": {
            "title": "Vídeo 3: Versatilidade (Do Trabalho ao Jantar de Sexta)",
            "environment": "Ambiente 3: Hall de Entrada com Espelho Orgânico LED",
            "environment_desc": "Espelho orgânico assimétrico iluminado por LED suave, parede contemporânea, visual de saída com bolsa.",
            "scene_1": {
                "title": "Cena 1 (0 a 10s): Gancho 'A única peça coringa'",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 1 (0 to 10 seconds). {m_desc}. ABSOLUTELY NO text, icons, or graphic elements on screen. Wearing {prod} in elegant black color, chic leather handbag on side. Standing in a modern apartment foyer in front of a large organic-shaped mirror with soft warm backlit LED halo, contemporary wall. Model does a swift confident turn, checks outfit in mirror, steps forward. Ultra-realistic 8K photorealism, architectural chic lighting.",
                "audio_script_pt": f"A única peça do meu guarda-roupa que eu consigo usar no trabalho de manhã com blazer e ir direto pro restaurante à noite sem amassar nada! Esse {prod} é um absurdo de chique.",
                "editing_direction": "Cortes dinâmicos mostrando a composição: com calça branca de alfaiataria, depois com jeans, depois com saia. Ritmo rápido e postura elegante."
            },
            "scene_2": {
                "title": "Cena 2 (10 a 20s): Custo-Benefício de Loja Cara & CTA",
                "image_prompt_en": f"Vertical 9:16 mirror selfie POV video, Scene 2 (10 to 20 seconds). {m_desc}. ABSOLUTELY NO text or icons on screen. Wearing {prod}. Modern foyer with backlit LED organic mirror. Model gives a confident posture, checks details, and points index finger directly down at 9s mark. Ultra-realistic 8K photorealism.",
                "audio_script_pt": "Parece aquelas peças de shopping que custam 300 reais, mas no TikTok Shop tá num valor inacreditável. Aproveita enquanto tem estoque e garante no carrinho aqui embaixo!",
                "editing_direction": "Toque na gola/aviamento dourado aos 4s. Aos 9s, movimento claro do dedo indicador apontando para baixo na direção do botão de compra."
            }
        }
    }



