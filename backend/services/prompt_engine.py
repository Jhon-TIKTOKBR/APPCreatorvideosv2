import re
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.models import PromptTemplate

DEFAULT_VIDEO_ANALYSIS_SYSTEM = """Você é um especialista sênior em engenharia reversa de criativos virais para TikTok Shop e Short-Form Video Marketing.
Sua missão é dissecar vídeos comerciais e de concorrentes com rigor cirúrgico, identificando a psicologia de vendas e as estruturas persuasivas utilizadas.
Responda SEMPRE em formato JSON estruturado estrito, sem textos adicionais fora do JSON."""

DEFAULT_VIDEO_ANALYSIS_TEMPLATE = """Analise o seguinte vídeo do TikTok Shop com base nos dados disponíveis:

TÍTULO / LEGENDA DO VÍDEO:
{{caption}}

TRANSCRIÇÃO DO ÁUDIO / ROTEIRO FALADO:
{{transcript}}

PRODUTO ASSOCIADO:
{{product_name}}

Identifique com precisão cirúrgica:
1. "hook_initial": Qual foi o gancho dos primeiros 3 segundos? (Texto exato, quebra de padrão ou gatilho visual/auditivo).
2. "headline": Qual é a headline ou promessa central implícita ou explícita?
3. "pain_or_desire": Qual dor urgente ou desejo aspiracional o vídeo aborda?
4. "sales_argument": Qual foi o argumento central de vendas (USP, prova, transformação ou oferta)?
5. "video_structure": Qual o framework ou estrutura do vídeo? (ex: Problema-Agitação-Solução, Antes/Depois, Teste Extremo, Unboxing com Reação, História Pessoal, etc.).
6. "product_demo_style": Como o produto é demonstrado visualmente? (ex: Demonstração de uso real, efeito imediato, teste de resistência, comparativo lado a lado, estética/ASMR).
7. "cta_type": Qual chamada para ação foi feita? (ex: Carrinho amarelo no TikTok Shop, link na bio, engajamento nos comentários).
8. "hook_strength_score": Nota de 1.0 a 10.0 para a força de retenção do gancho.
9. "key_takeaways": 2 ou 3 lições práticas do porquê esse vídeo vende ou atrai atenção.

Responda em formato JSON com essas chaves exatas:
{
  "hook_initial": "...",
  "headline": "...",
  "pain_or_desire": "...",
  "sales_argument": "...",
  "video_structure": "...",
  "product_demo_style": "...",
  "cta_type": "...",
  "hook_strength_score": 8.5,
  "key_takeaways": "..."
}"""

DEFAULT_PATTERN_SYSTEM = """Você é um estrategista de inteligência de mercado focado em análise quanti-qualitativa de criativos no TikTok Shop.
Você compara grupos de vídeos do mesmo produto para mapear correlações entre técnicas criativas e alto desempenho em visualizações e engajamento."""

DEFAULT_PATTERN_TEMPLATE = """Analise este grupo de vídeos concorrentes para o produto: {{product_name}}

DADOS DOS VÍDEOS ANALISADOS (Ordenados pelo desempenho):
{{videos_data}}

Sua tarefa:
1. Descobrir quais ganchos (hooks) geraram maior retenção e engajamento.
2. Identificar quais estruturas narrativas foram dominantes nos melhores vídeos.
3. Mapear os argumentos de venda e gatilhos que mais converteram a atenção do público.
4. Identificar falhas ou fraquezas nos vídeos com pior desempenho.
5. Sintetizar um relatório executivo de padrões vencedores para guiar novos criativos sem copiar concorrentes.

Responda em JSON com as chaves:
{
  "common_winning_hooks": ["...", "..."],
  "dominant_structures": ["...", "..."],
  "key_selling_arguments": ["...", "..."],
  "engagement_insights": "...",
  "synthesis_report": "..."
}"""

DEFAULT_IDEA_SYSTEM = """Você é um diretor criativo de alta conversão para TikTok Shop.
Sua especialidade é criar conceitos originais, ângulos inovadores e novos roteiros que superam a concorrência sem copiar nada diretamente."""

DEFAULT_IDEA_TEMPLATE = """Com base na análise de concorrência do produto "{{product_name}}", gere {{count}} ideias e novos ângulos de vídeos originais.

PADRÕES QUE COMPROVADAMENTE FUNCIONARAM NO MERCADO:
{{winning_patterns}}

INSTRUÇÕES ESPECÍFICAS DO USUÁRIO:
{{custom_instructions}}

REGRAS:
- NÃO copie os concorrentes. Use a psicologia por trás do que funcionou e crie um novo ângulo (ex: outro ponto de vista, demonstração não óbvia, quebra de mito, teste cego, rotina de vida real).
- Cada ideia deve ser adaptada para o formato dinâmico do TikTok Shop.

Responda em JSON com o formato:
{
  "ideas": [
    {
      "angle_title": "Título conciso do ângulo",
      "hook_suggestion": "Sugestão detalhada do gancho para os primeiros 3 segundos",
      "proposed_structure": "Estrutura recomendada (ex: Hook > Choque > Solução > CTA)",
      "differentiation_angle": "Por que esta abordagem é original e se destaca dos concorrentes",
      "script_outline": "Passo a passo resumido do roteiro"
    }
  ]
}"""


def render_template(template_str: str, context: Dict[str, Any]) -> str:
    """
    Substitui tags no formato {{variable_name}} pelo valor correspondente no contexto.
    Preserva chaves não preenchidas ou substitui por padrão legível.
    """
    rendered = template_str
    for key, val in context.items():
        pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
        rendered = re.sub(pattern, str(val or ""), rendered)
    return rendered


def seed_default_templates(db: Session):
    """
    Garante que os templates padrão existam no banco sem sobrescrever alterações do usuário.
    """
    templates = [
        {
            "name": "Análise Estrutural do Vídeo (Padrão)",
            "template_type": "video_analysis",
            "system_prompt": DEFAULT_VIDEO_ANALYSIS_SYSTEM,
            "template_content": DEFAULT_VIDEO_ANALYSIS_TEMPLATE
        },
        {
            "name": "Identificação de Padrões Vencedores",
            "template_type": "pattern_matching",
            "system_prompt": DEFAULT_PATTERN_SYSTEM,
            "template_content": DEFAULT_PATTERN_TEMPLATE
        },
        {
            "name": "Geração de Ideias Originais Sem Cópia",
            "template_type": "idea_generation",
            "system_prompt": DEFAULT_IDEA_SYSTEM,
            "template_content": DEFAULT_IDEA_TEMPLATE
        }
    ]

    for item in templates:
        existing = db.query(PromptTemplate).filter(PromptTemplate.template_type == item["template_type"]).first()
        if not existing:
            new_tmpl = PromptTemplate(
                name=item["name"],
                template_type=item["template_type"],
                system_prompt=item["system_prompt"],
                template_content=item["template_content"],
                is_active=True
            )
            db.add(new_tmpl)
    db.commit()
