import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models import ProductCluster, VideoAsset, VideoIdea, PromptTemplate, VideoAnalysis
from backend.schemas import GenerateIdeasRequest, VideoIdeaResponse
from backend.services.prompt_engine import render_template
from backend.services.ai_analyzer import generate_differentiated_ideas

router = APIRouter(prefix="/api/ideas", tags=["Idea Generation"])


@router.post("/generate", response_model=List[VideoIdeaResponse])
def generate_ideas(req: GenerateIdeasRequest, db: Session = Depends(get_db)):
    """
    Gera novas ideias de vídeos com ângulos originais baseados nos padrões
    vencedores da concorrência, sem cópia direta.
    """
    product = db.query(ProductCluster).filter(ProductCluster.id == req.product_cluster_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")

    # Coleta análises dos vídeos deste produto para extrair o que funcionou
    videos = db.query(VideoAsset).filter(VideoAsset.product_cluster_id == product.id).all()
    patterns_summary = []
    for v in videos:
        analysis = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == v.id).first()
        if analysis:
            patterns_summary.append({
                "hook": analysis.hook_initial,
                "structure": analysis.video_structure,
                "selling_point": analysis.sales_argument,
                "pain": analysis.pain_or_desire
            })

    winning_patterns_str = json.dumps(patterns_summary, ensure_ascii=False) if patterns_summary else "Foco em demonstrações práticas e ganchos de quebra de padrão."

    # Carrega template de prompts configurado
    tmpl = db.query(PromptTemplate).filter(
        PromptTemplate.template_type == "idea_generation",
        PromptTemplate.is_active == True
    ).first()

    system_prompt = tmpl.system_prompt if tmpl else ""
    user_template = tmpl.template_content if tmpl else ""

    context = {
        "product_name": product.name,
        "count": req.count or 3,
        "winning_patterns": winning_patterns_str,
        "custom_instructions": req.custom_instructions or "Criar roteiros originais e envolventes focados no carrinho do TikTok Shop."
    }
    rendered_user_prompt = render_template(user_template, context)

    ideas_result = generate_differentiated_ideas(
        product_name=product.name,
        winning_patterns_str=winning_patterns_str,
        custom_instructions=context["custom_instructions"],
        count=context["count"],
        system_prompt=system_prompt,
        user_prompt=rendered_user_prompt
    )

    created_ideas = []
    for item in ideas_result.get("ideas", []):
        idea = VideoIdea(
            product_cluster_id=product.id,
            angle_title=item.get("angle_title", "Novo Ângulo Criativo"),
            hook_suggestion=item.get("hook_suggestion", ""),
            proposed_structure=item.get("proposed_structure", ""),
            differentiation_angle=item.get("differentiation_angle", ""),
            script_outline=item.get("script_outline", "")
        )
        db.add(idea)
        created_ideas.append(idea)

    db.commit()
    for idea in created_ideas:
        db.refresh(idea)

    return [VideoIdeaResponse.from_orm(i) for i in created_ideas]


@router.get("/product/{product_id}", response_model=List[VideoIdeaResponse])
def get_product_ideas(product_id: str, db: Session = Depends(get_db)):
    ideas = db.query(VideoIdea).filter(VideoIdea.product_cluster_id == product_id).order_by(VideoIdea.created_at.desc()).all()
    return [VideoIdeaResponse.from_orm(i) for i in ideas]


@router.delete("/{idea_id}")
def delete_idea(idea_id: str, db: Session = Depends(get_db)):
    idea = db.query(VideoIdea).filter(VideoIdea.id == idea_id).first()
    if not idea:
        raise HTTPException(status_code=404, detail="Ideia não encontrada.")
    db.delete(idea)
    db.commit()
    return {"message": "Ideia removida com sucesso."}
