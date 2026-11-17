import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models import ProductCluster, VideoAsset, MetricSnapshot, VideoAnalysis, PromptTemplate
from backend.schemas import PatternAnalysisResponse, VideoPerformanceItem
from backend.services.prompt_engine import render_template
from backend.services.ai_analyzer import analyze_pattern_cluster

router = APIRouter(prefix="/api/patterns", tags=["Patterns & Comparison"])


@router.get("/product/{product_id}", response_model=PatternAnalysisResponse)
def analyze_product_patterns(product_id: str, db: Session = Depends(get_db)):
    """
    Compara todos os vídeos vinculados a um mesmo produto e mapeia
    padrões vencedores nos conteúdos de melhor desempenho.
    """
    product = db.query(ProductCluster).filter(ProductCluster.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")

    videos = db.query(VideoAsset).filter(VideoAsset.product_cluster_id == product_id).all()
    if not videos:
        raise HTTPException(status_code=400, detail="Nenhum vídeo cadastrado para este produto ainda.")

    performance_items: List[VideoPerformanceItem] = []
    videos_summary_list = []

    for v in videos:
        latest_snap = db.query(MetricSnapshot).filter(MetricSnapshot.video_id == v.id).order_by(MetricSnapshot.recorded_at.desc()).first()
        views = latest_snap.views if latest_snap else None
        likes = latest_snap.likes if latest_snap else None
        comments = latest_snap.comments if latest_snap else None
        shares = latest_snap.shares if latest_snap else None

        # Cálculo da taxa de engajamento (quando views existem)
        engagement_rate = None
        if views and views > 0:
            total_eng = (likes or 0) + (comments or 0) + (shares or 0)
            engagement_rate = round((total_eng / views) * 100, 2)

        analysis = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == v.id).first()

        is_own = bool(v.competitor and v.competitor.is_own_profile)
        creator_name = v.competitor.username if v.competitor else None

        item = VideoPerformanceItem(
            video_id=v.id,
            tiktok_url=v.tiktok_url,
            title_caption=v.title_caption,
            thumbnail_url=v.thumbnail_url,
            creator_username=creator_name,
            is_own_video=is_own,
            views=views,
            likes=likes,
            comments=comments,
            shares=shares,
            engagement_rate=engagement_rate,
            hook_initial=analysis.hook_initial if analysis else None,
            headline=analysis.headline if analysis else None,
            video_structure=analysis.video_structure if analysis else None,
            sales_argument=analysis.sales_argument if analysis else None,
            cta_type=analysis.cta_type if analysis else None,
            hook_strength_score=analysis.hook_strength_score if analysis else None
        )
        performance_items.append(item)

        videos_summary_list.append({
            "url": v.tiktok_url,
            "views": views,
            "likes": likes,
            "shares": shares,
            "engagement_rate": f"{engagement_rate}%" if engagement_rate is not None else "N/A",
            "hook": analysis.hook_initial if analysis else "Ainda não analisado",
            "structure": analysis.video_structure if analysis else "N/A",
            "sales_argument": analysis.sales_argument if analysis else "N/A"
        })

    # Ordena vídeos por engajamento ou views (do maior para o menor)
    performance_items.sort(
        key=lambda x: (x.engagement_rate or 0, x.views or 0),
        reverse=True
    )

    # Execução do prompt de padrões
    tmpl = db.query(PromptTemplate).filter(
        PromptTemplate.template_type == "pattern_matching",
        PromptTemplate.is_active == True
    ).first()

    system_prompt = tmpl.system_prompt if tmpl else ""
    user_template = tmpl.template_content if tmpl else ""

    context = {
        "product_name": product.name,
        "videos_data": json.dumps(videos_summary_list, indent=2, ensure_ascii=False)
    }
    rendered_user_prompt = render_template(user_template, context)

    pattern_result = analyze_pattern_cluster(
        product_name=product.name,
        videos_data_json=context["videos_data"],
        system_prompt=system_prompt,
        user_prompt=rendered_user_prompt
    )

    return PatternAnalysisResponse(
        product_cluster_id=product.id,
        product_name=product.name,
        total_videos_analyzed=len(videos),
        top_performing_videos=performance_items,
        common_winning_hooks=pattern_result.get("common_winning_hooks", []),
        dominant_structures=pattern_result.get("dominant_structures", []),
        key_selling_arguments=pattern_result.get("key_selling_arguments", []),
        engagement_insights=pattern_result.get("engagement_insights", ""),
        synthesis_report=pattern_result.get("synthesis_report", "")
    )
