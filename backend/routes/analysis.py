import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from backend.database import get_db
from backend.models import VideoAsset, VideoAnalysis, PromptTemplate
from backend.schemas import AnalyzeVideoRequest, VideoAnalysisResponse
from backend.services.prompt_engine import render_template
from backend.services.ai_analyzer import analyze_video_content

router = APIRouter(prefix="/api/analysis", tags=["Video Analysis"])


@router.post("/video", response_model=VideoAnalysisResponse)
def analyze_video(req: AnalyzeVideoRequest, db: Session = Depends(get_db)):
    """
    Executa a análise de IA identificando:
    1. Gancho inicial
    2. Headline
    3. Dor ou desejo explorado
    4. Argumento de venda
    5. Estrutura do vídeo
    6. Demonstração do produto
    7. CTA utilizado
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == req.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    transcript = req.transcript_override or video.raw_transcript or ""
    caption = video.title_caption or ""
    product_name = video.product_cluster.name if video.product_cluster else "Produto Genérico"

    # Seleciona o template de prompt ativo ou customizado
    if req.custom_prompt_id:
        tmpl = db.query(PromptTemplate).filter(PromptTemplate.id == req.custom_prompt_id).first()
    else:
        tmpl = db.query(PromptTemplate).filter(
            PromptTemplate.template_type == "video_analysis",
            PromptTemplate.is_active == True
        ).first()

    system_prompt = tmpl.system_prompt if tmpl else ""
    user_template = tmpl.template_content if tmpl else ""

    # Interpolação das variáveis
    context = {
        "caption": caption,
        "transcript": transcript,
        "product_name": product_name
    }
    rendered_user_prompt = render_template(user_template, context)

    # Execução da análise via IA (Gemini Flash com baixo custo)
    analysis_data = analyze_video_content(
        caption=caption,
        transcript=transcript,
        product_name=product_name,
        system_prompt=system_prompt,
        user_prompt=rendered_user_prompt
    )

    # Atualiza ou cria o registro de VideoAnalysis
    existing = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == video.id).first()
    if existing:
        existing.hook_initial = analysis_data.get("hook_initial")
        existing.headline = analysis_data.get("headline")
        existing.pain_or_desire = analysis_data.get("pain_or_desire")
        existing.sales_argument = analysis_data.get("sales_argument")
        existing.video_structure = analysis_data.get("video_structure")
        existing.product_demo_style = analysis_data.get("product_demo_style")
        existing.cta_type = analysis_data.get("cta_type")
        existing.hook_strength_score = analysis_data.get("hook_strength_score")
        existing.key_takeaways = analysis_data.get("key_takeaways")
        existing.raw_analysis_json = json.dumps(analysis_data, ensure_ascii=False)
        existing.analyzed_at = datetime.utcnow()
        analysis_record = existing
    else:
        analysis_record = VideoAnalysis(
            video_id=video.id,
            hook_initial=analysis_data.get("hook_initial"),
            headline=analysis_data.get("headline"),
            pain_or_desire=analysis_data.get("pain_or_desire"),
            sales_argument=analysis_data.get("sales_argument"),
            video_structure=analysis_data.get("video_structure"),
            product_demo_style=analysis_data.get("product_demo_style"),
            cta_type=analysis_data.get("cta_type"),
            hook_strength_score=analysis_data.get("hook_strength_score"),
            key_takeaways=analysis_data.get("key_takeaways"),
            raw_analysis_json=json.dumps(analysis_data, ensure_ascii=False)
        )
        db.add(analysis_record)

    video.status = "analyzed"
    if req.transcript_override and not video.raw_transcript:
        video.raw_transcript = req.transcript_override

    db.commit()
    db.refresh(analysis_record)
    return VideoAnalysisResponse.from_orm(analysis_record)


@router.get("/video/{video_id}", response_model=VideoAnalysisResponse)
def get_video_analysis(video_id: str, db: Session = Depends(get_db)):
    analysis = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == video_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Análise não encontrada para este vídeo.")
    return VideoAnalysisResponse.from_orm(analysis)


@router.post("/generate-scenes/{video_id}")
def generate_scenes_for_video(video_id: str, db: Session = Depends(get_db)):
    """
    Gera automaticamente os prompts de imagem (em inglês) para Nano Banana / Imagen 3
    e o roteiro de áudio com micro-cortes e CTA, no formato Provador POV 9:16.
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    analysis = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == video.id).first()
    product_name = video.product_cluster.name if video.product_cluster else (video.title_caption or "Look de Moda Feminina")
    caption = video.title_caption or ""

    hook = analysis.hook_initial if analysis else "Gancho de quebra de padrão visual"
    pain = analysis.pain_or_desire if analysis else "Evitar roupas que marcam ou ficam transparentes"
    sales = analysis.sales_argument if analysis else "Caimento perfeito e sofisticação acessível"

    from backend.services.ai_analyzer import generate_scenes_from_analysis
    scenes_data = generate_scenes_from_analysis(
        product_name=product_name,
        caption=caption,
        hook=hook,
        pain_or_desire=pain,
        sales_argument=sales
    )
    return scenes_data


@router.post("/generate-tripack/{video_id}")
def generate_tripack_for_video(video_id: str, db: Session = Depends(get_db)):
    """
    Gera o pacote com 3 vídeos (3 Ambientes diferentes e 3 Ângulos persuasivos)
    mantendo a consistência da mesma modelo para o produto.
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    analysis = db.query(VideoAnalysis).filter(VideoAnalysis.video_id == video.id).first()
    product_name = video.product_cluster.name if video.product_cluster else (video.title_caption or "Look de Moda Feminina")
    caption = video.title_caption or ""

    hook = analysis.hook_initial if analysis else "Gancho de quebra de padrão visual"
    pain = analysis.pain_or_desire if analysis else "Evitar roupas que marcam ou ficam transparentes"
    sales = analysis.sales_argument if analysis else "Caimento perfeito e sofisticação acessível"

    from backend.services.ai_analyzer import generate_tripack_from_video
    tripack_data = generate_tripack_from_video(
        product_name=product_name,
        caption=caption,
        hook=hook,
        pain_or_desire=pain,
        sales_argument=sales
    )
    return tripack_data
