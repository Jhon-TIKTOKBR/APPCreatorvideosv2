from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from backend.database import get_db
from backend.models import VideoAsset, MetricSnapshot, CompetitorProfile, ProductCluster, VideoAnalysis
from backend.schemas import (
    VideoAssetCreate, VideoAssetUpdate, VideoAssetResponse,
    MetricSnapshotCreate, MetricSnapshotResponse
)
from backend.services.tiktok_collector import extract_tiktok_ids, fetch_tiktok_oembed, fetch_public_metadata_ytdlp

router = APIRouter(prefix="/api/videos", tags=["Videos"])


@router.get("/preview-url")
async def preview_tiktok_url(url: str = Query(..., description="URL pública do TikTok")):
    """
    Endpoint para a UI: ao colar o link, busca metadados oficiais oEmbed
    e faz tentativa leve com yt-dlp para pré-preencher o formulário.
    """
    id_info = extract_tiktok_ids(url)
    oembed_res = await fetch_tiktok_oembed(url)
    
    extracted = {
        "url": url,
        "username": id_info.get("username"),
        "video_id": id_info.get("video_id"),
        "title": oembed_res.get("title") if oembed_res.get("success") else None,
        "author_name": oembed_res.get("author_name") if oembed_res.get("success") else None,
        "author_url": oembed_res.get("author_url") if oembed_res.get("success") else None,
        "thumbnail_url": oembed_res.get("thumbnail_url") if oembed_res.get("success") else None,
        "views": None,
        "likes": None,
        "comments": None,
        "shares": None,
        "published_at": None,
        "extraction_source": "oembed" if oembed_res.get("success") else "manual"
    }

    # Tentativa com yt-dlp apenas se necessário para obter números públicos
    ytdlp_res = fetch_public_metadata_ytdlp(url)
    if ytdlp_res.get("success"):
        if ytdlp_res.get("views") is not None:
            extracted["views"] = ytdlp_res["views"]
            extracted["extraction_source"] = "yt_dlp"
        if ytdlp_res.get("likes") is not None:
            extracted["likes"] = ytdlp_res["likes"]
        if ytdlp_res.get("comments") is not None:
            extracted["comments"] = ytdlp_res["comments"]
        if ytdlp_res.get("shares") is not None:
            extracted["shares"] = ytdlp_res["shares"]
        if ytdlp_res.get("published_at"):
            extracted["published_at"] = ytdlp_res["published_at"].isoformat()
        if not extracted["title"] and ytdlp_res.get("title"):
            extracted["title"] = ytdlp_res["title"]
        if not extracted["thumbnail_url"] and ytdlp_res.get("thumbnail_url"):
            extracted["thumbnail_url"] = ytdlp_res["thumbnail_url"]

    return extracted


@router.get("", response_model=List[VideoAssetResponse])
def list_videos(
    competitor_id: Optional[str] = None,
    product_cluster_id: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(VideoAsset)
    if competitor_id:
        query = query.filter(VideoAsset.competitor_id == competitor_id)
    if product_cluster_id:
        query = query.filter(VideoAsset.product_cluster_id == product_cluster_id)
    if status:
        query = query.filter(VideoAsset.status == status)

    videos = query.order_by(VideoAsset.created_at.desc()).all()
    results = []
    for v in videos:
        res = VideoAssetResponse.from_orm(v)
        # Snapshot mais recente
        latest_snap = db.query(MetricSnapshot).filter(MetricSnapshot.video_id == v.id).order_by(MetricSnapshot.recorded_at.desc()).first()
        if latest_snap:
            res.latest_snapshot = MetricSnapshotResponse.from_orm(latest_snap)
        if v.competitor:
            res.competitor_username = v.competitor.username
        if v.product_cluster:
            res.product_name = v.product_cluster.name
        results.append(res)
    return results


@router.post("", response_model=VideoAssetResponse)
async def create_video(video_in: VideoAssetCreate, db: Session = Depends(get_db)):
    clean_url = video_in.tiktok_url.strip()
    id_info = extract_tiktok_ids(clean_url)
    
    # Se competitor_id não foi informado, mas detectamos o @username na URL, podemos vincular ou criar perfil automaticamente
    competitor_id = video_in.competitor_id
    if not competitor_id and id_info.get("username"):
        comp = db.query(CompetitorProfile).filter(CompetitorProfile.username == id_info["username"]).first()
        if not comp:
            comp = CompetitorProfile(
                username=id_info["username"],
                nickname=id_info["username"],
                profile_url=f"https://www.tiktok.com/{id_info['username']}"
            )
            db.add(comp)
            db.commit()
            db.refresh(comp)
        competitor_id = comp.id

    # Busca metadados oEmbed se faltar título ou thumbnail
    title = video_in.title_caption
    author_name = None
    author_url = None
    thumbnail_url = None

    oembed_data = await fetch_tiktok_oembed(clean_url)
    if oembed_data.get("success"):
        title = title or oembed_data.get("title")
        author_name = oembed_data.get("author_name")
        author_url = oembed_data.get("author_url")
        thumbnail_url = oembed_data.get("thumbnail_url")

    video = VideoAsset(
        tiktok_url=clean_url,
        tiktok_video_id=id_info.get("video_id"),
        competitor_id=competitor_id,
        product_cluster_id=video_in.product_cluster_id,
        title_caption=title,
        author_name=author_name,
        author_url=author_url,
        thumbnail_url=thumbnail_url,
        published_at=video_in.published_at,
        raw_transcript=video_in.raw_transcript,
        status="pending_analysis"
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # Criação do primeiro snapshot de métricas se algum valor foi informado
    has_metrics = any(val is not None for val in [video_in.views, video_in.likes, video_in.comments, video_in.shares])
    if has_metrics:
        snapshot = MetricSnapshot(
            video_id=video.id,
            views=video_in.views,
            likes=video_in.likes,
            comments=video_in.comments,
            shares=video_in.shares,
            extraction_source=video_in.extraction_source or "manual"
        )
        db.add(snapshot)
        db.commit()

    res = VideoAssetResponse.from_orm(video)
    latest_snap = db.query(MetricSnapshot).filter(MetricSnapshot.video_id == video.id).order_by(MetricSnapshot.recorded_at.desc()).first()
    if latest_snap:
        res.latest_snapshot = MetricSnapshotResponse.from_orm(latest_snap)
    if video.competitor:
        res.competitor_username = video.competitor.username
    if video.product_cluster:
        res.product_name = video.product_cluster.name
    return res


@router.get("/{video_id}", response_model=VideoAssetResponse)
def get_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    res = VideoAssetResponse.from_orm(video)
    latest_snap = db.query(MetricSnapshot).filter(MetricSnapshot.video_id == video.id).order_by(MetricSnapshot.recorded_at.desc()).first()
    if latest_snap:
        res.latest_snapshot = MetricSnapshotResponse.from_orm(latest_snap)
    if video.competitor:
        res.competitor_username = video.competitor.username
    if video.product_cluster:
        res.product_name = video.product_cluster.name
    return res


@router.get("/{video_id}/history", response_model=List[MetricSnapshotResponse])
def get_video_metric_history(video_id: str, db: Session = Depends(get_db)):
    """
    Retorna o histórico de métricas ao longo do tempo para o vídeo.
    Permite visualizar a curva de crescimento de visualizações e engajamento.
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    snapshots = db.query(MetricSnapshot).filter(MetricSnapshot.video_id == video_id).order_by(MetricSnapshot.recorded_at.asc()).all()
    return [MetricSnapshotResponse.from_orm(s) for s in snapshots]


@router.post("/{video_id}/snapshot", response_model=MetricSnapshotResponse)
def add_metric_snapshot(video_id: str, snap_in: MetricSnapshotCreate, db: Session = Depends(get_db)):
    """
    Adiciona um novo registro pontual de métricas ao longo do tempo (Histórico de Concorrentes).
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    snap = MetricSnapshot(
        video_id=video_id,
        views=snap_in.views,
        likes=snap_in.likes,
        comments=snap_in.comments,
        shares=snap_in.shares,
        extraction_source=snap_in.extraction_source or "manual"
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)
    return MetricSnapshotResponse.from_orm(snap)


@router.put("/{video_id}", response_model=VideoAssetResponse)
def update_video(video_id: str, video_in: VideoAssetUpdate, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")

    if video_in.product_cluster_id is not None:
        video.product_cluster_id = video_in.product_cluster_id
    if video_in.competitor_id is not None:
        video.competitor_id = video_in.competitor_id
    if video_in.title_caption is not None:
        video.title_caption = video_in.title_caption
    if video_in.raw_transcript is not None:
        video.raw_transcript = video_in.raw_transcript
    if video_in.published_at is not None:
        video.published_at = video_in.published_at

    db.commit()
    db.refresh(video)
    return get_video(video_id, db)


@router.delete("/{video_id}")
def delete_video(video_id: str, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado.")
    db.delete(video)
    db.commit()
    return {"message": "Vídeo removido com sucesso."}
