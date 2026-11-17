from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import csv
import io
import re
from datetime import datetime

from backend.database import get_db
from backend.models import CompetitorProfile, VideoAsset, MetricSnapshot
from backend.schemas import CompetitorCreate, CompetitorUpdate, CompetitorResponse
from backend.services.tiktok_collector import extract_tiktok_ids

router = APIRouter(prefix="/api/competitors", tags=["Competitors"])

@router.get("", response_model=List[CompetitorResponse])
def list_competitors(db: Session = Depends(get_db)):
    competitors = db.query(CompetitorProfile).order_by(CompetitorProfile.created_at.desc()).all()
    results = []
    for c in competitors:
        count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.competitor_id == c.id).scalar() or 0
        c_dict = CompetitorResponse.from_orm(c)
        c_dict.video_count = count
        results.append(c_dict)
    return results


@router.post("", response_model=CompetitorResponse)
def create_competitor(comp_in: CompetitorCreate, db: Session = Depends(get_db)):
    clean_username = comp_in.username.strip()
    if not clean_username.startswith("@"):
        clean_username = f"@{clean_username}"

    existing = db.query(CompetitorProfile).filter(CompetitorProfile.username == clean_username).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Concorrente com o usuário {clean_username} já existe.")

    profile_url = comp_in.profile_url
    if not profile_url:
        profile_url = f"https://www.tiktok.com/{clean_username}"

    competitor = CompetitorProfile(
        username=clean_username,
        nickname=comp_in.nickname or clean_username,
        profile_url=profile_url,
        niche=comp_in.niche,
        notes=comp_in.notes,
        avatar_url=comp_in.avatar_url,
        is_own_profile=bool(comp_in.is_own_profile)
    )
    db.add(competitor)
    db.commit()
    db.refresh(competitor)

    res = CompetitorResponse.from_orm(competitor)
    res.video_count = 0
    return res


@router.get("/{competitor_id}", response_model=CompetitorResponse)
def get_competitor(competitor_id: str, db: Session = Depends(get_db)):
    comp = db.query(CompetitorProfile).filter(CompetitorProfile.id == competitor_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Concorrente não encontrado.")
    
    count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.competitor_id == comp.id).scalar() or 0
    res = CompetitorResponse.from_orm(comp)
    res.video_count = count
    return res


@router.put("/{competitor_id}", response_model=CompetitorResponse)
def update_competitor(competitor_id: str, comp_in: CompetitorUpdate, db: Session = Depends(get_db)):
    comp = db.query(CompetitorProfile).filter(CompetitorProfile.id == competitor_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Concorrente não encontrado.")

    if comp_in.nickname is not None:
        comp.nickname = comp_in.nickname
    if comp_in.niche is not None:
        comp.niche = comp_in.niche
    if comp_in.notes is not None:
        comp.notes = comp_in.notes
    if comp_in.avatar_url is not None:
        comp.avatar_url = comp_in.avatar_url
    if comp_in.is_own_profile is not None:
        comp.is_own_profile = comp_in.is_own_profile

    db.commit()
    db.refresh(comp)

    count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.competitor_id == comp.id).scalar() or 0
    res = CompetitorResponse.from_orm(comp)
    res.video_count = count
    return res


@router.delete("/{competitor_id}")
def delete_competitor(competitor_id: str, db: Session = Depends(get_db)):
    comp = db.query(CompetitorProfile).filter(CompetitorProfile.id == competitor_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Concorrente não encontrado.")
    db.delete(comp)
    db.commit()
    return {"message": "Concorrente removido com sucesso."}


@router.post("/import-analytics-csv")
async def import_analytics_csv(
    file: UploadFile = File(...),
    profile_id: str = Form(...),
    product_cluster_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Importa relatório analítico exportado do TikTok Studio (.CSV).
    Lê automaticamente links de vídeos, títulos e contagem de views, likes, comments e shares.
    """
    profile = db.query(CompetitorProfile).filter(CompetitorProfile.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Perfil não encontrado.")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="Arquivo CSV vazio.")

    # Busca cabeçalho e mapeia índices de colunas
    header = [h.strip().lower() for h in rows[0]]
    imported_count = 0
    updated_count = 0

    def find_idx(*keywords):
        for idx, col in enumerate(header):
            for kw in keywords:
                if kw in col:
                    return idx
        return -1

    idx_url = find_idx("video link", "link", "url", "vídeo")
    idx_title = find_idx("video title", "title", "caption", "título", "legenda", "post")
    idx_views = find_idx("video views", "views", "visualizações", "visualizacoes")
    idx_likes = find_idx("likes", "curtidas")
    idx_comments = find_idx("comments", "comentários", "comentarios")
    idx_shares = find_idx("shares", "compartilhamentos")

    def clean_num(val):
        if not val:
            return None
        cleaned = re.sub(r"[^\d]", "", str(val))
        return int(cleaned) if cleaned else None

    for r in rows[1:]:
        if not r or len(r) <= max(0, idx_url):
            continue

        raw_url = r[idx_url].strip() if idx_url != -1 and idx_url < len(r) else ""
        if not raw_url or "tiktok.com" not in raw_url:
            continue

        title = r[idx_title].strip() if idx_title != -1 and idx_title < len(r) else None
        views = clean_num(r[idx_views]) if idx_views != -1 and idx_views < len(r) else None
        likes = clean_num(r[idx_likes]) if idx_likes != -1 and idx_likes < len(r) else None
        comments = clean_num(r[idx_comments]) if idx_comments != -1 and idx_comments < len(r) else None
        shares = clean_num(r[idx_shares]) if idx_shares != -1 and idx_shares < len(r) else None

        id_info = extract_tiktok_ids(raw_url)

        # Verifica se vídeo já existe
        existing_video = db.query(VideoAsset).filter(VideoAsset.tiktok_url == raw_url).first()
        if existing_video:
            # Atualiza dados se necessário
            if title and not existing_video.title_caption:
                existing_video.title_caption = title
            if product_cluster_id and not existing_video.product_cluster_id:
                existing_video.product_cluster_id = product_cluster_id
            
            # Adiciona novo snapshot de métricas
            if any(val is not None for val in [views, likes, comments, shares]):
                snap = MetricSnapshot(
                    video_id=existing_video.id,
                    views=views,
                    likes=likes,
                    comments=comments,
                    shares=shares,
                    extraction_source="csv_import"
                )
                db.add(snap)
            updated_count += 1
        else:
            new_video = VideoAsset(
                tiktok_url=raw_url,
                tiktok_video_id=id_info.get("video_id"),
                competitor_id=profile.id,
                product_cluster_id=product_cluster_id,
                title_caption=title,
                status="pending_analysis"
            )
            db.add(new_video)
            db.commit()
            db.refresh(new_video)

            if any(val is not None for val in [views, likes, comments, shares]):
                snap = MetricSnapshot(
                    video_id=new_video.id,
                    views=views,
                    likes=likes,
                    comments=comments,
                    shares=shares,
                    extraction_source="csv_import"
                )
                db.add(snap)
            imported_count += 1

    db.commit()
    return {
        "success": True,
        "imported_new": imported_count,
        "updated_existing": updated_count,
        "message": f"Processamento concluído: {imported_count} novos vídeos importados, {updated_count} vídeos atualizados."
    }

