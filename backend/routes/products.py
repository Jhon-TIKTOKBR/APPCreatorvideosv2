from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from backend.database import get_db
from backend.models import ProductCluster, VideoAsset, VideoIdea
from backend.schemas import ProductClusterCreate, ProductClusterUpdate, ProductClusterResponse

router = APIRouter(prefix="/api/products", tags=["Product Clusters"])

@router.get("", response_model=List[ProductClusterResponse])
def list_products(db: Session = Depends(get_db)):
    products = db.query(ProductCluster).order_by(ProductCluster.created_at.desc()).all()
    results = []
    for p in products:
        v_count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.product_cluster_id == p.id).scalar() or 0
        i_count = db.query(func.count(VideoIdea.id)).filter(VideoIdea.product_cluster_id == p.id).scalar() or 0
        p_dict = ProductClusterResponse.from_orm(p)
        p_dict.video_count = v_count
        p_dict.idea_count = i_count
        results.append(p_dict)
    return results


@router.post("", response_model=ProductClusterResponse)
def create_product(prod_in: ProductClusterCreate, db: Session = Depends(get_db)):
    product = ProductCluster(
        name=prod_in.name.strip(),
        category=prod_in.category,
        target_audience=prod_in.target_audience,
        description=prod_in.description
    )
    db.add(product)
    db.commit()
    db.refresh(product)

    res = ProductClusterResponse.from_orm(product)
    res.video_count = 0
    res.idea_count = 0
    return res


@router.get("/{product_id}", response_model=ProductClusterResponse)
def get_product(product_id: str, db: Session = Depends(get_db)):
    prod = db.query(ProductCluster).filter(ProductCluster.id == product_id).first()
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")

    v_count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.product_cluster_id == prod.id).scalar() or 0
    i_count = db.query(func.count(VideoIdea.id)).filter(VideoIdea.product_cluster_id == prod.id).scalar() or 0
    res = ProductClusterResponse.from_orm(prod)
    res.video_count = v_count
    res.idea_count = i_count
    return res


@router.put("/{product_id}", response_model=ProductClusterResponse)
def update_product(product_id: str, prod_in: ProductClusterUpdate, db: Session = Depends(get_db)):
    prod = db.query(ProductCluster).filter(ProductCluster.id == product_id).first()
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")

    if prod_in.name is not None:
        prod.name = prod_in.name
    if prod_in.category is not None:
        prod.category = prod_in.category
    if prod_in.target_audience is not None:
        prod.target_audience = prod_in.target_audience
    if prod_in.description is not None:
        prod.description = prod_in.description

    db.commit()
    db.refresh(prod)

    v_count = db.query(func.count(VideoAsset.id)).filter(VideoAsset.product_cluster_id == prod.id).scalar() or 0
    i_count = db.query(func.count(VideoIdea.id)).filter(VideoIdea.product_cluster_id == prod.id).scalar() or 0
    res = ProductClusterResponse.from_orm(prod)
    res.video_count = v_count
    res.idea_count = i_count
    return res


@router.delete("/{product_id}")
def delete_product(product_id: str, db: Session = Depends(get_db)):
    prod = db.query(ProductCluster).filter(ProductCluster.id == product_id).first()
    if not prod:
        raise HTTPException(status_code=404, detail="Produto não encontrado.")
    db.delete(prod)
    db.commit()
    return {"message": "Produto removido com sucesso."}
