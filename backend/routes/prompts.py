from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database import get_db
from backend.models import PromptTemplate
from backend.schemas import PromptTemplateCreate, PromptTemplateUpdate, PromptTemplateResponse
from backend.services.prompt_engine import seed_default_templates

router = APIRouter(prefix="/api/prompts", tags=["Prompt Templates"])


@router.get("", response_model=List[PromptTemplateResponse])
def list_prompt_templates(db: Session = Depends(get_db)):
    """
    Lista todos os templates de prompts (análise, padrões e ideias).
    Permite que o usuário insira e customize seus próprios modelos de prompts.
    """
    templates = db.query(PromptTemplate).order_by(PromptTemplate.created_at.asc()).all()
    if not templates:
        seed_default_templates(db)
        templates = db.query(PromptTemplate).order_by(PromptTemplate.created_at.asc()).all()
    return [PromptTemplateResponse.from_orm(t) for t in templates]


@router.post("", response_model=PromptTemplateResponse)
def create_prompt_template(tmpl_in: PromptTemplateCreate, db: Session = Depends(get_db)):
    new_tmpl = PromptTemplate(
        name=tmpl_in.name,
        template_type=tmpl_in.template_type,
        system_prompt=tmpl_in.system_prompt,
        template_content=tmpl_in.template_content,
        is_active=tmpl_in.is_active if tmpl_in.is_active is not None else True
    )
    db.add(new_tmpl)
    db.commit()
    db.refresh(new_tmpl)
    return PromptTemplateResponse.from_orm(new_tmpl)


@router.get("/{template_id}", response_model=PromptTemplateResponse)
def get_prompt_template(template_id: str, db: Session = Depends(get_db)):
    tmpl = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template não encontrado.")
    return PromptTemplateResponse.from_orm(tmpl)


@router.put("/{template_id}", response_model=PromptTemplateResponse)
def update_prompt_template(template_id: str, tmpl_in: PromptTemplateUpdate, db: Session = Depends(get_db)):
    """
    Permite ao usuário atualizar qualquer template de prompt com suas próprias
    estruturas e instruções proprietárias.
    """
    tmpl = db.query(PromptTemplate).filter(PromptTemplate.id == template_id).first()
    if not tmpl:
        raise HTTPException(status_code=404, detail="Template não encontrado.")

    if tmpl_in.name is not None:
        tmpl.name = tmpl_in.name
    if tmpl_in.system_prompt is not None:
        tmpl.system_prompt = tmpl_in.system_prompt
    if tmpl_in.template_content is not None:
        tmpl.template_content = tmpl_in.template_content
    if tmpl_in.is_active is not None:
        tmpl.is_active = tmpl_in.is_active

    db.commit()
    db.refresh(tmpl)
    return PromptTemplateResponse.from_orm(tmpl)
