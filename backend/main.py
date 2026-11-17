import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.database import engine, Base, SessionLocal
from backend.services.prompt_engine import seed_default_templates
from backend.routes import competitors, products, videos, analysis, patterns, ideas, prompts, wizard

# Criação das tabelas no banco de dados SQLite
Base.metadata.create_all(bind=engine)

# Inicialização dos templates padrão
with SessionLocal() as db:
    seed_default_templates(db)

app = FastAPI(
    title="TikTok Shop AI Studio - Inteligência de Concorrentes",
    description="Sistema de engenharia reversa de criativos virais, benchmark de concorrentes e ideação estratégica.",
    version="1.0.0"
)

# CORS liberado para integração fluida
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusão dos roteadores da API
app.include_router(competitors.router)
app.include_router(products.router)
app.include_router(videos.router)
app.include_router(analysis.router)
app.include_router(patterns.router)
app.include_router(ideas.router)
app.include_router(prompts.router)
app.include_router(wizard.router)

# Diretório da interface estática (SPA)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

REFS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_REFERENCIAS")
if os.path.exists(REFS_DIR):
    app.mount("/references", StaticFiles(directory=REFS_DIR), name="references")


@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "status": "online",
        "module": "TikTok Shop Competitor Intelligence",
        "api_docs": "/docs"
    }
