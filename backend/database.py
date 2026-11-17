import os
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://wezegapfsrejmgdxutrt.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndlemVnYXBmc3Jlam1nZHh1dHJ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTQ0MTM4NTcsImV4cCI6MjEwOTk4OTg1N30.KVgKmLPzKYrLhOe_c5B6OoSpMLrFwq3FLEIrHHCyyO0"
)

def save_generation_to_supabase(payload: dict) -> bool:
    """Salva o histórico de prompts gerados diretamente na tabela wizard_generations do Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/wizard_generations",
            json=payload,
            headers=headers,
            timeout=5.0
        )
        return resp.status_code in (200, 201, 204)
    except Exception:
        return False

# No ambiente Serverless da Vercel ou local, definimos o caminho do SQLite de fallback
if os.environ.get("VERCEL"):
    DATABASE_DIR = "/tmp"
    DB_PATH = os.path.join(DATABASE_DIR, "tiktok_studio.db")
else:
    DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(DATABASE_DIR, exist_ok=True)
    DB_PATH = os.path.join(DATABASE_DIR, "tiktok_studio.db")

# Prioriza o banco do Supabase (PostgreSQL) se configurado nas variáveis de ambiente
raw_db_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
if raw_db_url:
    # Ajusta prefixo postgres:// para postgresql:// exigido pelo SQLAlchemy 2.0
    if raw_db_url.startswith("postgres://"):
        raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
    DATABASE_URL = raw_db_url
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
