import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# No ambiente Serverless da Vercel, apenas a pasta /tmp tem permissão de escrita
if os.environ.get("VERCEL"):
    DATABASE_DIR = "/tmp"
    DB_PATH = os.path.join(DATABASE_DIR, "tiktok_studio.db")
else:
    DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(DATABASE_DIR, exist_ok=True)
    DB_PATH = os.path.join(DATABASE_DIR, "tiktok_studio.db")

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
