import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("=" * 65)
    print("🚀 TikTok Shop AI Studio - Módulo de Inteligência de Concorrentes")
    print(f"📡 Painel Web: http://localhost:{port}")
    print(f"📖 Documentação Interativa da API (Swagger): http://localhost:{port}/docs")
    print("=" * 65)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, reload=True)
