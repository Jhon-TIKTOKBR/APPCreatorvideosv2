import os
import sys

# Inclui a raiz do projeto no path de importação para carregar o módulo 'backend'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
