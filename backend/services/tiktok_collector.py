import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import httpx
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logger = logging.getLogger(__name__)

def extract_tiktok_ids(url: str) -> Dict[str, Optional[str]]:
    """
    Extrai o username e o video_id de links do TikTok.
    Suporta formatos:
    - https://www.tiktok.com/@username/video/7123456789012345678
    - https://vm.tiktok.com/ZMeXXXXX/
    - https://www.tiktok.com/@username
    """
    clean_url = url.split("?")[0].strip()
    video_id = None
    username = None

    # Padrão com @username/video/12345...
    video_match = re.search(r"@([a-zA-Z0-9_.-]+)/video/(\d+)", clean_url)
    if video_match:
        username = f"@{video_match.group(1)}"
        video_id = video_match.group(2)
        return {"username": username, "video_id": video_id}

    # Padrão de perfil @username
    profile_match = re.search(r"@([a-zA-Z0-9_.-]+)", clean_url)
    if profile_match:
        username = f"@{profile_match.group(1)}"

    return {"username": username, "video_id": video_id}


async def fetch_tiktok_oembed(url: str) -> Dict[str, Any]:
    """
    Consulta o endpoint oficial e gratuito de oEmbed do TikTok.
    Retorna título/legenda, autor, thumbnail e html do player.
    100% compliant com os termos públicos.
    """
    oembed_url = f"https://www.tiktok.com/oembed?url={url}"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(oembed_url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "title": data.get("title"),
                    "author_name": data.get("author_name"),
                    "author_url": data.get("author_url"),
                    "thumbnail_url": data.get("thumbnail_url"),
                    "html": data.get("html")
                }
            else:
                logger.warning(f"TikTok oEmbed retornou status {resp.status_code} para {url}")
                return {"success": False, "error": f"Status HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"Erro ao consultar oEmbed do TikTok: {e}")
        return {"success": False, "error": str(e)}


def fetch_public_metadata_ytdlp(url: str) -> Dict[str, Any]:
    """
    Tentativa segura de extração de métricas públicas via yt-dlp.
    Se falhar, retornar valores nulos sem travar o sistema.
    NUNCA inventa métricas.
    """
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'socket_timeout': 8,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False}

            # Extração de campos numéricos (somente se existirem)
            views = info.get("view_count")
            likes = info.get("like_count")
            comments = info.get("comment_count")
            shares = info.get("repost_count")

            # Timestamp de upload
            published_at = None
            timestamp = info.get("timestamp")
            if timestamp:
                try:
                    published_at = datetime.utcfromtimestamp(timestamp)
                except Exception:
                    pass

            return {
                "success": True,
                "title": info.get("title") or info.get("description"),
                "views": views if isinstance(views, int) else None,
                "likes": likes if isinstance(likes, int) else None,
                "comments": comments if isinstance(comments, int) else None,
                "shares": shares if isinstance(shares, int) else None,
                "published_at": published_at,
                "thumbnail_url": info.get("thumbnail"),
                "uploader": info.get("uploader"),
                "uploader_url": info.get("uploader_url")
            }
    except Exception as e:
        logger.info(f"yt-dlp não extraiu metadados (esperado caso haja captcha ou proteção): {e}")
        return {"success": False, "error": str(e)}
