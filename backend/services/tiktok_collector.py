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

BLACKLIST_PHRASES = [
    "tiktok - make your day", "make your day", "tiktok", "visit tiktok",
    "watch, follow, and discover", "trending content", "log in to tiktok",
    "descubra vídeos", "look feminino elegante", "tiktok shop", "para você", "for you",
    "assista a mais vídeos", "video", "vídeo", "foryou", "fyp", "viral",
    "slardar", "wafchallengeid", "please wait"
]

def is_blacklisted_or_invalid(name: Optional[str]) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    low = name.strip().lower()
    for b in BLACKLIST_PHRASES:
        if b in low or low == b:
            return True
    return False


async def unshorten_tiktok_url(url: str) -> str:
    """Resolve links curtos do TikTok (vt.tiktok.com, vm.tiktok.com, /t/)."""
    clean_url = url.strip()
    if any(k in clean_url for k in ["vt.tiktok.com", "vm.tiktok.com", "/t/"]):
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True) as client:
                resp = await client.head(clean_url)
                if str(resp.url) != clean_url:
                    return str(resp.url)
        except Exception:
            pass
    return clean_url


async def extract_tiktok_metadata_fast(url: str) -> Dict[str, Any]:
    """
    Tenta obter o texto (legenda ou título) do vídeo do TikTok via oEmbed, yt-dlp ou meta tags
    SEM baixar o vídeo e com timeout curto para evitar travar na Vercel.
    Retorna success=False caso o TikTok bloqueie via WAF/anti-bot.
    """
    clean_url = await unshorten_tiktok_url(url)

    # 1. Tentar oEmbed oficial do TikTok
    oembed = await fetch_tiktok_oembed(clean_url)
    if oembed.get("success") and oembed.get("title"):
        cand = oembed["title"].strip()
        if not is_blacklisted_or_invalid(cand):
            return {
                "success": True,
                "text": cand,
                "author_name": oembed.get("author_name"),
                "source": "oembed",
                "clean_url": clean_url
            }

    # 2. Tentar yt-dlp apenas para metadados (skip_download=True)
    if yt_dlp:
        try:
            ydl_res = fetch_public_metadata_ytdlp(clean_url)
            if ydl_res.get("success") and ydl_res.get("title"):
                cand = ydl_res["title"].strip()
                if not is_blacklisted_or_invalid(cand):
                    return {
                        "success": True,
                        "text": cand,
                        "author_name": ydl_res.get("uploader"),
                        "source": "ytdlp",
                        "clean_url": clean_url
                    }
        except Exception:
            pass

    # 3. Tentar leitura de meta tags com headers de navegador
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8"
        }
        async with httpx.AsyncClient(timeout=3.5, follow_redirects=True) as client:
            resp = await client.get(clean_url, headers=headers)
            if resp.status_code == 200:
                html_text = resp.text
                m_desc = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']', html_text, re.I)
                if m_desc and not is_blacklisted_or_invalid(m_desc.group(1)):
                    return {
                        "success": True,
                        "text": m_desc.group(1).strip(),
                        "source": "meta_description",
                        "clean_url": clean_url
                    }
                og_desc = re.search(r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']', html_text, re.I)
                if og_desc and not is_blacklisted_or_invalid(og_desc.group(1)):
                    return {
                        "success": True,
                        "text": og_desc.group(1).strip(),
                        "source": "og_description",
                        "clean_url": clean_url
                    }
    except Exception:
        pass

    return {
        "success": False,
        "text": None,
        "source": None,
        "clean_url": clean_url
    }


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
