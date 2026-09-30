import os
import re
import base64
import asyncio
import logging
from urllib.parse import urlparse, parse_qsl, urlencode, urljoin
import httpx
from curl_cffi.requests import AsyncSession

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "68e094699525b18a70bab2f86b1fa706")


async def get_tmdb_id_for_imdb(imdb: str, is_movie: bool) -> int | None:
    url = f"https://api.themoviedb.org/3/find/{imdb}?api_key={TMDB_API_KEY}&external_source=imdb_id"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("movie_results" if is_movie else "tv_results") or []
            if items:
                return items[0].get("id")
    return None


async def _extract_with_session(session: AsyncSession, base_host: str, page_ref: str, api_path: str, media_key: str) -> list[dict]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    # 1. API
    api_url = f"{base_host}{api_path}"
    resp_api = await session.get(api_url, headers={**headers, "Accept": "application/json", "Referer": page_ref})
    if resp_api.status_code != 200:
        raise RuntimeError(f"API status {resp_api.status_code}")
    api_json = resp_api.json()
    embed_path = api_json.get("src")
    if not embed_path:
        raise RuntimeError(f"No embed src in API response")

    # 2. Embed
    embed_url = urljoin(base_host, embed_path)
    resp_embed = await session.get(embed_url, headers={**headers, "Referer": page_ref})
    if resp_embed.status_code != 200:
        raise RuntimeError(f"Embed status {resp_embed.status_code}")
    embed_html = resp_embed.text

    master_match = re.search(
        r"window\.masterPlaylist\s*=\s*\{.*?params\s*:\s*\{(?P<params>.*?)\}\s*,\s*url\s*:\s*['\"](?P<url>[^'\"]+)['\"]",
        embed_html,
        re.DOTALL
    )
    if not master_match:
        raise RuntimeError(f"masterPlaylist block not found in embed {embed_url}")

    params_block = master_match.group("params")
    playlist_base = master_match.group("url").replace("\\/", "/").replace("\\", "")
    token_m = re.search(r"['\"]token['\"]\s*:\s*['\"]([^'\"]+)['\"]", params_block)
    expires_m = re.search(r"['\"]expires['\"]\s*:\s*['\"](\d+)['\"]", params_block)
    asn_m = re.search(r"['\"]asn['\"]\s*:\s*['\"]([^'\"]*)['\"]", params_block)

    if not token_m or not expires_m:
        raise RuntimeError("Missing token or expires in Vixsrc masterPlaylist params")

    token = token_m.group(1)
    expires = expires_m.group(1)
    asn = asn_m.group(1) if asn_m else ""

    parsed_pl = urlparse(playlist_base)
    q = parse_qsl(parsed_pl.query, keep_blank_values=True)
    q.extend([("token", token), ("expires", expires), ("lang", "it")])
    if "window.canPlayFHD = true" in embed_html or "canPlayFHD = true" in embed_html:
        q.append(("h", "1"))
    if asn:
        q.append(("asn", asn))

    master_url = parsed_pl._replace(query=urlencode(q)).geturl()

    # 3. Master
    resp_master = await session.get(master_url, headers={**headers, "Referer": embed_url})
    if resp_master.status_code != 200 or "#EXTM3U" not in resp_master.text:
        raise RuntimeError(f"Master status {resp_master.status_code}")
    master_text = resp_master.text

    # Parse audio tracks
    ita_uri, eng_uri = None, None
    for line in master_text.splitlines():
        if line.startswith("#EXT-X-MEDIA:TYPE=AUDIO"):
            low = line.lower()
            m = re.search(r'URI="([^"]+)"', line)
            if not m:
                continue
            if not ita_uri and any(k in low for k in ('language="ita"', 'language="it"', 'name="italian"')):
                ita_uri = m.group(1)
            elif not eng_uri and any(k in low for k in ('language="eng"', 'language="en"', 'name="english"')):
                eng_uri = m.group(1)

    if not ita_uri:
        raise RuntimeError("Italian audio track not found in master")

    async def fetch_track(uri: str, lang: str) -> dict | None:
        try:
            track_url = urljoin(master_url, uri)
            r_track = await session.get(track_url, headers={**headers, "Referer": embed_url})
            if r_track.status_code != 200:
                logger.warning(f"Audio track {lang} HTTP {r_track.status_code}")
                return None
            pl_text = r_track.text
            key_m = re.search(r'#EXT-X-KEY:METHOD=AES-128,URI="([^"]+)"', pl_text)
            key_b64 = ""
            if key_m:
                key_url = urljoin(track_url, key_m.group(1))
                r_key = await session.get(key_url, headers={**headers, "Referer": embed_url})
                if r_key.status_code == 200 and len(r_key.content) == 16:
                    key_b64 = base64.b64encode(r_key.content).decode("ascii")

            return {
                "playlist": pl_text,
                "key": key_b64,
                "mediaKey": media_key,
                "lang": lang,
                "baseUrl": track_url,
                "headers": {"User-Agent": headers["User-Agent"], "Referer": embed_url},
                "source": "vixsrc",
            }
        except Exception as e:
            logger.warning(f"Failed fetching {lang} track: {e}")
            return None

    ita_payload, eng_payload = await asyncio.gather(
        fetch_track(ita_uri, "ita"),
        fetch_track(eng_uri, "eng") if eng_uri else asyncio.sleep(0, result=None),
    )

    if not ita_payload:
        raise RuntimeError("Failed extracting complete Italian audio payload")

    return [ita_payload, eng_payload] if eng_payload else [ita_payload]


async def resolve_vixsrc_tracks(media_key: str, tor_proxy: str = "") -> list[dict]:
    parts = media_key.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid media_key {media_key}")
    m_type, imdb, s_str, e_str = parts
    season, episode = int(s_str), int(e_str)
    is_movie = (m_type == "movie")

    tmdb_id = await get_tmdb_id_for_imdb(imdb, is_movie)
    if not tmdb_id:
        raise RuntimeError(f"Could not resolve TMDB ID for {imdb}")

    base_host = "https://vixsrc.to"
    api_path = f"/api/movie/{tmdb_id}" if is_movie else f"/api/tv/{tmdb_id}/{season}/{episode}"
    page_ref = f"{base_host}/movie/{tmdb_id}" if is_movie else f"{base_host}/tv/{tmdb_id}/{season}/{episode}"

    proxy_url = tor_proxy or os.getenv("AUTOSYNC_PROXY", "socks5://tor-toast-1:9050")
    proxy_dict = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    async with AsyncSession(impersonate="chrome124", timeout=15, proxies=proxy_dict) as session:
        return await _extract_with_session(session, base_host, page_ref, api_path, media_key)
