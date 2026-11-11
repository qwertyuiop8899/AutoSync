import asyncio
import base64
import json
import logging
import os
import re
import socket
import subprocess
import urllib.parse
from urllib.parse import urljoin

import httpx

from vx_extractor import TMDB_API_KEY, get_tmdb_id_for_imdb

logger = logging.getLogger(__name__)

MOVY_SITE = "https://www.movy.sx"
DEFAULT_API_BASE = "https://api.wecollege.net"
MOVY_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
SERVERS = ["miami", "boise", "phoenix", "atlanta", "orlando"]
QUALITIES = [2160, 1080, 720]


def get_warp_proxy() -> str:
    """Resolve the WARP SOCKS5 proxy URL."""
    env_proxy = os.getenv("AUTOSYNC_WARP_PROXY", "").strip()
    if env_proxy:
        return env_proxy

    # Try local ports / known container addresses
    candidates = [
        ("127.0.0.1", 1080),
        ("172.20.0.2", 1080),
        ("warp-isolated", 1080),
        ("warp", 1080),
    ]
    for host, port in candidates:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return f"socks5://{host}:{port}"
        except Exception:
            pass

    # Try docker inspect if running on host
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$v.IPAddress}}{{end}}", "warp-isolated"],
            stderr=subprocess.DEVNULL, timeout=1.0
        ).decode().strip()
        if out:
            return f"socks5://{out}:1080"
    except Exception:
        pass

    return "socks5://127.0.0.1:1080"


def movy_headers(site: str = MOVY_SITE) -> dict[str, str]:
    return {
        "User-Agent": MOVY_UA,
        "Accept": "*/*",
        "Origin": site,
        "Referer": f"{site}/",
    }


def quality_of(label: str) -> int | None:
    val = label.lower()
    if "4k" in val or "2160" in val:
        return 2160
    m = re.search(r"(\d{3,4})p", val)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Media key parsing & TMDB resolution
# ---------------------------------------------------------------------------

async def parse_media_key(media_key: str) -> tuple[bool, int, str, int, int, str, str]:
    """
    Parse media_key string into:
    (is_movie, tmdb_id, imdb_id, season, episode, title, year)
    """
    key = str(media_key or "").strip()
    parts = key.split(":")
    is_movie = False
    imdb_id = ""
    tmdb_id = 0
    season = 1
    episode = 1

    if key.startswith("tmdb:"):
        # e.g. tmdb:movie:157336 or tmdb:tv:113962:2:2
        is_movie = (parts[1] == "movie")
        tmdb_id = int(parts[2])
        if len(parts) >= 5 and not is_movie:
            season = int(parts[3])
            episode = int(parts[4])
    elif len(parts) >= 4:
        # e.g. series:tt13111078:2:2 or movie:tt0816692:0:0
        is_movie = (parts[0].lower() == "movie")
        raw_id = parts[1]
        season = int(parts[2]) if not is_movie else 0
        episode = int(parts[3]) if not is_movie else 0
        if raw_id.startswith("tt"):
            imdb_id = raw_id
            tmdb_id = await resolve_tmdb_id(imdb_id, is_movie)
        elif raw_id.isdigit():
            tmdb_id = int(raw_id)
    elif len(parts) >= 2:
        # e.g. movie:tt0816692 or movie:157336
        is_movie = (parts[0].lower() == "movie")
        raw_id = parts[1]
        season = 0
        episode = 0
        if raw_id.startswith("tt"):
            imdb_id = raw_id
            tmdb_id = await resolve_tmdb_id(imdb_id, is_movie)
        elif raw_id.isdigit():
            tmdb_id = int(raw_id)

    if not tmdb_id:
        raise ValueError(f"Could not resolve TMDB ID for media_key: {media_key}")

    title, year, fetched_imdb = await fetch_tmdb_metadata(tmdb_id, is_movie)
    if not imdb_id and fetched_imdb:
        imdb_id = fetched_imdb

    return is_movie, tmdb_id, imdb_id, season, episode, title, year


async def resolve_tmdb_id(imdb_id: str, is_movie: bool) -> int:
    """Resolve IMDB ID (e.g. tt13111078) to integer TMDB ID."""
    # First try vx_extractor helper
    try:
        resolved = await get_tmdb_id_for_imdb(imdb_id, is_movie)
        if resolved:
            return resolved
    except Exception:
        pass

    # Try db.wecollege.net directly via WARP (doesn't require api key)
    warp_proxy = get_warp_proxy()
    try:
        async with httpx.AsyncClient(proxy=warp_proxy, timeout=10.0) as client:
            url = f"https://db.wecollege.net/3/find/{imdb_id}?external_source=imdb_id"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("movie_results" if is_movie else "tv_results") or []
                if items:
                    return int(items[0]["id"])
    except Exception:
        pass

    # Try api.themoviedb.org
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_API_KEY}&external_source=imdb_id"
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("movie_results" if is_movie else "tv_results") or []
                if items:
                    return int(items[0]["id"])
    except Exception:
        pass

    raise RuntimeError(f"Failed resolving TMDB ID for IMDB ID: {imdb_id}")


async def fetch_tmdb_metadata(tmdb_id: int, is_movie: bool) -> tuple[str, str, str]:
    """Fetch title, year, and imdb_id for a given TMDB ID."""
    warp_proxy = get_warp_proxy()
    media_type = "movie" if is_movie else "tv"

    # Try db.wecollege.net via WARP
    try:
        async with httpx.AsyncClient(proxy=warp_proxy, timeout=10.0) as client:
            resp = await client.get(f"https://db.wecollege.net/3/{media_type}/{tmdb_id}")
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title" if is_movie else "name") or data.get("original_title" if is_movie else "original_name") or ""
                release_date = data.get("release_date" if is_movie else "first_air_date") or ""
                year = release_date[:4] if len(release_date) >= 4 else ""
                imdb = data.get("imdb_id") or ""
                return title, year, imdb
    except Exception:
        pass

    # Try api.themoviedb.org
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}?api_key={TMDB_API_KEY}")
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title" if is_movie else "name") or data.get("original_title" if is_movie else "original_name") or ""
                release_date = data.get("release_date" if is_movie else "first_air_date") or ""
                year = release_date[:4] if len(release_date) >= 4 else ""
                imdb = data.get("imdb_id") or ""
                return title, year, imdb
    except Exception:
        pass

    return "", "", ""


# ---------------------------------------------------------------------------
# Movy Cipher Decryption
# ---------------------------------------------------------------------------

def _int32(x: int) -> int:
    x = x & 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return x


def _uint32(x: int) -> int:
    return x & 0xFFFFFFFF


def _ushr(x: int, shift: int) -> int:
    return (x & 0xFFFFFFFF) >> shift


def _rotl(value: int, shift: int) -> int:
    t = shift & 31
    if t == 0:
        return _int32(value)
    u = _uint32(value)
    return _int32((u << t) | (u >> (32 - t)))


def _uhash(value: int) -> int:
    e = _int32(value)
    e = _int32(e ^ _ushr(e, 16))
    e = _int32(e * _int32(0x85EBCA6B))
    e = _int32(e ^ _ushr(e, 13))
    e = _int32(e * _int32(0xC2B2AE35))
    e = _int32(e ^ _ushr(e, 16))
    return e


class _CipherState:
    def __init__(self, s: list[int], acc: int, assigned: list[bool]):
        self.s = s
        self.acc = acc
        self.assigned = assigned


def _init_key(seed: str, media_id: int) -> _CipherState:
    h = _int32(0x811C9DC5)
    for ch in seed:
        h = _int32((h ^ ord(ch)) * 0x1000193)
    i = _uhash(_uhash(h) ^ _uhash(_int32(media_id ^ _int32(0x9E3779B9))))
    s = [0] * 61
    assigned = [False] * 61
    for e in range(8):
        t = _uint32(i) % 61
        i = _rotl(_int32(i + _int32(0x9E3779B9)), 7 + (7 & e))
        s[t] = _int32(i ^ _uhash(i))
        assigned[t] = True
        i = _uhash(_int32(i + t))
    return _CipherState(s, _uhash(_int32(_int32(0xA5A5A5A5) ^ i)), assigned)


def _next_word(state: _CipherState, round_num: int) -> int:
    idx = _uint32(state.acc) % 61
    in_bounds = -1 if state.assigned[idx] else 0
    s = _int32(state.s[idx] ^ _int32(_int32(0x9E3779B9) * (round_num + 1)))
    c = _int32((state.acc ^ s) | (state.acc & s & in_bounds))
    c = _int32(_rotl(_int32(c + state.acc), idx) ^ _rotl(state.acc, idx * 7))
    acc = _uhash(_int32(c + _int32(0x9E3779B9)))
    state.acc = acc
    state.s[idx] = acc
    state.assigned[idx] = True
    return acc


def decrypt(payload: str, seed: str, media_id: int) -> dict | None:
    """Decrypt Movy response payload."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    clean = payload.strip().replace("-", "+").replace("_", "/")
    clean = "".join(c for c in clean if c in alphabet)
    pad = len(clean) % 4
    if pad:
        clean += "=" * (4 - pad)
    raw = bytearray(base64.b64decode(clean))
    if len(raw) < 5:
        return None
    state = _init_key(seed, media_id)
    out = bytearray(len(raw))
    i = 0
    round_num = 0
    while i < len(raw):
        word = _next_word(state, round_num)
        b = 0
        while b < 4 and i < len(raw):
            w_byte = (_ushr(word, 8 * b)) & 255
            out[i] = raw[i] ^ w_byte
            i += 1
            b += 1
        round_num += 1
    if out[:4] != b"mvm1":
        return None
    try:
        text = out[4:].decode("utf-8")
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Error parsing decrypted JSON: {e}")
        return None


# ---------------------------------------------------------------------------
# Movy API Operations
# ---------------------------------------------------------------------------

async def discover_api(client: httpx.AsyncClient, site: str = MOVY_SITE) -> str:
    """Scan Movy site HTML for updated API endpoint."""
    try:
        resp = await client.get(f"{site}/", headers={"User-Agent": MOVY_UA, "Accept": "text/html"}, timeout=15.0)
        if resp.status_code == 200:
            matches = re.findall(r"https://api\.[a-z0-9-]+(?:\.[a-z0-9-]+)+", resp.text, re.IGNORECASE)
            for m in matches:
                if not re.search(r"tmdb|themoviedb|google|cloudflare", m, re.IGNORECASE):
                    return m.rstrip("/")
    except Exception:
        pass
    return DEFAULT_API_BASE


async def get_seed(client: httpx.AsyncClient, media_id: int, api_base: str = DEFAULT_API_BASE) -> str | None:
    """Fetch Movy session seed for media_id."""
    headers = movy_headers()
    url = f"{api_base}/seed?mediaId={media_id}"
    try:
        resp = await client.get(url, headers=headers, timeout=15.0)
        if resp.status_code == 200:
            data = resp.json()
            seed = str(data.get("seed") or "").strip()
            if seed:
                return seed
    except Exception as e:
        logger.warning(f"Failed getting seed from {url}: {e}")

    # Fallback to discoverApi
    discovered = await discover_api(client)
    if discovered and discovered != api_base:
        try:
            resp = await client.get(f"{discovered}/seed?mediaId={media_id}", headers=headers, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                seed = str(data.get("seed") or "").strip()
                if seed:
                    return seed
        except Exception:
            pass

    return None


async def fetch_sources(
    client: httpx.AsyncClient,
    server: str,
    media_id: int,
    season: int,
    episode: int,
    seed: str,
    is_movie: bool = False,
    imdb_id: str = "",
    title: str = "",
    year: str = "",
    api_base: str = DEFAULT_API_BASE,
) -> str | None:
    """Query server sources endpoint for encrypted media stream."""
    headers = movy_headers()
    params = {
        "title": title,
        "mediaType": "movie" if is_movie else "tv",
        "year": str(year) if year else "",
        "tmdbId": str(media_id),
        "imdbId": imdb_id or "",
        "enc": "2",
        "seed": seed,
    }
    if not is_movie:
        params["seasonId"] = str(season or 1)
        params["episodeId"] = str(episode or 1)

    url = f"{api_base}/{server}/sources?{urllib.parse.urlencode(params)}"
    try:
        resp = await client.get(url, headers=headers, timeout=18.0)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip()
    except Exception as e:
        logger.debug(f"fetch_sources failed on {server}: {e}")
    return None


async def resolve_video(client: httpx.AsyncClient, url: str, quality: int, headers: dict) -> tuple[str, int] | None:
    """Resolve m3u8 playlist: if master playlist, pick best variant <= quality; if media, return directly."""
    try:
        req_headers = {**headers, "Range": "bytes=0-65535"}
        resp = await client.get(url, headers=req_headers, timeout=15.0)
        if resp.status_code not in (200, 206):
            return None
        body = resp.text
    except Exception:
        return None

    if "#EXT-X-STREAM-INF" in body:
        variants = []
        lines = body.splitlines()
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            if not line.startswith("#EXT-X-STREAM-INF:"):
                continue
            m = re.search(r"RESOLUTION=(\d+)x\d+", line)
            width = int(m.group(1)) if m else 0
            uri = lines[i + 1].strip()
            if not uri or uri.startswith("#"):
                continue
            if width >= 3200:
                q = 2160
            elif width >= 2400:
                q = 1440
            elif width >= 1800:
                q = 1080
            elif width >= 1200:
                q = 720
            elif width >= 800:
                q = 480
            elif width > 0:
                q = 360
            else:
                q = quality
            variants.append((urljoin(url, uri), q))

        if not variants:
            return None
        matching = [v for v in variants if v[1] <= quality]
        if matching:
            return max(matching, key=lambda v: v[1])
        return min(variants, key=lambda v: v[1])

    if "#EXTINF" in body:
        return url, quality

    return None


# ---------------------------------------------------------------------------
# Main Extraction Entrypoint
# ---------------------------------------------------------------------------

async def extract_movy_stream(
    media_key: str,
    preferred_server: str | None = None,
    warp_proxy: str | None = None,
) -> list[dict]:
    """
    Autonomously extract fresh Movy stream renditions via WARP proxy.
    Returns list of rendition dicts:
    [{"resolution": quality, "url": stream_url, "headers": headers, "server": server}, ...]
    """
    proxy = warp_proxy or get_warp_proxy()
    is_movie, tmdb_id, imdb_id, season, episode, title, year = await parse_media_key(media_key)

    headers = movy_headers()
    async with httpx.AsyncClient(proxy=proxy, timeout=20.0, follow_redirects=True) as client:
        seed = await get_seed(client, tmdb_id)
        if not seed:
            raise RuntimeError(f"Could not retrieve Movy seed for tmdb_id={tmdb_id}")

        servers_to_try = list(SERVERS)
        if preferred_server:
            pref = preferred_server.strip().lower()
            if pref in servers_to_try:
                servers_to_try.remove(pref)
                servers_to_try.insert(0, pref)
            else:
                servers_to_try.insert(0, pref)

        sources = []
        have_qualities = set()

        for server in servers_to_try:
            if 2160 in have_qualities and 1080 in have_qualities:
                break

            payload = await fetch_sources(
                client=client,
                server=server,
                media_id=tmdb_id,
                season=season,
                episode=episode,
                seed=seed,
                is_movie=is_movie,
                imdb_id=imdb_id,
                title=title,
                year=year,
            )
            if not payload:
                continue

            data = decrypt(payload, seed, tmdb_id)
            if not data or not data.get("sources"):
                continue

            server_sources = []
            for node in data.get("sources", []):
                raw_url = str(node.get("url") or "").strip()
                if not raw_url or ".m3u8" not in raw_url.lower():
                    continue
                q = quality_of(str(node.get("quality") or ""))
                if not q or q not in QUALITIES or q in have_qualities:
                    continue

                resolved = await resolve_video(client, raw_url, q, headers)
                if not resolved:
                    continue

                stream_url, resolved_q = resolved
                if resolved_q in have_qualities:
                    continue

                have_qualities.add(resolved_q)
                server_sources.append({
                    "resolution": resolved_q,
                    "url": stream_url,
                    "headers": {
                        "User-Agent": MOVY_UA,
                        "Origin": MOVY_SITE,
                        "Referer": f"{MOVY_SITE}/",
                    },
                    "server": server,
                })

            if server_sources:
                sources.extend(server_sources)
                if any(s["resolution"] >= 1080 for s in server_sources):
                    break

        if not sources:
            raise RuntimeError(f"No working Movy streams found for {media_key} across servers {servers_to_try}")

        sources.sort(key=lambda s: s["resolution"], reverse=True)
        return sources
