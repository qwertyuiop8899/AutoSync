import hashlib
import re
from urllib.parse import urljoin, urlparse


def video_source_fingerprint(source: dict) -> str:
    """Stable video fingerprint matching ToastFlix main.py:361."""
    parsed = urlparse(str(source.get("url") or ""))
    provider = str(source.get("provider") or "nuvio").strip().lower()
    server = str(source.get("server") or "").strip().lower()
    path = parsed.path
    if provider in ("vidfast", "movy"):
        # Vidfast and Movy put a rotating session token between /vd/ or /vdb/ and the rendition.
        # It must not invalidate the offset cache for the same server/rendition.
        match = re.search(r"(/index-s\d+p[^/]*)$", path, re.IGNORECASE)
        path = (match.group(1) if match else path.rsplit("/", 1)[-1]).lower()
    elif provider == "strigil":
        # Strigil rotates the opaque session segment on every playback URL.
        # Keep the title/rendition identity stable so a measured offset is reusable.
        match = re.match(r"^(/vp/[^/]+)/[^/]+(/.*)$", path, re.IGNORECASE)
        if match:
            path = f"{match.group(1)}{match.group(2)}".lower()
    elif provider == "cineby":
        # Cineby rotates the opaque media token in both current URL shapes.
        match = re.match(r"^/vd/[^/]+(/.*)$", path, re.IGNORECASE)
        if match:
            path = match.group(1).lower()
        else:
            match = re.match(r"^(/r2/cdn1)/[^/]+(/.*)$", path, re.IGNORECASE)
            if match:
                path = f"{match.group(1)}{match.group(2)}".lower()
    stable = (
        f"strigil-v3|{provider}|{server}|{path}"
        if provider == "strigil" else f"{provider}|{server}|{path}"
    )
    return hashlib.sha1(stable.encode()).hexdigest()[:20]


def audio_source_fingerprint(playlist: str, base_url: str = "") -> str:
    """Stable audio source fingerprint matching ToastFlix audio.register / /dual/aprep."""
    lines = [line.strip() for line in (playlist or "").splitlines() if line.strip()]
    segments = []
    for line in lines:
        if line.startswith("#"):
            continue
        seg_url = urljoin(base_url, line) if base_url else line
        segments.append(seg_url)
    stable = [urlparse(url).path for url in segments[:3]]
    return hashlib.sha1(("|".join(stable) + str(len(segments))).encode()).hexdigest()[:20]


def offset_cache_key(media_key: str, resolution: int, video_fp: str, audio_fp: str) -> str:
    """Exact cache_key matching ToastFlix dual_offsets lookup."""
    return hashlib.sha1(f"{media_key}|{resolution}|{video_fp}|{audio_fp}".encode()).hexdigest()[:20]
