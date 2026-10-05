import asyncio
import math
import os
import re
import shutil
import statistics
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import numpy as np

from fingerprints import audio_source_fingerprint, video_source_fingerprint
from security import resolves_publicly, valid_public_url


# Speed hypotheses to evaluate (resampling factors)
SPEED_HYPOTHESES = (
    1.0,
    1000.0 / 1001.0,  # 24.000 -> 23.976 fps
    1001.0 / 1000.0,  # 23.976 -> 24.000 fps
    24.0 / 25.0,      # PAL -> Cinema
    25.0 / 24.0,      # Cinema -> PAL
    23.976 / 25.0,    # PAL -> NTSC
    25.0 / 23.976,    # NTSC -> PAL
)


def envelope_log100(samples: np.ndarray, sr: int = 8000) -> np.ndarray:
    """Compute 100 Hz logarithmic envelope (25 ms window, 10 ms step) with z-score normalization."""
    samples = np.asarray(samples, dtype=np.float32)
    win_len = int(sr * 0.025)  # 25 ms = 200 samples
    step = int(sr * 0.010)     # 10 ms = 80 samples
    if len(samples) < win_len:
        return np.zeros(0, dtype=np.float32)

    abs_s = np.abs(samples)
    cumsum = np.pad(np.cumsum(abs_s, dtype=np.float64), (1, 0))
    n_frames = (len(samples) - win_len) // step + 1
    starts = np.arange(n_frames) * step
    ends = starts + win_len
    means = (cumsum[ends] - cumsum[starts]) / win_len
    env = np.log1p(means)

    std = float(np.std(env))
    if std > 1e-6:
        env = (env - float(np.mean(env))) / std
    return env.astype(np.float32)


def envelope_lowpass(env: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Smooth envelope to emphasize bass / music / effects."""
    if len(env) < window_size:
        return env
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    smoothed = np.convolve(env, kernel, mode="same")
    std = float(np.std(smoothed))
    if std > 1e-6:
        smoothed = (smoothed - float(np.mean(smoothed))) / std
    return smoothed.astype(np.float32)


def resample_envelope(env: np.ndarray, k: float) -> np.ndarray:
    """Resample envelope by factor k to match different playback speeds."""
    if abs(k - 1.0) < 1e-5:
        return env
    m = len(env)
    m_k = int(round(m * k))
    if m_k <= 1:
        return env
    x_old = np.arange(m)
    x_new = np.linspace(0, m - 1, m_k)
    return np.interp(x_new, x_old, env).astype(np.float32)


def cross_correlate_valid(ref_env: np.ndarray, cand_env: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation via FFT in 'valid' mode (full overlap)."""
    n = len(ref_env)
    m = len(cand_env)
    if n < m or m == 0:
        return np.zeros(0, dtype=np.float32)

    cand_norm = cand_env - float(np.mean(cand_env))
    cand_std = float(np.std(cand_env))
    if cand_std < 1e-6:
        return np.zeros(n - m + 1, dtype=np.float32)
    cand_norm /= (cand_std * math.sqrt(m))

    # Fast convolution via FFT: conv(ref, cand_norm[::-1])
    n_conv = n - m + 1
    fft_size = 1 << ((n + m - 1).bit_length())
    f_ref = np.fft.rfft(ref_env, fft_size)
    f_cand = np.fft.rfft(cand_norm[::-1], fft_size)
    raw_conv = np.fft.irfft(f_ref * f_cand, fft_size)[m - 1 : n]

    # Local window standard deviation of ref_env
    ref_cumsum = np.pad(np.cumsum(ref_env, dtype=np.float64), (1, 0))
    ref_sq_cumsum = np.pad(np.cumsum(ref_env.astype(np.float64) ** 2, dtype=np.float64), (1, 0))
    starts = np.arange(n_conv)
    ends = starts + m
    ref_mean = (ref_cumsum[ends] - ref_cumsum[starts]) / m
    ref_var = (ref_sq_cumsum[ends] - ref_sq_cumsum[starts]) / m - ref_mean ** 2
    ref_std = np.sqrt(np.maximum(ref_var, 1e-12))

    corr = raw_conv / (ref_std * math.sqrt(m))
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def calculate_psr(corr: np.ndarray, peak_idx: int, exclude_radius: int = 100) -> float:
    """Peak Sharpness Ratio = peak / max secondary peak outside +/- 1.0s (100 frames)."""
    if len(corr) == 0:
        return 0.0
    peak_val = float(corr[peak_idx])
    if peak_val <= 0.0:
        return 0.0
    left_end = max(0, peak_idx - exclude_radius)
    right_start = min(len(corr), peak_idx + exclude_radius + 1)

    sec_peaks = []
    if left_end > 0:
        sec_peaks.append(float(np.max(corr[:left_end])))
    if right_start < len(corr):
        sec_peaks.append(float(np.max(corr[right_start:])))
    if not sec_peaks:
        return 999.0
    second_peak = max(0.0, max(sec_peaks))
    if second_peak < 1e-6:
        return 999.0
    return peak_val / second_peak


def parabolic_peak(corr: np.ndarray, peak_idx: int, step: float = 0.01) -> tuple[float, float]:
    """Sub-frame refinement of peak using 3-point parabolic interpolation."""
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return float(peak_idx * step), float(corr[peak_idx])
    y_prev = float(corr[peak_idx - 1])
    y_curr = float(corr[peak_idx])
    y_next = float(corr[peak_idx + 1])
    denom = 2.0 * (y_prev - 2.0 * y_curr + y_next)
    if abs(denom) < 1e-12:
        return float(peak_idx * step), y_curr
    delta = (y_prev - y_next) / denom
    refined_t = (peak_idx + delta) * step
    refined_val = y_curr - 0.25 * (y_prev - y_next) * delta
    return float(refined_t), float(refined_val)


def gcc_phat(sig_ref: np.ndarray, sig_cand: np.ndarray, sr: int = 8000, max_tau_ms: float = 50.0) -> float:
    """GCC-PHAT on raw PCM for high-resolution timing refinement (+/- 50 ms)."""
    n = len(sig_ref) + len(sig_cand)
    fft_size = 1 << ((n - 1).bit_length())
    X_ref = np.fft.rfft(sig_ref, fft_size)
    X_cand = np.fft.rfft(sig_cand, fft_size)
    cross = X_ref * np.conj(X_cand)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12
    norm_cross = cross / denom
    cc = np.fft.irfft(norm_cross, fft_size)
    cc = np.concatenate((cc[-(len(cc) // 2) :], cc[: (len(cc) // 2)]))
    center = len(cc) // 2
    max_samples = int(sr * max_tau_ms / 1000.0)
    window = cc[center - max_samples : center + max_samples + 1]
    best_idx = int(np.argmax(window)) - max_samples
    return float(best_idx / sr)


def theil_sen(positions: list[float], offsets: list[float]) -> tuple[float, float]:
    """Robust linear regression via Theil-Sen estimator (slope s, intercept c)."""
    n = len(positions)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j] - positions[i]
            if abs(dx) > 1e-3:
                slopes.append((offsets[j] - offsets[i]) / dx)
    if not slopes:
        return 0.0, float(statistics.median(offsets))
    s = float(statistics.median(slopes))
    c = float(statistics.median([offsets[i] - s * positions[i] for i in range(n)]))
    return s, c


class OffsetEngine:
    """AutoSync Precision v2 Audio/Video Offset Measurement Engine."""

    VIDFAST_SAMPLE_RESOLUTIONS = (360, 480, 720, 1080)

    def __init__(self, proxy: str = ""):
        if isinstance(proxy, str):
            self.proxies = [p.strip() for p in proxy.split(",") if p.strip()]
        elif isinstance(proxy, (list, tuple)):
            self.proxies = list(proxy)
        else:
            self.proxies = []
        self.proxy = self.proxies[0] if self.proxies else ""
        self._proxy_idx = 0

    def _next_proxy(self) -> str | None:
        if not self.proxies:
            return None
        p = self.proxies[self._proxy_idx % len(self.proxies)]
        self._proxy_idx += 1
        return p

    def _needs_proxy(self, url: str) -> bool:
        """Video CDNs (Vidfast, Cinejoy, Moon, etc.) don't block datacenter IPs and download 10x faster directly.
        Vixsrc, Vidsrc, and Italian audio hosters use Cloudflare/geo-blocks and require Tor proxy rotation."""
        u = str(url).lower()
        if any(k in u for k in ("vixsrc", "vidsrc", "bravecastle", "sc-u11", "partite.cc", "storage/enc.key")):
            return True
        return False

    async def _get(self, url: str, headers: dict) -> httpx.Response:
        if not valid_public_url(url) or not await resolves_publicly(url):
            raise ValueError(f"media URL is not public HTTPS: {url}")

        use_proxy = self._needs_proxy(url) and bool(self.proxies)
        proxy = self._next_proxy() if use_proxy else None

        kwargs = {"timeout": 30.0, "follow_redirects": False}
        if proxy:
            kwargs["proxy"] = proxy

        try:
            async with httpx.AsyncClient(**kwargs) as client:
                response = await client.get(url, headers=headers)
        except Exception:
            if not proxy and self.proxies:
                proxy = self._next_proxy()
                kwargs["proxy"] = proxy
                async with httpx.AsyncClient(**kwargs) as client:
                    response = await client.get(url, headers=headers)
            else:
                raise

        if response.status_code == 403 and not proxy and self.proxies:
            for _ in range(len(self.proxies)):
                try:
                    p = self._next_proxy()
                    async with httpx.AsyncClient(proxy=p, timeout=20.0, follow_redirects=False) as client:
                        r = await client.get(url, headers=headers)
                        if r.status_code == 200:
                            return r
                except Exception:
                    pass

        if response.status_code in (301, 302, 307, 308):
            loc = response.headers.get("location", "")
            redirect_url = urljoin(url, loc)
            if not await resolves_publicly(redirect_url):
                raise ValueError("redirect target is not public HTTPS")
            return await self._get(redirect_url, headers)
        response.raise_for_status()
        return response

    @staticmethod
    def _parse_playlist(text: str, master_url: str):
        entries, pending, elapsed = [], None, 0.0
        map_url = None
        key_iv = None
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("#EXT-X-MAP:"):
                match = re.search(r'URI="([^"]+)"', line)
                map_url = urljoin(master_url, match.group(1)) if match else None
            elif line.startswith("#EXT-X-KEY:"):
                match_iv = re.search(r'IV=(0x[0-9A-Fa-f]+)', line)
                if match_iv:
                    key_iv = match_iv.group(1)
            elif line.startswith("#EXTINF:"):
                pending = float(line.split(":", 1)[1].split(",", 1)[0])
            elif pending is not None and line and not line.startswith("#"):
                entries.append({"url": urljoin(master_url, line), "duration": pending, "start": elapsed})
                elapsed += pending
                pending = None
        if not entries:
            raise ValueError("empty media playlist")
        return entries, map_url, key_iv

    async def _video_entries(self, url: str, headers: dict):
        resp = await self._get(url, headers)
        entries, map_url, _ = self._parse_playlist(resp.text, url)
        return entries, map_url

    async def _find_light_rendition(self, url: str, headers: dict, target_duration: float) -> str:
        """Find light 360/480/720p rendition if playlist duration matches within 1.0s."""
        match = re.search(r"index-s(\d+)p", url, re.IGNORECASE)
        if not match:
            return url
        curr_res = int(match.group(1))
        for res in self.VIDFAST_SAMPLE_RESOLUTIONS:
            if res >= curr_res:
                break
            candidate = re.sub(r"index-s\d+p", f"index-s{res}p", url, count=1, flags=re.IGNORECASE)
            try:
                entries, _ = await self._video_entries(candidate, headers)
                dur = sum(item["duration"] for item in entries)
                if abs(dur - target_duration) <= 1.0:
                    return candidate
            except Exception:
                continue
        return url

    async def _media_start_time(self, url: str, headers: dict) -> float:
        """Read container video start_time via ffprobe on first segment."""
        try:
            resp = await self._get(url, headers)
            match = re.search(r'#EXT-X-MAP:URI="([^"]+)"', resp.text)
            first_seg = next((l.strip() for l in resp.text.splitlines() if l.strip() and not l.startswith("#")), "")
            if not first_seg:
                return 0.0
            seg_url = urljoin(url, first_seg)
            if match:
                init_resp, seg_resp = await asyncio.gather(
                    self._get(urljoin(url, match.group(1)), headers),
                    self._get(seg_url, headers),
                )
                sample_bytes = init_resp.content + seg_resp.content
                ext = ".mp4"
            else:
                seg_resp = await self._get(seg_url, headers)
                sample_bytes = seg_resp.content
                ext = ".ts"

            with tempfile.TemporaryDirectory(prefix="ffprobe-vst-") as tmp:
                sample_file = Path(tmp) / f"sample{ext}"
                sample_file.write_bytes(sample_bytes)
                proc = await asyncio.create_subprocess_exec(
                    "ffprobe", "-v", "error", "-show_entries", "stream=start_time",
                    "-of", "default=nw=1:nk=1", str(sample_file),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
                vals = [float(x) for x in stdout.decode(errors="replace").strip().splitlines() if x.strip()]
                return round(vals[0], 3) if vals else 0.0
        except Exception:
            return 0.0

    async def _download_entire_audio_pcm(self, playlist_text: str, base_url: str,
                                         headers: dict, key_b64: str, out_pcm: Path) -> float:
        """Download entire HLS audio playlist into a single local mono 8kHz 16-bit PCM file."""
        entries, map_url, key_iv = self._parse_playlist(playlist_text, base_url)
        total_duration = sum(item["duration"] for item in entries)

        with tempfile.TemporaryDirectory(prefix="audio-full-dl-") as tmp:
            tmp_path = Path(tmp)
            seg_ext = ".m4s" if map_url else ".ts"
            m3u8_lines = ["#EXTM3U", "#EXT-X-VERSION:7" if map_url else "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-TARGETDURATION:15"]
            if map_url:
                init_content = (await self._get(map_url, headers)).content
                (tmp_path / "init.mp4").write_bytes(init_content)
                m3u8_lines.append('#EXT-X-MAP:URI="init.mp4"')

            if key_b64:
                import base64
                key_bytes = base64.b64decode(key_b64)
                key_path = tmp_path / "enc.key"
                key_path.write_bytes(key_bytes)
                iv_part = f",IV={key_iv}" if key_iv else ""
                m3u8_lines.append(f'#EXT-X-KEY:METHOD=AES-128,URI="enc.key"{iv_part}')

            # Download segments concurrently with retry (batch of 10)
            async def _fetch_seg(idx: int, seg_info: dict):
                for attempt in range(3):
                    try:
                        content = (await self._get(seg_info["url"], headers)).content
                        seg_file = tmp_path / f"seg-{idx:05d}{seg_ext}"
                        seg_file.write_bytes(content)
                        return
                    except Exception as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.5 * (attempt + 1))

            batch_size = 10
            for b in range(0, len(entries), batch_size):
                chunk = entries[b : b + batch_size]
                await asyncio.gather(*[_fetch_seg(b + i, item) for i, item in enumerate(chunk)])

            for i, item in enumerate(entries):
                m3u8_lines.append(f"#EXTINF:{item['duration']:.6f},")
                m3u8_lines.append(f"seg-{i:05d}{seg_ext}")
            m3u8_lines.append("#EXT-X-ENDLIST")

            local_m3u8 = tmp_path / "audio.m3u8"
            local_m3u8.write_text("\n".join(m3u8_lines) + "\n")

            # Convert to mono 8kHz s16le PCM via ffmpeg
            cmd = [
                "ffmpeg", "-v", "error", "-allowed_extensions", "ALL",
                "-protocol_whitelist", "file,crypto", "-i", str(local_m3u8),
                "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-y", str(out_pcm),
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
            if proc.returncode != 0 or not out_pcm.exists() or out_pcm.stat().st_size == 0:
                raise RuntimeError(f"full audio PCM decode failed: {stderr.decode(errors='replace')[:200]}")

        return total_duration

    async def _sample_video_pcm(self, video_url: str, video_headers: dict,
                                position: float, duration: float, out_pcm: Path):
        """Extract a short local PCM sample from video at specified position."""
        entries, map_url = await self._video_entries(video_url, video_headers)
        # Find entries covering [position, position + duration + 4.0]
        target = next((i for i, it in enumerate(entries) if it["start"] <= position < it["start"] + it["duration"]), len(entries) - 1)
        first = max(0, target - 1)
        local_seek = max(0.0, position - entries[first]["start"])
        selected = []
        avail = 0.0
        for it in entries[first:]:
            selected.append(it)
            avail += it["duration"]
            if avail >= local_seek + duration + 4.0:
                break

        seg_ext = ".m4s" if map_url else ".ts"
        with tempfile.TemporaryDirectory(prefix="vid-sample-") as tmp:
            tmp_path = Path(tmp)
            lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-TARGETDURATION:10"]
            if map_url:
                init_data = (await self._get(map_url, video_headers)).content
                (tmp_path / "init.mp4").write_bytes(init_data)
                lines.append('#EXT-X-MAP:URI="init.mp4"')

            async def _fetch_vseg(i: int, item: dict):
                for attempt in range(3):
                    try:
                        data = (await self._get(item["url"], video_headers)).content
                        (tmp_path / f"vseg-{i}{seg_ext}").write_bytes(data)
                        return
                    except Exception as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.5 * (attempt + 1))

            await asyncio.gather(*[_fetch_vseg(i, it) for i, it in enumerate(selected)])
            for i, it in enumerate(selected):
                lines.append(f"#EXTINF:{it['duration']:.6f},")
                lines.append(f"vseg-{i}{seg_ext}")
            lines.append("#EXT-X-ENDLIST")

            local_m3u8 = tmp_path / "video.m3u8"
            local_m3u8.write_text("\n".join(lines) + "\n")

            cmd = [
                "ffmpeg", "-v", "error", "-allowed_extensions", "ALL",
                "-protocol_whitelist", "file,crypto", "-i", str(local_m3u8),
                "-ss", f"{local_seek:.3f}", "-t", f"{duration:.3f}",
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-y", str(out_pcm),
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=45.0)
            if proc.returncode != 0 or not out_pcm.exists() or out_pcm.stat().st_size == 0:
                raise RuntimeError(f"video sample decode failed: {stderr.decode(errors='replace')[:200]}")

    async def _sample_audio_pcm(self, playlist_text: str, base_url: str,
                                headers: dict, key_b64: str,
                                position: float, duration: float, out_pcm: Path):
        """Extract a short local PCM sample from HLS audio at specified position without downloading entire stream."""
        entries, map_url, key_iv = self._parse_playlist(playlist_text, base_url)
        target = next((i for i, it in enumerate(entries) if it["start"] <= position < it["start"] + it["duration"]), len(entries) - 1)
        first = max(0, target - 1)
        local_seek = max(0.0, position - entries[first]["start"])
        selected = []
        avail = 0.0
        for it in entries[first:]:
            selected.append(it)
            avail += it["duration"]
            if avail >= local_seek + duration + 4.0:
                break

        seg_ext = ".m4s" if map_url else ".ts"
        with tempfile.TemporaryDirectory(prefix="aud-sample-") as tmp:
            tmp_path = Path(tmp)
            lines = ["#EXTM3U", "#EXT-X-VERSION:7" if map_url else "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-TARGETDURATION:15"]
            if map_url:
                init_data = (await self._get(map_url, headers)).content
                (tmp_path / "init.mp4").write_bytes(init_data)
                lines.append('#EXT-X-MAP:URI="init.mp4"')

            if key_b64:
                import base64
                key_bytes = base64.b64decode(key_b64)
                (tmp_path / "enc.key").write_bytes(key_bytes)
                iv_part = f",IV={key_iv}" if key_iv else ""
                lines.append(f'#EXT-X-KEY:METHOD=AES-128,URI="enc.key"{iv_part}')

            async def _fetch_aseg(i: int, item: dict):
                for attempt in range(3):
                    try:
                        data = (await self._get(item["url"], headers)).content
                        (tmp_path / f"aseg-{i}{seg_ext}").write_bytes(data)
                        return
                    except Exception as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(0.5 * (attempt + 1))

            await asyncio.gather(*[_fetch_aseg(i, it) for i, it in enumerate(selected)])
            for i, it in enumerate(selected):
                lines.append(f"#EXTINF:{it['duration']:.6f},")
                lines.append(f"aseg-{i}{seg_ext}")
            lines.append("#EXT-X-ENDLIST")

            local_m3u8 = tmp_path / "audio.m3u8"
            local_m3u8.write_text("\n".join(lines) + "\n")

            cmd = [
                "ffmpeg", "-v", "error", "-allowed_extensions", "ALL",
                "-protocol_whitelist", "file,crypto", "-i", str(local_m3u8),
                "-ss", f"{local_seek:.3f}", "-t", f"{duration:.3f}",
                "-map", "0:a:0?", "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-y", str(out_pcm),
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=45.0)
            if proc.returncode != 0 or not out_pcm.exists() or out_pcm.stat().st_size == 0:
                raise RuntimeError(f"audio sample decode failed: {stderr.decode(errors='replace')[:200]}")

    @staticmethod
    def _is_silent(pcm_samples: np.ndarray, min_rms: float = 120.0) -> bool:
        """Check if audio sample is silent or lacks dynamic energy."""
        if len(pcm_samples) == 0:
            return True
        rms = math.sqrt(float(np.mean(pcm_samples.astype(np.float64) ** 2)))
        return rms < min_rms

    async def measure(self, payload: dict) -> dict:
        """Measure offset between video and audio reference with Two-Tier sync (FastPass v2 Smart + Deep Search fallback)."""
        media_key = str(payload.get("media_key") or "")
        resolution = int(payload.get("resolution") or 1080)
        provider = str(payload.get("provider") or "").strip().lower()
        server = str(payload.get("server") or "").strip()
        video_url = str(payload.get("video_url") or "")
        video_headers = payload.get("video_headers") if isinstance(payload.get("video_headers"), dict) else {}
        audio_tracks = payload.get("audio_tracks") or []
        audio_source = str(payload.get("audio_source") or "vixsrc").strip().lower()

        # Fallback to single audio if passed old format
        if not audio_tracks and payload.get("audio_playlist"):
            audio_tracks = [{
                "lang": "ita",
                "playlist": payload.get("audio_playlist"),
                "key": payload.get("audio_key", ""),
                "base_url": payload.get("audio_base_url", ""),
                "headers": payload.get("audio_headers") or {},
                "source": audio_source,
            }]

        if not audio_tracks:
            return {"status": "inconclusive", "error": "No audio tracks provided"}

        # Prioritize ENG reference for higher correlation (>0.90) and no translation discrepancy
        eng_track = next((t for t in audio_tracks if t.get("lang") == "eng"), None)
        ita_track = next((t for t in audio_tracks if t.get("lang") == "ita"), None)
        primary_track = eng_track or ita_track or audio_tracks[0]
        is_eng = (primary_track.get("lang") == "eng")
        b_url = primary_track.get("base_url") or primary_track.get("baseUrl") or ""
        if not primary_track.get("playlist") and b_url:
            resp = await self._get(b_url, primary_track.get("headers") or {})
            primary_track["playlist"] = resp.text
        audio_fp = audio_source_fingerprint(primary_track.get("playlist", ""), b_url)

        # Parse audio playlist in memory to get duration in ~0.001s without downloading segments
        audio_entries, _, _ = self._parse_playlist(primary_track["playlist"], b_url)
        audio_duration = sum(item["duration"] for item in audio_entries)

        # Video metadata and light rendition
        video_entries, _ = await self._video_entries(video_url, video_headers)
        video_duration = sum(item["duration"] for item in video_entries)
        light_video_url = await self._find_light_rendition(video_url, video_headers, video_duration)
        video_start_time = await self._media_start_time(light_video_url, video_headers)

        common = min(video_duration, audio_duration)
        delta = audio_duration - video_duration
        credits_discrepancy = abs(delta) > 5.0

        if common < 90.0:
            return {
                "status": "incompatible",
                "error": f"Durata comune insufficiente ({common:.1f}s < 90s)",
                "video_duration": video_duration,
                "audio_duration": audio_duration,
            }

        audio_key = primary_track.get("key", "")
        audio_headers = primary_track.get("headers") or {}
        audio_pl = primary_track["playlist"]

        with tempfile.TemporaryDirectory(prefix="offset-engine-") as work_dir:
            work_path = Path(work_dir)

            # -------------------------------------------------------------
            # TIER 1: FASTPASS V2 SMART (Anchor 15% + Center 50%)
            # -------------------------------------------------------------
            anchor_base = min(900.0, max(120.0, 0.15 * common))
            anchor_pos = anchor_base
            anchor_sample_sec = 15.0
            window_sec = max(60.0, abs(delta) + 30.0)

            anchor_found = False
            best_anchor_lag = 0.0
            best_anchor_corr = 0.0
            best_k = 1.0

            min_corr_thresh = 0.60 if is_eng else 0.50

            ratio = audio_duration / video_duration
            sorted_k = sorted(SPEED_HYPOTHESES, key=lambda k: abs(k - ratio))

            for shift_count in range(4):
                cand_pcm_path = work_path / f"anchor_v_{shift_count}.pcm"
                ref_pcm_path = work_path / f"anchor_a_{shift_count}.pcm"

                try:
                    await self._sample_video_pcm(light_video_url, video_headers, anchor_pos, anchor_sample_sec, cand_pcm_path)
                    cand_pcm = np.fromfile(cand_pcm_path, dtype=np.int16)
                    if self._is_silent(cand_pcm) and shift_count < 3:
                        anchor_pos += 30.0
                        continue

                    # Slice audio window around anchor_pos (+/- window_sec)
                    aud_start = max(0.0, anchor_pos - window_sec)
                    aud_end = min(audio_duration, anchor_pos + anchor_sample_sec + window_sec)
                    aud_dur = aud_end - aud_start

                    await self._sample_audio_pcm(audio_pl, b_url, audio_headers, audio_key, aud_start, aud_dur, ref_pcm_path)
                    ref_pcm = np.fromfile(ref_pcm_path, dtype=np.int16)
                    if len(ref_pcm) == 0:
                        anchor_pos += 30.0
                        continue

                    ref_env = envelope_log100(ref_pcm)
                    cand_env_raw = envelope_log100(cand_pcm)

                    for k_hyp in sorted_k:
                        cand_env = resample_envelope(cand_env_raw, k_hyp)
                        corr = cross_correlate_valid(ref_env, cand_env)
                        if len(corr) == 0:
                            continue
                        pk_idx = int(np.argmax(corr))
                        pk_val = float(corr[pk_idx])
                        psr = calculate_psr(corr, pk_idx, exclude_radius=100)

                        if pk_val >= min_corr_thresh and psr >= 1.25:
                            refined_t, refined_corr = parabolic_peak(corr, pk_idx, step=0.01)
                            lag = (aud_start + refined_t) - anchor_pos

                            # Sub-millisecond GCC-PHAT refinement if ENG vs ENG
                            if is_eng:
                                aligned_a_start = int(refined_t * 8000)
                                if 0 <= aligned_a_start and aligned_a_start + 5 * 8000 <= len(ref_pcm):
                                    ref_5s = ref_pcm[aligned_a_start : aligned_a_start + 5 * 8000]
                                    cand_5s = cand_pcm[: 5 * 8000]
                                    tau_refine = gcc_phat(ref_5s, cand_5s, sr=8000, max_tau_ms=50.0)
                                    lag += tau_refine

                            best_anchor_lag = lag
                            best_anchor_corr = refined_corr
                            best_k = k_hyp
                            anchor_found = True
                            break

                    if anchor_found:
                        break
                    anchor_pos += 30.0
                except Exception as ex:
                    anchor_pos += 30.0

            # Step 1.2: Center verification (~50% duration)
            center_verified = False
            center_lag = 0.0
            best_center_corr = 0.0
            center_dur = 15.0
            center_pos = 0.50 * common
            if anchor_found:
                for shift_c in range(3):
                    center_pos = (0.50 * common) + shift_c * 30.0
                    if center_pos + center_dur > common:
                        break
                    v_center_path = work_path / f"center_v_{shift_c}.pcm"
                    a_center_path = work_path / f"center_a_{shift_c}.pcm"

                    try:
                        await self._sample_video_pcm(light_video_url, video_headers, center_pos, center_dur, v_center_path)
                        v_center_pcm = np.fromfile(v_center_path, dtype=np.int16)
                        if self._is_silent(v_center_pcm) and shift_c < 2:
                            continue

                        # Search window around expected lag (+/- 6.0s)
                        exp_ref_lag = best_anchor_lag + (best_k - 1.0) * (center_pos - anchor_pos)
                        aud_center_start = max(0.0, center_pos + exp_ref_lag - 6.0)
                        aud_center_dur = center_dur + 12.0
                        await self._sample_audio_pcm(audio_pl, b_url, audio_headers, audio_key, aud_center_start, aud_center_dur, a_center_path)
                        a_center_pcm = np.fromfile(a_center_path, dtype=np.int16)

                        if len(v_center_pcm) > 0 and len(a_center_pcm) > 0:
                            ref_env_c = envelope_log100(a_center_pcm)
                            cand_env_c_raw = envelope_log100(v_center_pcm)

                            for k_hyp in sorted_k:
                                cand_env_c = resample_envelope(cand_env_c_raw, k_hyp)
                                corr_c = cross_correlate_valid(ref_env_c, cand_env_c)
                                if len(corr_c) == 0:
                                    continue
                                c_pk_idx = int(np.argmax(corr_c))
                                c_pk_val = float(corr_c[c_pk_idx])
                                c_psr = calculate_psr(corr_c, c_pk_idx, exclude_radius=100)

                                if c_pk_val >= min_corr_thresh and c_psr >= 1.2:
                                    c_refined_t, c_refined_corr = parabolic_peak(corr_c, c_pk_idx, step=0.01)
                                    calc_c_lag = (aud_center_start + c_refined_t) - center_pos

                                    if is_eng:
                                        aligned_c_start = int(c_refined_t * 8000)
                                        if 0 <= aligned_c_start and aligned_c_start + 5 * 8000 <= len(a_center_pcm):
                                            ref_5s_c = a_center_pcm[aligned_c_start : aligned_c_start + 5 * 8000]
                                            cand_5s_c = v_center_pcm[: 5 * 8000]
                                            tau_refine_c = gcc_phat(ref_5s_c, cand_5s_c, sr=8000, max_tau_ms=50.0)
                                            calc_c_lag += tau_refine_c

                                    exp_lag = best_anchor_lag + (k_hyp - 1.0) * (center_pos - anchor_pos)
                                    dev = abs(calc_c_lag - exp_lag)
                                    if dev <= 0.080:  # 80ms strict tolerance
                                        center_lag = calc_c_lag
                                        best_center_corr = c_refined_corr
                                        best_k = k_hyp
                                        center_verified = True
                                        break

                        if center_verified:
                            break
                    except Exception as ex:
                        pass

            # If FastPass v2 Smart succeeded with high confidence:
            if anchor_found and center_verified:
                med_lag = 0.5 * (best_anchor_lag + center_lag)
                final_offset = round(-med_lag + video_start_time, 3)
                mean_conf = round(float(0.5 * (best_anchor_corr + best_center_corr)), 3)
                deviation = round(abs(best_anchor_lag - center_lag), 4)
                return {
                    "status": "ok",
                    "offset": final_offset,
                    "rate": float(best_k if abs(best_k - 1.0) > 0.001 else 1.0),
                    "confidence": mean_conf,
                    "sync_mode": "fastpass-v2",
                    "deviation": deviation,
                    "provider": provider,
                    "server": server,
                    "audio_source": audio_source,
                    "audio_fingerprint": audio_fp,
                    "video_duration": round(video_duration, 2),
                    "audio_duration": round(audio_duration, 2),
                    "video_start_time": round(video_start_time, 3),
                    "credits_discrepancy": credits_discrepancy,
                    "sync_algorithm": "autosync-v1",
                    "measurements": [
                        {"position": round(anchor_pos, 3), "duration": anchor_sample_sec, "lag": round(best_anchor_lag, 4), "correlation": round(best_anchor_corr, 3)},
                        {"position": round(center_pos, 3), "duration": center_dur, "lag": round(center_lag, 4), "correlation": round(best_center_corr, 3)},
                    ],
                }

            # -------------------------------------------------------------
            # TIER 2: DEEP SEARCH FALLBACK (7 verification points)
            # -------------------------------------------------------------
            ratios = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
            verify_sample_sec = 15.0
            measurements = []

            for i, r in enumerate(ratios):
                v_pos = r * common
                expected_lag = (best_anchor_lag + (best_k - 1.0) * (v_pos - anchor_pos)) if anchor_found else 0.0
                search_radii = (3.5, 60.0) if anchor_found else (60.0,)

                for search_radius in search_radii:
                    aud_pos = max(0.0, v_pos + expected_lag - search_radius)
                    aud_dur = verify_sample_sec + search_radius * 2.0
                    if aud_pos + aud_dur > audio_duration:
                        aud_dur = max(0.0, audio_duration - aud_pos)
                    if aud_dur < verify_sample_sec:
                        continue

                    cand_path = work_path / f"deep_v_{i}_{int(search_radius)}.pcm"
                    ref_path = work_path / f"deep_a_{i}_{int(search_radius)}.pcm"

                    try:
                        await self._sample_video_pcm(light_video_url, video_headers, v_pos, verify_sample_sec, cand_path)
                        cand_pcm = np.fromfile(cand_path, dtype=np.int16)
                        if self._is_silent(cand_pcm):
                            break

                        await self._sample_audio_pcm(audio_pl, b_url, audio_headers, audio_key, aud_pos, aud_dur, ref_path)
                        ref_pcm = np.fromfile(ref_path, dtype=np.int16)
                        if len(ref_pcm) == 0:
                            continue

                        ref_env = envelope_log100(ref_pcm)
                        cand_env = resample_envelope(envelope_log100(cand_pcm), best_k)
                        corr = cross_correlate_valid(ref_env, cand_env)
                        if len(corr) == 0:
                            continue

                        pk_idx = int(np.argmax(corr))
                        pk_val = float(corr[pk_idx])
                        psr = calculate_psr(corr, pk_idx)

                        if pk_val >= min_corr_thresh and psr >= 1.20:
                            refined_t, refined_corr = parabolic_peak(corr, pk_idx, step=0.01)
                            lag = (aud_pos + refined_t) - v_pos

                            if is_eng:
                                aligned_a_start = int(refined_t * 8000)
                                if 0 <= aligned_a_start and aligned_a_start + 5 * 8000 <= len(ref_pcm):
                                    ref_5s = ref_pcm[aligned_a_start : aligned_a_start + 5 * 8000]
                                    cand_5s = cand_pcm[: 5 * 8000]
                                    tau_refine = gcc_phat(ref_5s, cand_5s, sr=8000, max_tau_ms=50.0)
                                    lag += tau_refine

                            measurements.append({
                                "position": round(v_pos, 3),
                                "duration": verify_sample_sec,
                                "lag": round(lag, 4),
                                "correlation": round(refined_corr, 3),
                                "psr": round(psr, 2),
                            })
                            break
                    except Exception:
                        continue

            # Step 3: Classification of Tier 2 measurements
            valid_pts = [m for m in measurements if m["correlation"] >= (0.65 if is_eng else 0.55)]

            # Case A: Constant Offset (>= 4 valid points within 80ms)
            if len(valid_pts) >= 4:
                lags = [m["lag"] for m in valid_pts]
                med_lag = float(statistics.median(lags))
                max_dev = max(abs(l - med_lag) for l in lags)
                span = max(m["position"] for m in valid_pts) - min(m["position"] for m in valid_pts)

                if max_dev <= 0.080 and span >= 0.40 * common:
                    final_offset = round(-med_lag + video_start_time, 3)
                    return {
                        "status": "ok",
                        "offset": final_offset,
                        "rate": float(best_k if abs(best_k - 1.0) > 0.001 else 1.0),
                        "confidence": round(float(np.mean([m["correlation"] for m in valid_pts])), 3),
                        "sync_mode": "constant",
                        "deviation": round(max_dev, 4),
                        "provider": provider,
                        "server": server,
                        "audio_source": audio_source,
                        "audio_fingerprint": audio_fp,
                        "video_duration": round(video_duration, 2),
                        "audio_duration": round(audio_duration, 2),
                        "video_start_time": round(video_start_time, 3),
                        "credits_discrepancy": credits_discrepancy,
                        "sync_algorithm": "autosync-v1",
                        "measurements": measurements,
                    }

            # Case B: Linear Drift (Theil-Sen regression max error <= 80ms)
            if len(valid_pts) >= 4:
                pos_list = [m["position"] for m in valid_pts]
                lag_list = [m["lag"] for m in valid_pts]
                slope, intercept = theil_sen(pos_list, lag_list)
                residuals = [abs(lag_list[j] - (intercept + slope * pos_list[j])) for j in range(len(valid_pts))]
                max_res = max(residuals)

                rate_val = 1.0 + slope
                matches_speed = any(abs(rate_val - kh) < 0.0005 for kh in SPEED_HYPOTHESES) or abs(slope) <= 0.002

                if max_res <= 0.080 and matches_speed:
                    final_offset = round(-intercept + video_start_time, 3)
                    return {
                        "status": "ok",
                        "offset": final_offset,
                        "rate": round(rate_val, 7),
                        "confidence": round(float(np.mean([m["correlation"] for m in valid_pts])), 3),
                        "sync_mode": "linear",
                        "deviation": round(max_res, 4),
                        "provider": provider,
                        "server": server,
                        "audio_source": audio_source,
                        "audio_fingerprint": audio_fp,
                        "video_duration": round(video_duration, 2),
                        "audio_duration": round(audio_duration, 2),
                        "video_start_time": round(video_start_time, 3),
                        "credits_discrepancy": credits_discrepancy,
                        "sync_algorithm": "autosync-v1",
                        "measurements": measurements,
                    }

            # Case C: Piecewise Cuts (Bisection)
            if len(valid_pts) >= 4:
                sorted_pts = sorted(valid_pts, key=lambda x: x["position"])
                cut_idx = -1
                for j in range(len(sorted_pts) - 1):
                    if abs(sorted_pts[j + 1]["lag"] - sorted_pts[j]["lag"]) > 0.5:
                        cut_idx = j
                        break

                if cut_idx != -1 and cut_idx >= 1 and (len(sorted_pts) - 1 - cut_idx) >= 1:
                    left_pos = sorted_pts[cut_idx]["position"]
                    right_pos = sorted_pts[cut_idx + 1]["position"]
                    o_left = sorted_pts[cut_idx]["lag"]
                    o_right = sorted_pts[cut_idx + 1]["lag"]

                    for _ in range(4):
                        mid_pos = 0.5 * (left_pos + right_pos)
                        if (right_pos - left_pos) <= 2.0:
                            break
                        cand_p = work_path / f"bisect_{mid_pos:.1f}.pcm"
                        ref_p = work_path / f"bisect_a_{mid_pos:.1f}.pcm"
                        try:
                            await self._sample_video_pcm(light_video_url, video_headers, mid_pos, 15.0, cand_p)
                            c_pcm = np.fromfile(cand_p, dtype=np.int16)
                            ref_win_start = max(0.0, mid_pos + o_left - 10.0)
                            ref_win_dur = 35.0
                            await self._sample_audio_pcm(audio_pl, b_url, audio_headers, audio_key, ref_win_start, ref_win_dur, ref_p)
                            r_pcm = np.fromfile(ref_p, dtype=np.int16)
                            if len(r_pcm) > 0 and len(c_pcm) > 0:
                                c_corr = cross_correlate_valid(envelope_log100(r_pcm), envelope_log100(c_pcm))
                                if len(c_corr) > 0 and np.max(c_corr) >= 0.60:
                                    p_idx = int(np.argmax(c_corr))
                                    r_t, _ = parabolic_peak(c_corr, p_idx)
                                    m_lag = (ref_win_start + r_t) - mid_pos
                                    if abs(m_lag - o_left) < 0.2:
                                        left_pos = mid_pos
                                    else:
                                        right_pos = mid_pos
                                else:
                                    right_pos = mid_pos
                            else:
                                right_pos = mid_pos
                        except Exception:
                            break

                    cut_pos = round(0.5 * (left_pos + right_pos), 1)
                    return {
                        "status": "incompatible",
                        "sync_mode": "piecewise",
                        "confidence": 0.85,
                        "provider": provider,
                        "server": server,
                        "audio_source": audio_source,
                        "audio_fingerprint": audio_fp,
                        "video_duration": round(video_duration, 2),
                        "audio_duration": round(audio_duration, 2),
                        "video_start_time": round(video_start_time, 3),
                        "sync_algorithm": "autosync-v1",
                        "segments": [
                            {"start": 0.0, "end": cut_pos, "offset": round(-o_left + video_start_time, 3)},
                            {"start": cut_pos, "end": round(video_duration, 1), "offset": round(-o_right + video_start_time, 3)},
                        ],
                        "measurements": measurements,
                    }

            # Case D: Incompatible (>= 2 points >= 0.75 that disagree)
            high_conf = [m for m in valid_pts if m["correlation"] >= 0.75]
            if len(high_conf) >= 2:
                dev = max(high_conf, key=lambda x: x["lag"])["lag"] - min(high_conf, key=lambda x: x["lag"])["lag"]
                if dev > 0.150:
                    return {
                        "status": "incompatible",
                        "confidence": round(float(np.mean([m["correlation"] for m in high_conf])), 3),
                        "error": f"Discrepanza non lineare tra punti ad alta confidenza ({dev*1000:.0f}ms)",
                        "provider": provider,
                        "server": server,
                        "audio_source": audio_source,
                        "audio_fingerprint": audio_fp,
                        "video_duration": round(video_duration, 2),
                        "audio_duration": round(audio_duration, 2),
                        "video_start_time": round(video_start_time, 3),
                        "sync_algorithm": "autosync-v1",
                        "measurements": measurements,
                    }

            # Case E: Inconclusive
            return {
                "status": "inconclusive",
                "error": "Punti di verifica insufficienti o correlazione debole",
                "video_duration": round(video_duration, 2),
                "audio_duration": round(audio_duration, 2),
                "measurements": measurements,
            }
