import asyncio
import hashlib
import hmac
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from fingerprints import offset_cache_key, video_source_fingerprint
from offset_engine import OffsetEngine
from security import resolves_publicly, valid_public_url
from vx_extractor import resolve_vx_tracks


MEDIA_KEY_REGEX = re.compile(r"^(movie|series):tt\d{5,10}:\d{1,3}:\d{1,4}$")
ALLOWED_HEADERS = {"referer", "origin", "user-agent", "cookie"}

router = APIRouter(prefix="/plugin", tags=["plugin_jobs"])

# Module-level singletons set during initialization
_db_path: Path | None = None
_engine: OffsetEngine | None = None
_worker_tasks: list[asyncio.Task] = []
_worker_stop_event: asyncio.Event | None = None


def _connect_db() -> sqlite3.Connection:
    if not _db_path:
        raise RuntimeError("plugin_jobs database not initialized")
    conn = sqlite3.connect(_db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: Path):
    global _db_path
    _db_path = db_path
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_jobs (
                job_key           TEXT PRIMARY KEY,
                media_key         TEXT NOT NULL,
                provider          TEXT NOT NULL,
                server            TEXT NOT NULL,
                audio_source      TEXT NOT NULL DEFAULT 'vixsrc',
                video_duration    REAL NOT NULL,
                renditions        TEXT NOT NULL,
                audio_tracks      TEXT NOT NULL,
                status            TEXT NOT NULL,
                requests          INTEGER NOT NULL DEFAULT 1,
                net_attempts      INTEGER NOT NULL DEFAULT 0,
                measure_attempts  INTEGER NOT NULL DEFAULT 0,
                result            TEXT,
                last_error        TEXT,
                urls_updated_at   REAL NOT NULL,
                next_run_at       REAL NOT NULL,
                created_at        REAL NOT NULL,
                updated_at        REAL NOT NULL,
                expires_at        REAL NOT NULL,
                stage             TEXT DEFAULT 'tier1'
            )
        """)
        try:
            conn.execute("ALTER TABLE plugin_jobs ADD COLUMN stage TEXT DEFAULT 'tier1'")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plugin_jobs_poll
            ON plugin_jobs(status, next_run_at, requests DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plugin_rate (
                ip TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plugin_rate_ip
            ON plugin_rate(ip, ts)
        """)


def check_rate_limit(ip: str, limit_per_min: int = 20) -> bool:
    now = time.time()
    cutoff = now - 60.0
    with _connect_db() as conn:
        conn.execute("DELETE FROM plugin_rate WHERE ts < ?", (cutoff,))
        count = conn.execute("SELECT COUNT(*) FROM plugin_rate WHERE ip = ? AND ts >= ?", (ip, cutoff)).fetchone()[0]
        if count >= limit_per_min:
            return False
        conn.execute("INSERT INTO plugin_rate (ip, ts) VALUES (?, ?)", (ip, now))
    return True


def sanitize_headers(headers: dict | None) -> dict:
    if not isinstance(headers, dict):
        return {}
    cleaned = {}
    for k, v in headers.items():
        k_str = str(k).strip()
        if k_str.lower() in ALLOWED_HEADERS and isinstance(v, (str, int, float)):
            cleaned[k_str] = str(v).strip()[:1000]
    return cleaned


async def validate_url_security(url: str):
    if not valid_public_url(url):
        raise HTTPException(status_code=400, detail=f"URL is not a valid public HTTPS URL: {url}")
    if not await resolves_publicly(url):
        raise HTTPException(status_code=400, detail=f"URL resolves to a private or non-routable IP: {url}")


def compute_job_key(media_key: str, provider: str, audio_source: str, video_duration: float) -> str:
    raw = f"{media_key}|{provider.strip().lower()}|{audio_source.strip().lower()}|{round(video_duration, 1)}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@router.post("/jobs")
async def create_jobs(request: Request):
    prisynx_secret = os.getenv("PRISYNX_SECRET", "").strip()
    if prisynx_secret:
        key_header = request.headers.get("X-PriSynx-Key", "").strip()
        if not key_header or not hmac.compare_digest(key_header, prisynx_secret):
            raise HTTPException(status_code=401, detail="Unauthorized")

    ip = request.client.host if request.client else "127.0.0.1"
    rate_limit = int(os.getenv("PLUGIN_RATE_PER_MIN", "20"))
    if not check_rate_limit(ip, rate_limit):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    body_bytes = await request.body()
    if len(body_bytes) > 512 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large (max 512KB)")

    try:
        body = json.loads(body_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    media_key = str(body.get("media_key") or "").strip()
    if not MEDIA_KEY_REGEX.match(media_key):
        raise HTTPException(status_code=400, detail="Invalid media_key format (expected movie:ttX:0:0 or series:ttX:S:E)")

    # Extract audio tracks from root or items
    root_audio = body.get("audio") or {}
    audio_tracks = body.get("audio_tracks") or []
    if root_audio and not audio_tracks:
        audio_tracks = [{
            "lang": "ita",
            "playlist": root_audio.get("playlist", ""),
            "key": root_audio.get("key", ""),
            "base_url": root_audio.get("base_url", ""),
            "headers": sanitize_headers(root_audio.get("headers")),
            "source": str(root_audio.get("source") or "vixsrc").strip().lower(),
        }]

    # Validate audio base_url
    for t in audio_tracks:
        b_url = t.get("base_url")
        if b_url:
            await validate_url_security(b_url)

    items = body.get("items") or []
    if len(items) > 6:
        raise HTTPException(status_code=400, detail="Too many items (max 6)")

    max_queue = int(os.getenv("PLUGIN_QUEUE_MAX", "500"))
    ttl_s = float(os.getenv("PLUGIN_JOB_TTL_S", "21600"))
    now = time.time()

    results = []

    with _connect_db() as conn:
        q_count = conn.execute("SELECT COUNT(*) FROM plugin_jobs WHERE status = 'queued'").fetchone()[0]
        if q_count >= max_queue:
            return JSONResponse(status_code=200, content={
                "items": [{"state": "rejected", "reason": "queue full"} for _ in items]
            })

        for it in items:
            provider = str(it.get("provider") or "").strip().lower()
            server = str(it.get("server") or "").strip()
            video_dur = float(it.get("video_duration") or 0.0)
            headers = sanitize_headers(it.get("headers"))
            raw_renditions = it.get("renditions") or []

            if video_dur < 30.0 or not raw_renditions:
                results.append({"video_duration": video_dur, "state": "rejected", "reason": "invalid duration or renditions"})
                continue

            # Validate renditions: >= 1080 and valid public URL
            valid_renditions = []
            for r in raw_renditions:
                res = int(r.get("resolution") or 0)
                u = str(r.get("url") or "").strip()
                if res >= 1080 and u:
                    await validate_url_security(u)
                    valid_renditions.append({
                        "resolution": res,
                        "url": u,
                        "headers": sanitize_headers(r.get("headers") or headers),
                    })

            if not valid_renditions:
                results.append({"video_duration": video_dur, "state": "rejected", "reason": "no renditions >= 1080"})
                continue

            audio_source = "vixsrc"
            if audio_tracks and audio_tracks[0].get("source"):
                audio_source = audio_tracks[0]["source"]

            job_key = compute_job_key(media_key, provider, audio_source, video_dur)

            # Check existing job
            row = conn.execute("SELECT status, requests, result, updated_at FROM plugin_jobs WHERE job_key = ?", (job_key,)).fetchone()
            if row:
                st, reqs, res_json, upd_at = row
                if st == "done" and res_json:
                    results.append({
                        "job_key": job_key,
                        "video_duration": video_dur,
                        "state": "done",
                        "offset": json.loads(res_json),
                    })
                    continue
                if st == "incompatible" and res_json:
                    results.append({
                        "job_key": job_key,
                        "video_duration": video_dur,
                        "state": "failed",
                        "reason": "incompatible version",
                    })
                    continue
                if st == "failed" and (now - upd_at < 86400):
                    results.append({
                        "job_key": job_key,
                        "video_duration": video_dur,
                        "state": "failed",
                        "reason": "failed 3 times, retry in 24h",
                    })
                    continue

                # Refresh URLs and bump requests
                conn.execute("""
                    UPDATE plugin_jobs
                    SET renditions = ?, audio_tracks = ?, requests = requests + 1,
                        urls_updated_at = ?, expires_at = ?,
                        stage = CASE WHEN status = 'waiting_refresh' THEN 'tier1' ELSE stage END,
                        status = CASE WHEN status = 'waiting_refresh' THEN 'queued' ELSE status END
                    WHERE job_key = ?
                """, (json.dumps(valid_renditions), json.dumps(audio_tracks), now, now + ttl_s, job_key))

                results.append({
                    "job_key": job_key,
                    "video_duration": video_dur,
                    "state": "queued" if st == "waiting_refresh" else st,
                    "requests": reqs + 1,
                })
            else:
                conn.execute("""
                    INSERT INTO plugin_jobs (
                        job_key, media_key, provider, server, audio_source, video_duration,
                        renditions, audio_tracks, status, requests, net_attempts, measure_attempts,
                        urls_updated_at, next_run_at, created_at, updated_at, expires_at, stage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1, 0, 0, ?, ?, ?, ?, ?, 'tier1')
                """, (
                    job_key, media_key, provider, server, audio_source, video_dur,
                    json.dumps(valid_renditions), json.dumps(audio_tracks),
                    now, now, now, now, now + ttl_s,
                ))
                results.append({
                    "job_key": job_key,
                    "video_duration": video_dur,
                    "state": "queued",
                    "requests": 1,
                })

    return {"items": results}


@router.get("/status")
@router.get("/jobs/status")
async def get_jobs_status(media_key: str):
    if not media_key:
        raise HTTPException(status_code=400, detail="Missing media_key")
    with _connect_db() as conn:
        rows = conn.execute("""
            SELECT job_key, provider, video_duration, status, result, last_error, updated_at, stage, requests
            FROM plugin_jobs WHERE media_key = ?
        """, (media_key,)).fetchall()

        items = []
        for r in rows:
            j_key, prov, v_dur, st, res_json, err, upd_at, stage, reqs = r
            item = {
                "job_key": j_key,
                "provider": prov,
                "video_duration": v_dur,
                "status": st,
                "stage": stage or "tier1",
                "updated_at": upd_at,
            }
            if st == "queued":
                ahead = conn.execute("""
                    SELECT COUNT(*) FROM plugin_jobs
                    WHERE status = 'queued'
                      AND (requests > ? OR (requests = ? AND updated_at > ?))
                """, (reqs, reqs, upd_at)).fetchone()[0]
                item["queue_ahead"] = ahead

            if res_json and st in ("done", "incompatible"):
                item["result"] = json.loads(res_json)
            if err:
                item["error"] = err
            items.append(item)
    return {"media_key": media_key, "items": items}


@router.get("/queue", response_class=HTMLResponse)
async def get_queue_dashboard(x_admin_key: str | None = Header(None)):
    admin_key = os.getenv("PLUGIN_ADMIN_KEY", "").strip()
    if not admin_key or not x_admin_key or not hmac.compare_digest(x_admin_key.strip(), admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    with _connect_db() as conn:
        rows = conn.execute("""
            SELECT job_key, media_key, provider, server, video_duration, status, stage,
                   requests, net_attempts, measure_attempts, last_error, created_at, updated_at
            FROM plugin_jobs
            ORDER BY updated_at DESC LIMIT 100
        """).fetchall()

    html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AutoSync Queue</title>
    <style>body{font-family:monospace;background:#111;color:#eee;padding:20px}table{width:100%;border-collapse:collapse}
    th,td{border:1px solid #333;padding:8px;text-align:left}th{background:#222}.done{color:#4ade80}.failed{color:#f87171}
    .queued{color:#facc15}.running{color:#60a5fa}</style></head><body>
    <h2>AutoSync Jobs Queue</h2>
    <table><tr><th>Job Key</th><th>Media Key</th><th>Provider</th><th>Duration</th><th>Status</th><th>Stage</th><th>Reqs</th><th>Attempts</th><th>Last Error</th><th>Updated</th></tr>"""
    for r in rows:
        j_key, m_key, prov, srv, dur, st, stg, reqs, n_att, m_att, err, cr_at, upd_at = r
        upd_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(upd_at))
        err_str = (err or "")[:50]
        html += f"<tr><td>{j_key[:8]}..</td><td>{m_key}</td><td>{prov}/{srv}</td><td>{dur:.1f}s</td><td class='{st}'>{st}</td><td>{stg or 'tier1'}</td><td>{reqs}</td><td>N:{n_att}/M:{m_att}</td><td>{err_str}</td><td>{upd_str}</td></tr>"
    html += "</table></body></html>"
    return HTMLResponse(content=html)


@router.post("/queue/{job_key}/retry")
async def retry_queue_job(job_key: str, x_admin_key: str | None = Header(None)):
    admin_key = os.getenv("PLUGIN_ADMIN_KEY", "").strip()
    if not admin_key or not x_admin_key or not hmac.compare_digest(x_admin_key.strip(), admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized")

    now = time.time()
    with _connect_db() as conn:
        row = conn.execute("SELECT status FROM plugin_jobs WHERE job_key = ?", (job_key,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        conn.execute("""
            UPDATE plugin_jobs
            SET status = 'queued', stage = 'tier1', net_attempts = 0, measure_attempts = 0, next_run_at = ?, updated_at = ?
            WHERE job_key = ?
        """, (now, now, job_key))
    return {"job_key": job_key, "status": "queued"}


# ---------------------------------------------------------------------------
# Background Worker
# ---------------------------------------------------------------------------

async def _report_to_toastflix(payload: dict, result: dict, host: str, access_token: str):
    base_url = (host or os.getenv("TOASTFLIX_URL", os.getenv("OFFSET_API_URL", "https://toastflix.stremio-italia.eu"))).strip().rstrip("/")
    if not base_url:
        return
    api_url = f"{base_url}/dual/offset/report" if not base_url.endswith("/dual/offset") else f"{base_url}/report"

    body = {
        "access": payload.get("access") or access_token,
        "media_key": payload["media_key"],
        "resolution": payload["resolution"],
        "video_fingerprint": payload["video_fingerprint"],
        "audio_fingerprint": payload["audio_fingerprint"],
        "cache_key": payload.get("cache_key") or offset_cache_key(
            payload["media_key"], payload["resolution"], payload["video_fingerprint"], payload["audio_fingerprint"]
        ),
        "offset": result,
    }

    api_token = os.getenv("OFFSET_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(api_url, json=body, headers=headers)
            if resp.status_code >= 400:
                print(f"[report_to_toastflix] ToastFlix returned {resp.status_code}: {resp.text[:200]}")
            else:
                print(f"[report_to_toastflix] Successfully reported to ToastFlix for {payload['media_key']} ({payload['resolution']}p)")
    except Exception as e:
        print(f"[report_to_toastflix] Network error reporting to ToastFlix: {e}")


def get_dual_access_token() -> str:
    raw = os.getenv("OFFSET_API_ACCESS", "").strip()
    if not raw:
        return ""
    if len(raw) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.lower()
    return hmac.new(raw.encode(), b"dual-remuxed", hashlib.sha256).hexdigest()[:32]


async def _worker_loop(worker_id: int = 1):
    print(f"[plugin_jobs worker {worker_id}] Worker {worker_id} started successfully.")
    offset_access = get_dual_access_token()

    while _worker_stop_event and not _worker_stop_event.is_set():
        try:
            now = time.time()
            job = None
            with _connect_db() as conn:
                row = conn.execute("""
                    SELECT job_key, media_key, provider, server, audio_source, video_duration,
                           renditions, audio_tracks, urls_updated_at, net_attempts, measure_attempts
                    FROM plugin_jobs
                    WHERE status = 'queued' AND next_run_at <= ?
                    ORDER BY requests DESC, updated_at DESC
                    LIMIT 1
                """, (now,)).fetchone()

                if row:
                    j_key, m_key, prov, srv, a_src, v_dur, rends_raw, auds_raw, u_upd, n_att, m_att = row
                    # Check token age (>30m -> waiting_refresh)
                    if (now - u_upd) > 1800:
                        conn.execute("UPDATE plugin_jobs SET status = 'waiting_refresh', updated_at = ? WHERE job_key = ?", (now, j_key))
                        continue

                    conn.execute("UPDATE plugin_jobs SET status = 'running', stage = 'tier1', updated_at = ? WHERE job_key = ?", (now, j_key))
                    job = {
                        "job_key": j_key, "media_key": m_key, "provider": prov, "server": srv,
                        "audio_source": a_src, "video_duration": v_dur,
                        "renditions": json.loads(rends_raw), "audio_tracks": json.loads(auds_raw),
                        "net_attempts": n_att, "measure_attempts": m_att,
                    }

            if not job:
                await asyncio.sleep(2.0)
                continue

            print(f"[plugin_jobs worker] Processing {job['media_key']} ({job['provider']} {job['video_duration']}s)...")

            # Pick highest resolution rendition as primary measurement candidate
            target_rend = sorted(job["renditions"], key=lambda r: int(r.get("resolution") or 0), reverse=True)[0]

            audio_tracks = job["audio_tracks"]
            if str(job.get("audio_source") or "").lower() in ("vx", "vixsrc") or not any(t.get("playlist") for t in audio_tracks):
                try:
                    resolved = await resolve_vx_tracks(job["media_key"])
                    if resolved:
                        audio_tracks = resolved
                        print(f"[plugin_jobs worker] VX extracted {len(resolved)} fresh track(s) for {job['media_key']}")
                except Exception as vx_err:
                    print(f"[plugin_jobs worker] VX extraction warning for {job['media_key']}: {vx_err}")

            measure_payload = {
                "media_key": job["media_key"],
                "resolution": target_rend.get("resolution", 1080),
                "provider": job["provider"],
                "server": job["server"],
                "video_url": target_rend["url"],
                "video_headers": target_rend.get("headers") or {},
                "audio_tracks": audio_tracks,
                "audio_source": job["audio_source"],
            }

            async def _on_stage(stage_name: str):
                try:
                    with _connect_db() as db_conn:
                        db_conn.execute(
                            "UPDATE plugin_jobs SET stage = ?, updated_at = ? WHERE job_key = ?",
                            (stage_name, time.time(), job["job_key"]),
                        )
                    print(f"[plugin_jobs worker] Job {job['media_key']} entered stage '{stage_name}'")
                except Exception as stage_err:
                    print(f"[plugin_jobs worker] Error setting stage '{stage_name}': {stage_err}")

            try:
                if not _engine:
                    raise RuntimeError("OffsetEngine not initialized")
                res = await _engine.measure(measure_payload, stage_callback=_on_stage)
                now_done = time.time()
                status = res.get("status")

                if status == "ok":
                    print(f"[plugin_jobs worker] Sync OK for {job['media_key']}: offset={res.get('offset')}s, rate={res.get('rate')}")
                    # Multi-rendition reporting for all renditions >= 1080
                    for rend in job["renditions"]:
                        try:
                            r_url = rend["url"]
                            r_res = int(rend.get("resolution") or 1080)
                            r_headers = rend.get("headers") or {}
                            r_fp = video_source_fingerprint({"url": r_url, "provider": job["provider"], "server": job["server"]})
                            r_vst = await _engine._media_start_time(r_url, r_headers)

                            rend_result = dict(res)
                            # Adjust offset for specific rendition video_start_time
                            raw_offset = res.get("offset", 0.0) - res.get("video_start_time", 0.0)
                            rend_result["offset"] = round(raw_offset + r_vst, 3)
                            rend_result["video_start_time"] = r_vst

                            report_payload = {
                                "access": offset_access,
                                "media_key": job["media_key"],
                                "resolution": r_res,
                                "video_fingerprint": r_fp,
                                "audio_fingerprint": res.get("audio_fingerprint", ""),
                                "cache_key": offset_cache_key(job["media_key"], r_res, r_fp, res.get("audio_fingerprint", "")),
                            }
                            await _report_to_toastflix(report_payload, rend_result, job.get("vpsHost", ""), offset_access)
                        except Exception as r_err:
                            print(f"[plugin_jobs worker] Report error for rendition {rend.get('resolution')}: {r_err}")

                    with _connect_db() as conn:
                        conn.execute("""
                            UPDATE plugin_jobs
                            SET status = 'done', result = ?, updated_at = ?
                            WHERE job_key = ?
                        """, (json.dumps(res), now_done, job["job_key"]))

                elif status == "incompatible" and res.get("measurements"):
                    print(f"[plugin_jobs worker] Sync INCOMPATIBLE for {job['media_key']}")
                    # Report incompatible if verified
                    r_fp = video_source_fingerprint({"url": target_rend["url"], "provider": job["provider"], "server": job["server"]})
                    report_payload = {
                        "access": offset_access,
                        "media_key": job["media_key"],
                        "resolution": int(target_rend.get("resolution") or 1080),
                        "video_fingerprint": r_fp,
                        "audio_fingerprint": res.get("audio_fingerprint", ""),
                        "cache_key": offset_cache_key(job["media_key"], int(target_rend.get("resolution") or 1080), r_fp, res.get("audio_fingerprint", "")),
                    }
                    await _report_to_toastflix(report_payload, res, job.get("vpsHost", ""), offset_access)

                    with _connect_db() as conn:
                        conn.execute("""
                            UPDATE plugin_jobs
                            SET status = 'incompatible', result = ?, updated_at = ?
                            WHERE job_key = ?
                        """, (json.dumps(res), now_done, job["job_key"]))

                else:
                    # Inconclusive -> DO NOT report to ToastFlix, retry locally
                    m_att = job["measure_attempts"] + 1
                    print(f"[plugin_jobs worker] Sync inconclusive for {job['media_key']} (attempt {m_att}/3)")
                    with _connect_db() as conn:
                        if m_att < 3:
                            delay = 60.0 * (2 ** m_att)
                            conn.execute("""
                                UPDATE plugin_jobs
                                SET status = 'queued', stage = 'tier1', measure_attempts = ?, next_run_at = ?, last_error = ?, updated_at = ?
                                WHERE job_key = ?
                            """, (m_att, now_done + delay, res.get("error", "inconclusive"), now_done, job["job_key"]))
                        else:
                            conn.execute("""
                                UPDATE plugin_jobs
                                SET status = 'inconclusive', measure_attempts = ?, last_error = ?, updated_at = ?
                                WHERE job_key = ?
                            """, (m_att, res.get("error", "inconclusive"), now_done, job["job_key"]))

            except (httpx.HTTPError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
                now_err = time.time()
                n_att = job["net_attempts"] + 1
                err_msg = str(exc)[:300]
                print(f"[plugin_jobs worker] Network/Transient error on {job['media_key']} (net_attempt {n_att}/3): {err_msg}")
                with _connect_db() as conn:
                    if n_att < 3:
                        # 2 min, then 10 min, then 60 min
                        delay = [120.0, 600.0, 3600.0][n_att - 1]
                        conn.execute("""
                            UPDATE plugin_jobs
                            SET status = 'queued', stage = 'tier1', net_attempts = ?, next_run_at = ?, last_error = ?, updated_at = ?
                            WHERE job_key = ?
                        """, (n_att, now_err + delay, err_msg, now_err, job["job_key"]))
                    else:
                        conn.execute("""
                            UPDATE plugin_jobs
                            SET status = 'failed', net_attempts = ?, last_error = ?, updated_at = ?
                            WHERE job_key = ?
                        """, (n_att, err_msg, now_err, job["job_key"]))

        except Exception as loop_err:
            print(f"[plugin_jobs worker] Unexpected loop error: {loop_err}")
            await asyncio.sleep(2.0)


def start_worker(engine: OffsetEngine, db_path: Path):
    global _engine, _worker_tasks, _worker_stop_event
    init_db(db_path)
    _engine = engine
    _worker_stop_event = asyncio.Event()
    worker_count = int(os.getenv("AUTOSYNC_WORKERS", "2"))
    _worker_tasks = [asyncio.create_task(_worker_loop(i + 1)) for i in range(worker_count)]


async def stop_worker():
    global _worker_tasks, _worker_stop_event
    if _worker_stop_event:
        _worker_stop_event.set()
    if _worker_tasks:
        for t in _worker_tasks:
            try:
                await asyncio.wait_for(t, timeout=5.0)
            except Exception:
                t.cancel()
        _worker_tasks = []
