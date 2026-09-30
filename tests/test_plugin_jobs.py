import json
import os
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import plugin_jobs
from app import app


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = tmp_path / "test_plugin_jobs.db"
    plugin_jobs.init_db(db_file)
    os.environ["PLUGIN_ADMIN_KEY"] = "secret_admin_test_123"
    yield
    # Cleanup


def test_job_key_deterministic():
    k1 = plugin_jobs.compute_job_key("movie:tt0172495:0:0", "vidfast", "vixsrc", 7200.46)
    k2 = plugin_jobs.compute_job_key("movie:tt0172495:0:0", "vidfast", "vixsrc", 7200.49)
    assert k1 == k2  # both round to 7200.5


def test_create_job_invalid_media_key():
    client = TestClient(app)
    resp = client.post("/plugin/jobs", json={
        "v": 1,
        "media_key": "invalid_media_key_123",
        "items": []
    })
    assert resp.status_code == 400
    assert "Invalid media_key" in resp.json()["detail"]


def test_create_job_private_ip_rejected():
    client = TestClient(app)
    resp = client.post("/plugin/jobs", json={
        "v": 1,
        "media_key": "movie:tt0172495:0:0",
        "items": [{
            "provider": "vidfast",
            "server": "vRapid",
            "video_duration": 5400.0,
            "renditions": [{"resolution": 1080, "url": "https://127.0.0.1/video.m3u8"}]
        }]
    })
    assert resp.status_code == 400
    assert "private" in resp.json()["detail"].lower() or "not a valid" in resp.json()["detail"].lower()


def test_create_job_deduplication():
    client = TestClient(app)
    payload = {
        "v": 1,
        "media_key": "movie:tt0172495:0:0",
        "audio": {"base_url": "https://example.com/audio.m3u8"},
        "items": [{
            "provider": "movy",
            "server": "Miami",
            "video_duration": 6000.0,
            "renditions": [{"resolution": 1080, "url": "https://example.com/stream-1080p.m3u8"}]
        }]
    }
    # First submit -> queued, requests = 1
    r1 = client.post("/plugin/jobs", json=payload)
    assert r1.status_code == 200
    items1 = r1.json()["items"]
    assert len(items1) == 1
    assert items1[0]["state"] == "queued"
    assert items1[0]["requests"] == 1

    # Second submit -> queued, requests = 2
    r2 = client.post("/plugin/jobs", json=payload)
    assert r2.status_code == 200
    items2 = r2.json()["items"]
    assert len(items2) == 1
    assert items2[0]["requests"] == 2


def test_admin_queue_auth():
    client = TestClient(app)
    # Missing header -> 401
    r_unauth = client.get("/plugin/queue")
    assert r_unauth.status_code == 401

    # Wrong header -> 401
    r_wrong = client.get("/plugin/queue", headers={"X-Admin-Key": "wrong_key"})
    assert r_wrong.status_code == 401

    # Correct header -> 200 HTML
    r_ok = client.get("/plugin/queue", headers={"X-Admin-Key": "secret_admin_test_123"})
    assert r_ok.status_code == 200
    assert "AutoSync Jobs Queue" in r_ok.text


def test_waiting_refresh_and_requeue(tmp_path):
    client = TestClient(app)
    payload = {
        "v": 1,
        "media_key": "series:tt11280740:1:1",
        "audio": {"base_url": "https://example.com/audio.m3u8"},
        "items": [{
            "provider": "vidfast",
            "server": "vRapid",
            "video_duration": 3431.8,
            "renditions": [{"resolution": 1080, "url": "https://example.com/index-s1080p.m3u8"}]
        }]
    }
    r = client.post("/plugin/jobs", json=payload)
    job_key = r.json()["items"][0]["job_key"]

    # Artificially age the URL to 35 minutes ago (2100s)
    with plugin_jobs._connect_db() as conn:
        conn.execute("UPDATE plugin_jobs SET urls_updated_at = ? WHERE job_key = ?", (time.time() - 2100, job_key))

    # Worker polls next job
    with plugin_jobs._connect_db() as conn:
        row = conn.execute("SELECT urls_updated_at FROM plugin_jobs WHERE job_key = ?", (job_key,)).fetchone()
        assert (time.time() - row[0]) > 1800
        # Trigger age check logic
        conn.execute("UPDATE plugin_jobs SET status = 'waiting_refresh' WHERE job_key = ?", (job_key,))

    # Plugin sends fresh job -> status resets to queued
    r2 = client.post("/plugin/jobs", json=payload)
    assert r2.json()["items"][0]["state"] == "queued"
    with plugin_jobs._connect_db() as conn:
        st = conn.execute("SELECT status FROM plugin_jobs WHERE job_key = ?", (job_key,)).fetchone()[0]
        assert st == "queued"


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "AutoSync"

