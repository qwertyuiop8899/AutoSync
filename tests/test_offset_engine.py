import numpy as np
import pytest

from offset_engine import (
    calculate_psr,
    cross_correlate_valid,
    envelope_log100,
    envelope_lowpass,
    gcc_phat,
    parabolic_peak,
    resample_envelope,
    theil_sen,
)


def test_envelope_log100():
    sr = 8000
    samples = np.sin(np.linspace(0, 100, sr * 2, dtype=np.float32))
    env = envelope_log100(samples, sr=sr)
    assert len(env) > 0
    # Mean should be ~0 and std should be ~1 due to z-score
    assert abs(float(np.mean(env))) < 1e-4
    assert abs(float(np.std(env)) - 1.0) < 1e-4


def test_cross_correlate_valid():
    np.random.seed(42)
    ref = np.random.randn(500).astype(np.float32)
    cand = ref[100:200].copy()
    corr = cross_correlate_valid(ref, cand)
    assert len(corr) == 500 - 100 + 1
    # Best peak must be exactly at index 100
    assert np.argmax(corr) == 100
    assert corr[100] > 0.99


def test_psr():
    corr = np.zeros(200, dtype=np.float32)
    corr[100] = 0.90
    corr[20] = 0.30
    psr = calculate_psr(corr, 100, exclude_radius=50)
    assert abs(psr - (0.90 / 0.30)) < 1e-5


def test_parabolic_peak():
    corr = np.array([0.2, 0.6, 0.95, 0.8, 0.3], dtype=np.float32)
    refined_t, val = parabolic_peak(corr, 2, step=0.01)
    # Expected peak slightly to the right of index 2 because 0.8 > 0.6
    assert refined_t > 0.020
    assert val >= 0.95


def test_resample_speed():
    env = np.ones(1000, dtype=np.float32)
    k = 25.0 / 24.0
    res = resample_envelope(env, k)
    assert len(res) == int(round(1000 * k))


def test_gcc_phat():
    sr = 8000
    np.random.seed(123)
    t = np.linspace(0, 2.0, int(2.0 * sr), dtype=np.float32)
    s1 = np.sin(2 * np.pi * 350 * t) + np.random.randn(len(t)).astype(np.float32) * 0.1
    delay_samples = int(0.008 * sr)  # 8 ms delay
    s2 = np.roll(s1, delay_samples)
    tau = gcc_phat(s2, s1, sr=sr, max_tau_ms=30.0)
    assert abs(tau - 0.008) < 0.0005  # sub-millisecond precision


def test_theil_sen():
    positions = [200.0, 500.0, 800.0, 1200.0, 1600.0]
    # Linear drift with rate = 1.001 (slope = 0.001) and intercept = 2.5s
    offsets = [2.5 + 0.001 * p for p in positions]
    slope, intercept = theil_sen(positions, offsets)
    assert abs(slope - 0.001) < 1e-6
    assert abs(intercept - 2.5) < 1e-5


def test_synthetic_offset_detection():
    # End-to-end synthetic audio test: verify known offset is recovered with error <= 15 ms
    sr = 8000
    duration = 80
    np.random.seed(777)
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    # Realistic speech-like bursts: envelope modulation
    bursts = (np.sin(2 * np.pi * 0.8 * t) > 0.2).astype(np.float32)
    ref_audio = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)) * bursts + np.random.randn(len(t)) * 0.1

    known_offset = 3.250  # 3.25 seconds offset
    v_pos = 25.0
    cand_audio = ref_audio[int((v_pos + known_offset) * sr) : int((v_pos + known_offset + 15.0) * sr)]

    # Reference window around v_pos (+/- 10s)
    ref_win_start = v_pos - 10.0
    ref_win = ref_audio[int(ref_win_start * sr) : int((v_pos + 15.0 + 10.0) * sr)]

    ref_env = envelope_log100(ref_win, sr=sr)
    cand_env = envelope_log100(cand_audio, sr=sr)

    corr = cross_correlate_valid(ref_env, cand_env)
    pk_idx = int(np.argmax(corr))
    refined_t, peak_val = parabolic_peak(corr, pk_idx, step=0.01)

    detected_lag = (ref_win_start + refined_t) - v_pos
    error_ms = abs(detected_lag - known_offset) * 1000.0
    assert peak_val > 0.90
    assert error_ms <= 15.0  # Error less than 15 ms!


def test_ffmpeg_audio_download_and_decode(tmp_path):
    import asyncio
    import subprocess
    from unittest.mock import AsyncMock
    from offset_engine import OffsetEngine

    async def _run():
        # Generate a small 4s TS segment using ffmpeg
        seg_file = tmp_path / "seg-0.ts"
        cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:a", "aac", "-f", "mpegts", "-y", str(seg_file)
        ]
        subprocess.run(cmd, check=True)
        assert seg_file.exists() and seg_file.stat().st_size > 0

        seg_bytes = seg_file.read_bytes()

        engine = OffsetEngine()
        # Mock _get to return our local segment bytes
        mock_resp = AsyncMock()
        mock_resp.content = seg_bytes
        engine._get = AsyncMock(return_value=mock_resp)

        playlist_text = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:4
#EXTINF:4.000000,
https://example.com/seg-0.ts
#EXT-X-ENDLIST
"""
        out_pcm = tmp_path / "out.pcm"
        dur = await engine._download_entire_audio_pcm(playlist_text, "https://example.com", {}, "", out_pcm)
        assert dur == 4.0
        assert out_pcm.exists()
        # At 8kHz mono 16-bit PCM (2 bytes/sample), 4 seconds = 4 * 8000 * 2 = 64,000 bytes
        pcm_bytes = out_pcm.stat().st_size
        assert 60000 <= pcm_bytes <= 68000
        pcm_samples = np.fromfile(out_pcm, dtype=np.int16)
        assert len(pcm_samples) > 0
        assert not engine._is_silent(pcm_samples)

    asyncio.run(_run())


def test_sample_audio_pcm(tmp_path):
    import asyncio
    import subprocess
    from unittest.mock import AsyncMock
    from offset_engine import OffsetEngine

    async def _run():
        seg_file = tmp_path / "seg-sample.ts"
        cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
            "-c:a", "aac", "-f", "mpegts", "-y", str(seg_file)
        ]
        subprocess.run(cmd, check=True)
        assert seg_file.exists() and seg_file.stat().st_size > 0

        engine = OffsetEngine("socks5://127.0.0.1:9050,socks5://127.0.0.1:9051")
        assert len(engine.proxies) == 2
        assert engine._next_proxy() == "socks5://127.0.0.1:9050"
        assert engine._next_proxy() == "socks5://127.0.0.1:9051"

        mock_resp = AsyncMock()
        mock_resp.content = seg_file.read_bytes()
        engine._get = AsyncMock(return_value=mock_resp)

        playlist_text = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:4
#EXTINF:4.000000,
https://example.com/seg-sample.ts
#EXT-X-ENDLIST
"""
        out_pcm = tmp_path / "sample_out.pcm"
        await engine._sample_audio_pcm(playlist_text, "https://example.com", {}, "", position=0.0, duration=2.0, out_pcm=out_pcm)
        assert out_pcm.exists()
        pcm_bytes = out_pcm.stat().st_size
        # 2 seconds at 8kHz mono 16-bit = 2 * 8000 * 2 = 32,000 bytes
        assert 28000 <= pcm_bytes <= 36000
        pcm_samples = np.fromfile(out_pcm, dtype=np.int16)
        assert len(pcm_samples) > 0
        assert not engine._is_silent(pcm_samples)

    asyncio.run(_run())


