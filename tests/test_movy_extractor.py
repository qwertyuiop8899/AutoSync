import asyncio
import pytest
import httpx

from movy_extractor import (
    MOVY_UA,
    MOVY_SITE,
    decrypt,
    extract_movy_stream,
    get_warp_proxy,
    movy_headers,
    parse_media_key,
    quality_of,
    _init_key,
    _next_word,
    _ushr,
)
from offset_engine import OffsetEngine


def test_quality_of():
    assert quality_of("2160p") == 2160
    assert quality_of("4k") == 2160
    assert quality_of("1080p") == 1080
    assert quality_of("720p") == 720
    assert quality_of("480p") == 480
    assert quality_of("unknown") is None


def test_parse_media_key_movie():
    async def _run():
        is_movie, tmdb_id, imdb_id, s, ep, title, year = await parse_media_key("movie:tt0816692:0:0")
        assert is_movie is True
        assert tmdb_id == 157336
        assert imdb_id == "tt0816692"
        assert s == 0
        assert ep == 0
        assert "interstellar" in title.lower()

    asyncio.run(_run())


def test_parse_media_key_series():
    async def _run():
        is_movie, tmdb_id, imdb_id, s, ep, title, year = await parse_media_key("series:tt13111078:2:2")
        assert is_movie is False
        assert tmdb_id == 113962
        assert imdb_id == "tt13111078"
        assert s == 2
        assert ep == 2

    asyncio.run(_run())


def test_decrypt_synthetic_payload():
    import base64
    seed = "test-seed-12345"
    media_id = 99999
    plain = b'mvm1{"sources":[{"quality":"1080p","url":"https://example.com/test.m3u8"}]}'
    
    # Encrypt using inverse operation
    state = _init_key(seed, media_id)
    cipher = bytearray(len(plain))
    i = 0
    round_num = 0
    while i < len(plain):
        word = _next_word(state, round_num)
        b = 0
        while b < 4 and i < len(plain):
            w_byte = (_ushr(word, 8 * b)) & 255
            cipher[i] = plain[i] ^ w_byte
            i += 1
            b += 1
        round_num += 1

    payload_b64 = base64.b64encode(cipher).decode("ascii")
    decrypted = decrypt(payload_b64, seed, media_id)
    assert decrypted is not None
    assert "sources" in decrypted
    assert decrypted["sources"][0]["quality"] == "1080p"


def test_extract_movy_stream_movie_and_download_m3u8():
    async def _run():
        proxy = get_warp_proxy()
        assert proxy.startswith("socks5://")

        # 1. Extract real stream for Interstellar
        media_key = "movie:tt0816692:0:0"
        renditions = await extract_movy_stream(media_key)
        assert len(renditions) > 0

        target = renditions[0]
        assert target["resolution"] >= 1080
        assert target["url"].startswith("https://")
        assert ".m3u8" in target["url"]

        # 2. Download m3u8 playlist via WARP with matching headers
        async with httpx.AsyncClient(proxy=proxy, timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(target["url"], headers=target["headers"])
            assert resp.status_code == 200
            assert "#EXTM3U" in resp.text
            assert ("#EXTINF" in resp.text or "#EXT-X-STREAM-INF" in resp.text)

    asyncio.run(_run())


def test_extract_movy_stream_series_and_download_m3u8():
    async def _run():
        proxy = get_warp_proxy()
        # 1. Extract real stream for series:tt13111078:2:2 (or series:tt5622316:5:1)
        media_key = "series:tt13111078:2:2"
        try:
            renditions = await extract_movy_stream(media_key)
        except Exception:
            # Fallback to Young Sheldon if Phoenix server is busy
            renditions = await extract_movy_stream("series:tt5622316:5:1", preferred_server="miami")

        assert len(renditions) > 0
        target = renditions[0]
        assert target["resolution"] >= 1080

        # 2. Download m3u8 playlist via WARP
        async with httpx.AsyncClient(proxy=proxy, timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(target["url"], headers=target["headers"])
            assert resp.status_code == 200
            assert "#EXTM3U" in resp.text

    asyncio.run(_run())


def test_offset_engine_movy_download():
    async def _run():
        engine = OffsetEngine()
        assert engine._is_movy_request("https://moon.goldenkite.top/vd/test/index.m3u8")
        assert engine._is_movy_request("https://random-cdn.com/test", headers={"Referer": "https://www.movy.sx/"})

        # Test real download through OffsetEngine._get
        renditions = await extract_movy_stream("movie:tt0816692:0:0")
        target_url = renditions[0]["url"]
        target_headers = renditions[0]["headers"]

        resp = await engine._get(target_url, target_headers)
        assert resp.status_code == 200
        assert "#EXTM3U" in resp.text

    asyncio.run(_run())
