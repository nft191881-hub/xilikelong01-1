import ipaddress
import os
import socket
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from database import get_recent_results, init_db, save_result
from utils.infer import infer_from_text
from utils.ocr import extract_text_from_image

MAX_TEXT_LEN = 200_000
MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 5
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Create the table on startup (no-op if DATABASE_URL is not set)
    await run_in_threadpool(init_db)
    yield


app = FastAPI(title="XILIkelong", version="0.3.0", lifespan=lifespan)


# ---------- helpers ----------

def _store(source_type: str, result: Dict[str, Any], raw_text: str,
           source_url: Optional[str] = None, image_name: Optional[str] = None):
    return save_result(
        source_type=source_type,
        source_url=source_url,
        image_name=image_name,
        frontend=result.get("frontend"),
        backend=result.get("backend"),
        database_name=result.get("database"),
        confidence=result.get("confidence", 0),
        evidence=result.get("evidence", []),
        skeleton=result.get("skeleton", []),
        raw_text=raw_text[:5000],
    )


def collect_candidate_signals(url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta = " ".join(str(m.get("content", "")) for m in soup.find_all("meta"))
    links = [a.get("href", "") for a in soup.find_all("a") if a.get("href")]
    scripts = [s.get("src", "") for s in soup.find_all("script") if s.get("src")]
    css = [l.get("href", "") for l in soup.find_all("link") if l.get("href")]
    body_text = soup.get_text(" ", strip=True)
    combined = " ".join([url, title, meta, body_text,
                         " ".join(links), " ".join(scripts), " ".join(css)])
    return {
        "title": title,
        "meta": meta[:300],
        "links": links[:30],
        "scripts": scripts[:30],
        "css": css[:30],
        "combined_text": combined,
    }


def _assert_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to private/internal IPs (SSRF guard)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http/https URLs are allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ValueError("Cannot resolve host")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            raise ValueError("URL points to a non-public address")


async def fetch_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; XILIkelong/1.0; +https://github.com/nft191881-hub/XILIKelong01)"
    }
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            # validate every hop, including redirects
            await run_in_threadpool(_assert_public_url, current)
            async with client.stream("GET", current, headers=headers) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise ValueError("Redirect without Location header")
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_HTML_BYTES:
                        break  # truncate oversized pages
                    chunks.append(chunk)
                return b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        raise ValueError("Too many redirects")


class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)


class UrlPayload(BaseModel):
    url: str = Field(..., min_length=4, max_length=2048)


# ---------- routes ----------

@app.get("/")
async def index():
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return JSONResponse({"message": "Frontend file not found. Create static/index.html first.",
                         "status": "pending"})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/history")
async def history(limit: int = 10):
    return {"items": await run_in_threadpool(get_recent_results, limit)}


async def _infer_plain(source_type: str, payload: TextPayload):
    result = await run_in_threadpool(infer_from_text, payload.text)
    await run_in_threadpool(_store, source_type, result, payload.text)
    return result


@app.post("/api/infer/text")
async def infer_text(payload: TextPayload):
    return await _infer_plain("text", payload)


@app.post("/api/infer/code")
async def infer_code(payload: TextPayload):
    return await _infer_plain("code", payload)


@app.post("/api/infer/url")
async def infer_url(payload: UrlPayload):
    try:
        html = await fetch_url(payload.url)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to fetch URL: {exc}", "source_url": payload.url},
                            status_code=400)

    signals = collect_candidate_signals(payload.url, html)
    result = await run_in_threadpool(infer_from_text, signals["combined_text"])
    result["source_url"] = payload.url
    result["title"] = signals.get("title")
    result["links"] = signals.get("links", [])[:10]
    result["scripts"] = signals.get("scripts", [])[:10]
    result["css"] = signals.get("css", [])[:10]
    await run_in_threadpool(_store, "url", result, signals["combined_text"], payload.url)
    return result


@app.post("/api/infer/image")
async def infer_image(file: UploadFile = File(...)):
    original_name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(400, f"Unsupported image type: {ext or 'unknown'}")

    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 10MB)")
    if not contents:
        raise HTTPException(400, "Empty file")

    # Server-generated temp name: never use the client filename as a path
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        try:
            ocr_text = await run_in_threadpool(extract_text_from_image, temp_path)
        except Exception as exc:
            return JSONResponse({"error": f"OCR failed: {exc}"}, status_code=500)

        result = await run_in_threadpool(infer_from_text, ocr_text)
        result["source_type"] = "image"
        result["image_file"] = original_name
        result["ocr_text"] = ocr_text[:2000]
        await run_in_threadpool(_store, "image", result, ocr_text, None, original_name)
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
