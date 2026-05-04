"""RAG Pipeline API server + status dashboard."""
import base64
import hashlib
import hmac
import logging
import os
import signal
import secrets
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import AppConfig, get_config, reload_config, write_config_with_backup
from ingest import IncrementalIndexer
from logger import setup_logger
from retriever import HybridRetriever
from vector_store import RAGVectorStore

log = setup_logger("rag")

store: Optional[RAGVectorStore] = None
retriever: Optional[HybridRetriever] = None
indexer: Optional[IncrementalIndexer] = None

started_at = time.time()
last_sync_at: Optional[float] = None
last_sync_result: Optional[dict] = None
sync_running = False
sync_mode: Optional[str] = None
sync_started_at: Optional[float] = None
sync_error: Optional[str] = None
shutdown_requested = False
shutdown_started_at: Optional[float] = None
shutdown_reason: Optional[str] = None
SYNC_LOCK = threading.Lock()

LOG_BUFFER_MAX = 1000
LOG_BUFFER = deque(maxlen=LOG_BUFFER_MAX)
LOG_LOCK = threading.Lock()
LOG_SEQ = 0
CONFIG_PATH = BASE_DIR / "config.yaml"
CONFIG_EXAMPLE_PATH = BASE_DIR / "config.example.yaml"
AUTH_COOKIE_NAME = "rag_session"
PASSWORD_HASH_ITERATIONS = 260000


class InMemoryLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        global LOG_SEQ
        try:
            message = record.getMessage()
            with LOG_LOCK:
                LOG_SEQ += 1
                LOG_BUFFER.append(
                    {
                        "id": LOG_SEQ,
                        "ts": time.time(),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": message,
                    }
                )
        except Exception:
            pass


def _attach_in_memory_log_handler():
    logger = logging.getLogger("rag")
    for h in logger.handlers:
        if isinstance(h, InMemoryLogHandler):
            return
    handler = InMemoryLogHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


_attach_in_memory_log_handler()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _unb64(salt_b64),
            int(iterations),
        )
        return hmac.compare_digest(_b64(digest), digest_b64)
    except Exception:
        return False


def _auth_active() -> bool:
    cfg = get_config()
    return bool(cfg.auth.enabled and cfg.auth.username and cfg.auth.password_hash)


def _auth_secret() -> str:
    cfg = get_config()
    return cfg.auth.session_secret or cfg.auth.password_hash or secrets.token_urlsafe(32)


def _sign_session(username: str, expires_at: int) -> str:
    payload = f"{username}:{expires_at}"
    signature = hmac.new(
        _auth_secret().encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{signature}"


def _session_username(request: Request) -> Optional[str]:
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return None
    try:
        username, expires_at_text, signature = token.rsplit(":", 2)
        expires_at = int(expires_at_text)
    except ValueError:
        return None
    if expires_at < int(time.time()):
        return None
    expected = _sign_session(username, expires_at).rsplit(":", 1)[1]
    if not hmac.compare_digest(signature, expected):
        return None
    cfg = get_config()
    if username != cfg.auth.username:
        return None
    return username


def _is_authenticated(request: Request) -> bool:
    return not _auth_active() or _session_username(request) is not None


def _probe_model_service(endpoint: str, timeout: int = 4) -> dict:
    if not endpoint:
        return {"status": "unknown", "endpoint": endpoint, "message": "endpoint is empty"}
    try:
        t0 = time.perf_counter()
        resp = requests.get(f"{endpoint}/v1/models", timeout=timeout)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "status": "online" if resp.ok else "error",
            "endpoint": endpoint,
            "http_status": resp.status_code,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        return {"status": "offline", "endpoint": endpoint, "error": str(e)}


def _runtime_status() -> dict:
    cfg = get_config()
    embed_status = _probe_model_service(cfg.embedding.endpoint)
    rerank_status = (
        {"status": "disabled", "endpoint": cfg.reranker.endpoint}
        if not cfg.reranker.enabled
        else _probe_model_service(cfg.reranker.endpoint)
    )
    if retriever is not None:
        rerank_status["circuit"] = retriever.reranker.circuit_state

    total_chunks = 0
    if store is not None:
        try:
            total_chunks = store.count()
        except Exception as e:
            log.warning(f"Failed to count chunks: {e}")

    source_items = [
        {
            "name": s.name,
            "path": s.path,
            "enabled": s.enabled,
            "recursive": s.recursive,
            "weight": s.weight,
            "file_types": s.file_types,
        }
        for s in cfg.knowledge_sources
    ]

    return {
        "api": {"status": "online", "uptime_seconds": round(time.time() - started_at, 1)},
        "embedding": embed_status,
        "reranker": rerank_status,
        "indexes": {
            "chroma_chunks": total_chunks,
            "indexed_files": len(indexer.hash_cache) if indexer else 0,
            "bm25_ready": bool(retriever and retriever.bm25 is not None),
            "bm25_docs": len(retriever.bm25_docs) if retriever else 0,
        },
        "watcher": {
            "enabled": cfg.watchdog.enabled,
            "running": bool(indexer and indexer.watcher_running),
            "debounce_seconds": cfg.watchdog.debounce_seconds,
        },
        "image_indexing": {
            "enabled": cfg.image_description.enabled,
            "pending_jobs": indexer.image_jobs_pending if indexer else 0,
            "active_jobs": indexer.image_jobs_active if indexer else 0,
        },
        "config": {
            "source_count": len(cfg.knowledge_sources),
            "enabled_source_count": len([s for s in cfg.knowledge_sources if s.enabled]),
            "sources": source_items,
        },
        "sync": {
            "running": sync_running,
            "mode": sync_mode,
            "started_at": sync_started_at,
            "error": sync_error,
        },
        "last_sync": {"at": last_sync_at, "result": last_sync_result},
        "shutdown": {
            "requested": shutdown_requested,
            "started_at": shutdown_started_at,
            "reason": shutdown_reason,
        },
    }


def _sync_mode_name(source_name: Optional[str], rebuild: bool) -> str:
    if rebuild and source_name:
        return f"rebuild:{source_name}"
    if rebuild:
        return "full_rebuild"
    if source_name:
        return f"sync_source:{source_name}"
    return "full_sync"


def _rebuild_bm25_after_index_change(reason: str):
    if retriever is None:
        return
    log.info(f"Rebuilding BM25 after index change: {reason}")
    retriever.rebuild_bm25()


def _indexing_state() -> dict:
    image_pending = indexer.image_jobs_pending if indexer else 0
    image_active = indexer.image_jobs_active if indexer else 0
    return {
        "busy": bool(sync_running or image_pending or image_active),
        "sync": {
            "running": sync_running,
            "mode": sync_mode,
            "started_at": sync_started_at,
        },
        "image_indexing": {
            "pending_jobs": image_pending,
            "active_jobs": image_active,
        },
    }


def _wait_for_indexing_idle(timeout_seconds: int) -> tuple[bool, dict]:
    deadline = time.time() + max(0, timeout_seconds)
    state = _indexing_state()
    while state["busy"]:
        if time.time() >= deadline:
            return False, state
        time.sleep(0.5)
        state = _indexing_state()
    return True, state


def _signal_process_shutdown(delay_seconds: float = 0.5):
    def _job():
        time.sleep(delay_seconds)
        log.info("Stopping RAG Pipeline process")
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_job, name="shutdown-signal", daemon=True).start()


def _signal_process_restart(delay_seconds: float = 0.5):
    def _job():
        time.sleep(delay_seconds)
        log.info("Restarting RAG Pipeline process")
        os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=_job, name="restart-signal", daemon=True).start()


def _cancel_shutdown_request():
    global shutdown_requested, shutdown_started_at, shutdown_reason
    shutdown_requested = False
    shutdown_started_at = None
    shutdown_reason = None
    if indexer:
        indexer.start_watcher()


def _run_sync(source_name: Optional[str] = None, rebuild: bool = False) -> dict:
    global last_sync_at, last_sync_result, sync_running, sync_mode, sync_started_at, sync_error

    if indexer is None:
        return {"status": "error", "message": "Service not initialized"}
    if shutdown_requested:
        return {"status": "shutting_down", "message": "Shutdown already requested"}
    if not SYNC_LOCK.acquire(blocking=False):
        return {"status": "busy", "message": "Sync already running", "mode": sync_mode}

    sync_running = True
    sync_mode = _sync_mode_name(source_name, rebuild)
    sync_started_at = time.time()
    sync_error = None

    try:
        if rebuild and source_name:
            result = indexer.rebuild_source(source_name)
        elif rebuild:
            result = indexer.full_rebuild()
        elif source_name:
            result = indexer.rebuild_source(source_name)
        else:
            result = indexer.full_sync()

        if retriever:
            retriever.rebuild_bm25()

        last_sync_at = time.time()
        last_sync_result = result
        log.info(f"Sync result: {result}")
        return result
    except Exception as e:
        sync_error = str(e)
        log.exception(f"Sync failed: {e}")
        return {"status": "error", "message": str(e), "mode": sync_mode}
    finally:
        sync_running = False
        sync_mode = None
        SYNC_LOCK.release()


async def startup():
    global store, retriever, indexer

    cfg = get_config()
    log.info(f"RAG Pipeline starting on {cfg.server.host}:{cfg.server.port}")

    embed = _probe_model_service(cfg.embedding.endpoint)
    log.info(f"Embedding service: {embed}")

    if cfg.reranker.enabled:
        rerank = _probe_model_service(cfg.reranker.endpoint)
        log.info(f"Reranker service: {rerank}")

    store = RAGVectorStore()
    indexer = IncrementalIndexer(store)
    retriever = HybridRetriever(store)
    indexer.on_index_changed = _rebuild_bm25_after_index_change
    indexer.start_watcher()

    def _initial_sync_job():
        log.info("Running initial sync in background...")
        result = _run_sync()
        log.info(f"Initial sync finished: {result}")

    threading.Thread(
        target=_initial_sync_job,
        name="initial-sync",
        daemon=True,
    ).start()


async def shutdown():
    if indexer:
        indexer.stop_watcher()
    log.info("RAG Pipeline shut down")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    try:
        yield
    finally:
        await shutdown()


app = FastAPI(title="RAG Pipeline API", version="0.3.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.middleware("http")
async def require_dashboard_auth(request: Request, call_next):
    public_paths = {"/", "/health", "/auth/login", "/auth/logout", "/auth/me"}
    path = request.url.path
    if path in public_paths or path.startswith("/static/"):
        return await call_next(request)
    if not _is_authenticated(request):
        return JSONResponse({"status": "unauthorized", "message": "Login required"}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=()",
    )
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'",
    )
    return response


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    debug: bool = False


class SyncRequest(BaseModel):
    source_name: Optional[str] = None
    rebuild: bool = False


class ShutdownRequest(BaseModel):
    wait_for_indexing: bool = True
    timeout_seconds: int = Field(default=300, ge=0, le=3600)
    force: bool = False
    reason: Optional[str] = "update"


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class ConfigTextUpdateRequest(BaseModel):
    yaml_text: str
    create_if_missing: bool = True


class KnowledgeSourcePathUpdate(BaseModel):
    name: str
    path: str


class ConfigQuickUpdateRequest(BaseModel):
    embedding_model: Optional[str] = None
    reranker_model: Optional[str] = None
    knowledge_sources: list[KnowledgeSourcePathUpdate] = Field(default_factory=list)
    auth_enabled: Optional[bool] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None


def _read_config_text() -> tuple[str, Path, bool]:
    """Return (content, path, exists). Prefer config.yaml, fallback to example."""
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text(encoding="utf-8"), CONFIG_PATH, True
    if CONFIG_EXAMPLE_PATH.exists():
        return CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8"), CONFIG_PATH, False
    return "", CONFIG_PATH, False


def _apply_runtime_reload() -> dict:
    """Reload config and reinitialize runtime components."""
    global store, retriever, indexer
    if indexer:
        indexer.stop_watcher()
    cfg = reload_config(str(CONFIG_PATH))
    store = RAGVectorStore(config_override=cfg)
    indexer = IncrementalIndexer(store, config_override=cfg)
    retriever = HybridRetriever(store, config_override=cfg)
    indexer.on_index_changed = _rebuild_bm25_after_index_change
    indexer.start_watcher()
    return {"status": "reloaded", "sources": len(cfg.knowledge_sources)}


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"request": request},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/auth/me")
async def auth_me(request: Request):
    cfg = get_config()
    username = _session_username(request) if _auth_active() else None
    return {
        "enabled": _auth_active(),
        "configured": bool(cfg.auth.password_hash),
        "authenticated": not _auth_active() or username is not None,
        "username": username or (cfg.auth.username if not _auth_active() else None),
    }


@app.post("/auth/login")
async def auth_login(req: AuthLoginRequest, request: Request):
    cfg = get_config()
    if not _auth_active():
        return {"status": "ok", "enabled": False, "message": "Authentication is disabled"}
    if req.username != cfg.auth.username or not verify_password(req.password, cfg.auth.password_hash):
        return JSONResponse({"status": "error", "message": "Invalid username or password"}, status_code=401)

    max_age = max(300, int(cfg.auth.session_max_age_seconds or 86400))
    expires_at = int(time.time()) + max_age
    response = JSONResponse({"status": "ok", "username": cfg.auth.username})
    response.set_cookie(
        AUTH_COOKIE_NAME,
        _sign_session(cfg.auth.username, expires_at),
        max_age=max_age,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
    )
    return response


@app.post("/auth/logout")
async def auth_logout():
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(AUTH_COOKIE_NAME)
    return response


@app.get("/status")
async def status():
    return _runtime_status()


@app.get("/logs")
async def logs(
    limit: int = Query(default=200, ge=1, le=1000),
    min_level: str = Query(default="DEBUG"),
):
    level_order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
    threshold = level_order.get(min_level.upper(), 10)
    with LOG_LOCK:
        rows = list(LOG_BUFFER)
        max_id = LOG_SEQ
    filtered = [r for r in rows if level_order.get(r["level"], 0) >= threshold]
    return {"count": len(filtered[-limit:]), "max_id": max_id, "items": filtered[-limit:]}


@app.get("/config/editor")
async def config_editor():
    text, target_path, exists = _read_config_text()
    parsed = None
    parse_error = None
    try:
        data = yaml.safe_load(text) if text.strip() else {}
        parsed_model = AppConfig(**(data or {}))
        parsed = parsed_model.model_dump() if hasattr(parsed_model, "model_dump") else parsed_model.dict()
    except Exception as e:
        parse_error = str(e)
    return {
        "path": str(target_path),
        "exists": exists,
        "raw_yaml": text,
        "config": parsed,
        "parse_error": parse_error,
    }


@app.post("/config/save-text")
async def save_config_text(req: ConfigTextUpdateRequest):
    if not req.yaml_text.strip():
        return {"status": "error", "message": "yaml_text is empty"}
    if not CONFIG_PATH.exists() and not req.create_if_missing:
        return {"status": "error", "message": f"{CONFIG_PATH} not found"}
    try:
        data = yaml.safe_load(req.yaml_text)
        AppConfig(**(data or {}))
    except Exception as e:
        return {"status": "error", "message": f"Invalid config: {e}"}

    backup_path = write_config_with_backup(CONFIG_PATH, req.yaml_text)
    log.info(f"Config saved: {CONFIG_PATH}")
    reload_result = _apply_runtime_reload()
    return {
        "status": "ok",
        "path": str(CONFIG_PATH),
        "backup": str(backup_path) if backup_path else None,
        "reload": reload_result,
    }


@app.post("/config/save-quick")
async def save_config_quick(req: ConfigQuickUpdateRequest):
    text, _, exists = _read_config_text()
    if not text.strip():
        return {"status": "error", "message": "No base config content found"}

    try:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            return {"status": "error", "message": "Config YAML root must be a mapping"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse config: {e}"}

    if req.embedding_model is not None:
        data.setdefault("embedding", {})
        data["embedding"]["model"] = req.embedding_model
    if req.reranker_model is not None:
        data.setdefault("reranker", {})
        data["reranker"]["model"] = req.reranker_model

    if req.auth_enabled is not None or req.auth_username is not None or req.auth_password is not None:
        auth = data.setdefault("auth", {})
        if not isinstance(auth, dict):
            return {"status": "error", "message": "auth must be a mapping"}
        if req.auth_enabled is not None:
            auth["enabled"] = req.auth_enabled
        if req.auth_username is not None:
            username = req.auth_username.strip()
            if not username:
                return {"status": "error", "message": "Auth username cannot be empty"}
            auth["username"] = username
        if req.auth_password:
            auth["password_hash"] = hash_password(req.auth_password)
            auth["session_secret"] = secrets.token_urlsafe(32)
        if auth.get("enabled") and not auth.get("password_hash"):
            return {"status": "error", "message": "Set a password before enabling authentication"}

    updated_sources = 0
    missing_sources: list[str] = []
    source_updates = {item.name: item.path for item in req.knowledge_sources}
    if source_updates:
        ks = data.get("knowledge_sources", [])
        if not isinstance(ks, list):
            return {"status": "error", "message": "knowledge_sources must be a list"}
        existing_names = set()
        for src in ks:
            if not isinstance(src, dict):
                continue
            name = src.get("name")
            if name in source_updates:
                src["path"] = source_updates[name]
                updated_sources += 1
                existing_names.add(name)
        missing_sources = [n for n in source_updates.keys() if n not in existing_names]

    try:
        AppConfig(**data)
    except Exception as e:
        return {"status": "error", "message": f"Invalid config after update: {e}"}

    yaml_text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    backup_path = write_config_with_backup(CONFIG_PATH, yaml_text)
    log.info(f"Config quick-updated: {CONFIG_PATH}")
    reload_result = _apply_runtime_reload()
    return {
        "status": "ok",
        "path": str(CONFIG_PATH),
        "backup": str(backup_path) if backup_path else None,
        "created": not exists and CONFIG_PATH.exists(),
        "updated_sources": updated_sources,
        "missing_sources": missing_sources,
        "reload": reload_result,
    }


@app.post("/search")
async def search(req: SearchRequest):
    if retriever is None:
        return {"error": "Service not initialized"}
    results = retriever.search(req.query, top_k=req.top_k, debug=req.debug)
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/sync")
async def sync(req: Optional[SyncRequest] = None):
    payload = req or SyncRequest()
    return _run_sync(payload.source_name, payload.rebuild)


@app.post("/shutdown")
def shutdown_system(req: Optional[ShutdownRequest] = None):
    global shutdown_requested, shutdown_started_at, shutdown_reason

    payload = req or ShutdownRequest()
    shutdown_requested = True
    shutdown_started_at = time.time()
    shutdown_reason = payload.reason

    if indexer:
        indexer.stop_file_watcher()

    state = _indexing_state()
    if state["busy"] and payload.wait_for_indexing and not payload.force:
        ok, state = _wait_for_indexing_idle(payload.timeout_seconds)
        if not ok:
            _cancel_shutdown_request()
            return {
                "status": "busy",
                "message": "Indexing is still running; shutdown was not started",
                "indexing": state,
            }

    if state["busy"] and not payload.force:
        _cancel_shutdown_request()
        return {
            "status": "busy",
            "message": "Indexing is running; set wait_for_indexing=true or force=true",
            "indexing": state,
        }

    log.info(f"Shutdown requested: reason={payload.reason}, force={payload.force}")
    _signal_process_shutdown()
    return {
        "status": "shutting_down",
        "reason": payload.reason,
        "force": payload.force,
        "indexing": state,
    }


@app.post("/restart")
def restart_system(req: Optional[ShutdownRequest] = None):
    global shutdown_requested, shutdown_started_at, shutdown_reason

    payload = req or ShutdownRequest(reason="restart")
    shutdown_requested = True
    shutdown_started_at = time.time()
    shutdown_reason = payload.reason

    if indexer:
        indexer.stop_file_watcher()

    state = _indexing_state()
    if state["busy"] and payload.wait_for_indexing and not payload.force:
        ok, state = _wait_for_indexing_idle(payload.timeout_seconds)
        if not ok:
            _cancel_shutdown_request()
            return {
                "status": "busy",
                "message": "Indexing is still running; restart was not started",
                "indexing": state,
            }

    if state["busy"] and not payload.force:
        _cancel_shutdown_request()
        return {
            "status": "busy",
            "message": "Indexing is running; set wait_for_indexing=true or force=true",
            "indexing": state,
        }

    log.info(f"Restart requested: reason={payload.reason}, force={payload.force}")
    _signal_process_restart()
    return {
        "status": "restarting",
        "reason": payload.reason,
        "force": payload.force,
        "indexing": state,
    }


@app.post("/reload-config")
async def reload_config_endpoint():
    if not CONFIG_PATH.exists():
        return {"status": "error", "message": f"Config not found: {CONFIG_PATH}"}
    return _apply_runtime_reload()


@app.get("/stats")
async def stats():
    if store is None:
        return {"error": "Service not initialized"}
    cfg = get_config()
    return {
        "total_chunks": store.count(),
        "knowledge_sources": [
            {"name": s.name, "path": s.path, "enabled": s.enabled}
            for s in cfg.knowledge_sources
        ],
        "indexed_files": len(indexer.hash_cache) if indexer else 0,
        "reranker_enabled": cfg.reranker.enabled,
    }


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
