"""RAG Pipeline API server + status dashboard."""
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import requests
import uvicorn
import yaml
from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Add project root to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import AppConfig, get_config, reload_config
from ingest import IncrementalIndexer
from logger import setup_logger
from retriever import HybridRetriever
from vector_store import RAGVectorStore

log = setup_logger("rag")
app = FastAPI(title="RAG Pipeline API", version="0.3.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

store: Optional[RAGVectorStore] = None
retriever: Optional[HybridRetriever] = None
indexer: Optional[IncrementalIndexer] = None

started_at = time.time()
last_sync_at: Optional[float] = None
last_sync_result: Optional[dict] = None

LOG_BUFFER_MAX = 1000
LOG_BUFFER = deque(maxlen=LOG_BUFFER_MAX)
LOG_LOCK = threading.Lock()
LOG_SEQ = 0
CONFIG_PATH = BASE_DIR / "config.yaml"
CONFIG_EXAMPLE_PATH = BASE_DIR / "config.example.yaml"


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
        "config": {
            "source_count": len(cfg.knowledge_sources),
            "enabled_source_count": len([s for s in cfg.knowledge_sources if s.enabled]),
            "sources": source_items,
        },
        "last_sync": {"at": last_sync_at, "result": last_sync_result},
    }


@app.on_event("startup")
async def startup():
    global store, retriever, indexer, last_sync_at, last_sync_result

    cfg = get_config()
    log.info(f"RAG Pipeline starting on {cfg.server.host}:{cfg.server.port}")

    embed = _probe_model_service(cfg.embedding.endpoint)
    log.info(f"Embedding service: {embed}")

    if cfg.reranker.enabled:
        rerank = _probe_model_service(cfg.reranker.endpoint)
        log.info(f"Reranker service: {rerank}")

    store = RAGVectorStore()
    indexer = IncrementalIndexer(store)

    log.info("Running initial sync...")
    result = indexer.full_sync()
    last_sync_result = result
    last_sync_at = time.time()
    log.info(f"Initial sync done: {result}")

    retriever = HybridRetriever(store)
    indexer.start_watcher()


@app.on_event("shutdown")
async def shutdown():
    if indexer:
        indexer.stop_watcher()
    log.info("RAG Pipeline shut down")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    debug: bool = False


class SyncRequest(BaseModel):
    source_name: Optional[str] = None
    rebuild: bool = False


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
    indexer.start_watcher()
    return {"status": "reloaded", "sources": len(cfg.knowledge_sources)}


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    CONFIG_PATH.write_text(req.yaml_text, encoding="utf-8")
    log.info(f"Config saved: {CONFIG_PATH}")
    reload_result = _apply_runtime_reload()
    return {"status": "ok", "path": str(CONFIG_PATH), "reload": reload_result}


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
    CONFIG_PATH.write_text(yaml_text, encoding="utf-8")
    log.info(f"Config quick-updated: {CONFIG_PATH}")
    reload_result = _apply_runtime_reload()
    return {
        "status": "ok",
        "path": str(CONFIG_PATH),
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
    global last_sync_at, last_sync_result
    if indexer is None:
        return {"error": "Service not initialized"}

    payload = req or SyncRequest()
    if payload.rebuild and payload.source_name:
        result = indexer.rebuild_source(payload.source_name)
    elif payload.rebuild:
        result = indexer.full_rebuild()
    elif payload.source_name:
        result = indexer.rebuild_source(payload.source_name)
    else:
        result = indexer.full_sync()

    if retriever:
        retriever.rebuild_bm25()

    last_sync_at = time.time()
    last_sync_result = result
    log.info(f"Sync result: {result}")
    return result


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
