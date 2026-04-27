"""RAG Pipeline API server.

Startup sequence:
  1. Load config from config.yaml
  2. Check Embedding service (:8080) availability
  3. Check Reranker service (:8081) availability (warn if offline, not fatal)
  4. Initialize ChromaDB vector store
  5. Run initial index sync (incremental, skips unchanged files)
  6. Build BM25 index from ChromaDB documents
  7. Start file watcher for incremental updates
  8. Start uvicorn on configured host:port
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

from config import get_config, reload_config
from vector_store import RAGVectorStore
from retriever import HybridRetriever
from ingest import IncrementalIndexer
from logger import setup_logger

log = setup_logger("rag")

app = FastAPI(title="RAG Pipeline API", version="0.1.0")

store: Optional[RAGVectorStore] = None
retriever: Optional[HybridRetriever] = None
indexer: Optional[IncrementalIndexer] = None
observer = None


@app.on_event("startup")
async def startup():
    global store, retriever, indexer, observer

    config = get_config()
    log.info(f"RAG Pipeline starting on {config.server.host}:{config.server.port}")

    # Check embedding service
    import requests
    emb_url = config.embedding.endpoint
    try:
        resp = requests.get(f"{emb_url}/v1/models", timeout=5)
        log.info(f"Embedding service at {emb_url}: OK ({resp.status_code})")
    except Exception as e:
        log.warning(f"Embedding service at {emb_url}: NOT REACHABLE ({e})")
        log.warning("Vector search will fail without embedding service!")

    # Check reranker service
    if config.reranker.enabled:
        rerank_url = config.reranker.endpoint
        try:
            resp = requests.get(f"{rerank_url}/v1/models", timeout=5)
            log.info(f"Reranker service at {rerank_url}: OK ({resp.status_code})")
        except Exception as e:
            log.warning(f"Reranker service at {rerank_url}: NOT REACHABLE ({e})")
            log.warning("Will fall back to RRF-only ranking")

    # Initialize store and indexer
    store = RAGVectorStore()
    indexer = IncrementalIndexer(store)

    # Initial sync
    log.info("Running initial sync...")
    result = indexer.full_sync()
    log.info(f"Sync done: {result}")

    # Initialize retriever (builds BM25 index)
    retriever = HybridRetriever(store)

    # Start file watcher
    observer = indexer.start_watcher()


@app.on_event("shutdown")
async def shutdown():
    global observer
    if observer:
        observer.stop()
        observer.join()
    log.info("RAG Pipeline shut down")


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    debug: bool = False


class SyncRequest(BaseModel):
    source_name: Optional[str] = None
    rebuild: bool = False


@app.post("/search")
async def search(req: SearchRequest):
    """Search the knowledge base."""
    if retriever is None:
        return {"error": "Service not initialized"}

    results = retriever.search(req.query, top_k=req.top_k, debug=req.debug)
    log.debug(f"Search '{req.query}' returned {len(results)} results")
    return {
        "query": req.query,
        "count": len(results),
        "results": results,
    }


@app.post("/sync")
async def sync(req: SyncRequest = None):
    """Trigger index synchronization."""
    global retriever
    if indexer is None:
        return {"error": "Service not initialized"}

    if req and req.rebuild:
        log.info("Full rebuild requested")
        result = indexer.full_rebuild()
    elif req and req.source_name:
        log.info(f"Rebuild source requested: {req.source_name}")
        result = indexer.rebuild_source(req.source_name)
    else:
        log.info("Incremental sync requested")
        result = indexer.full_sync()

    # Rebuild BM25 after sync
    if retriever:
        retriever.rebuild_bm25()

    log.info(f"Sync result: {result}")
    return result


@app.post("/reload-config")
async def reload_config_endpoint():
    """Reload configuration from file."""
    global store, retriever, indexer
    config = reload_config()
    store = RAGVectorStore(config_override=config)
    indexer = IncrementalIndexer(store, config_override=config)
    retriever = HybridRetriever(store, config_override=config)
    log.info(f"Config reloaded: {len(config.knowledge_sources)} sources")
    return {"status": "reloaded", "sources": len(config.knowledge_sources)}


@app.get("/stats")
async def stats():
    """Get index statistics."""
    if store is None:
        return {"error": "Service not initialized"}
    config = get_config()
    return {
        "total_chunks": store.count(),
        "knowledge_sources": [
            {"name": s.name, "path": s.path, "enabled": s.enabled}
            for s in config.knowledge_sources
        ],
        "indexed_files": len(indexer.hash_cache) if indexer else 0,
        "reranker_enabled": config.reranker.enabled,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(app, host=config.server.host, port=config.server.port)