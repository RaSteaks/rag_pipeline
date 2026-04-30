"""Hybrid retrieval: Vector + BM25 + RRF fusion + Reranker + Diversity filter.

Pipeline:
  1. Vector search (Embedding model, ChromaDB) → Top-N
  2. BM25 keyword search (jieba + rank_bm25) → Top-N
  3. RRF (Reciprocal Rank Fusion) with source weight adjustment → merged candidates
  4. Reranker cross-encoder via llama.cpp /v1/rerank → precision ranking
     Falls back to RRF order if reranker is unavailable/timeout.
  5. Diversity filter (max_chunks_per_doc) → final results

BM25 index is persisted to disk and loaded on startup if available,
avoiding full rebuild on every restart.
"""
import json
import hashlib
import threading
import jieba
import rank_bm25
from pathlib import Path
from typing import Optional

from config import get_config
from vector_store import RAGVectorStore
from reranker import RerankerClient
from logger import setup_logger

log = setup_logger("rag")


class HybridRetriever:
    def __init__(self, store: RAGVectorStore, config_override=None):
        self._config = config_override or get_config()
        self.store = store
        self.reranker = RerankerClient(config_override)
        self.bm25: Optional[rank_bm25.BM25Okapi] = None
        self.bm25_docs: list[str] = []
        self.bm25_ids: list[str] = []
        self._bm25_signature = ""
        self._bm25_lock = threading.RLock()
        self._init_bm25()

    @property
    def _bm25_cache_path(self) -> Path:
        bm25_dir = Path(self._config.indexes.bm25_path or (Path(__file__).parent / "bm25_index"))
        bm25_dir.mkdir(parents=True, exist_ok=True)
        return bm25_dir / "bm25_index.json"

    def _save_bm25(self):
        """Persist BM25 tokenized data to disk for fast reload."""
        if self.bm25 is None:
            return
        cache_path = self._bm25_cache_path
        data = {
            "corpus_size": self.bm25.corpus_size,
            "doc_len": list(self.bm25.doc_len),
            "avgdl": self.bm25.avgdl,
            "idf": {str(k): float(v) for k, v in self.bm25.idf.items()} if hasattr(self.bm25, 'idf') else {},
            "docs": self.bm25_docs,
            "ids": self.bm25_ids,
            "store_signature": self._bm25_signature,
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.info(f"BM25 index saved to {cache_path}")

    def _store_signature(self, markers: Optional[dict] = None) -> str:
        """Fingerprint Chroma contents by IDs and chunk content markers."""
        markers = markers or self.store.get_all_document_markers()
        ids = markers.get("ids") or []
        metadatas = markers.get("metadatas") or []

        hasher = hashlib.sha256()
        rows = []
        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            rows.append((
                str(chunk_id),
                str(meta.get("content_hash", "")),
                str(meta.get("hash", "")),
                str(meta.get("source", "")),
                str(meta.get("position", "")),
                str(meta.get("is_image_description", "")),
            ))

        for row in sorted(rows, key=lambda item: item[0]):
            hasher.update("\x1f".join(row).encode("utf-8", errors="replace"))
            hasher.update(b"\n")
        return hasher.hexdigest()

    def _load_bm25(self) -> bool:
        """Load BM25 index from disk. Returns True if loaded successfully."""
        cache_path = self._bm25_cache_path
        if not cache_path.exists():
            return False

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            markers = self.store.get_all_document_markers()
            signature = self._store_signature(markers)
            cached_signature = data.get("store_signature", "")
            if cached_signature != signature:
                log.info("BM25 cache stale (store fingerprint mismatch), rebuilding...")
                return False

            with self._bm25_lock:
                self.bm25_docs = data["docs"]
                self.bm25_ids = data["ids"]
                self._bm25_signature = signature

                # Rebuild BM25 from tokenized corpus
                tokenized = [list(jieba.cut(doc)) for doc in self.bm25_docs]
                self.bm25 = rank_bm25.BM25Okapi(tokenized)
            log.info(f"BM25 index loaded from cache ({len(self.bm25_docs)} docs)")
            return True
        except Exception as e:
            log.warning(f"Failed to load BM25 cache: {e}, rebuilding...")
            return False

    def _init_bm25(self):
        """Build BM25 index from ChromaDB documents. Try cache first."""
        if self._load_bm25():
            return

        # Cache miss or stale, rebuild from ChromaDB
        all_docs = self.store.get_all_documents()
        if all_docs["documents"]:
            with self._bm25_lock:
                self.bm25_docs = all_docs["documents"]
                self.bm25_ids = all_docs["ids"]
                self._bm25_signature = self._store_signature({
                    "ids": all_docs["ids"],
                    "metadatas": all_docs.get("metadatas", []),
                })
                tokenized = [list(jieba.cut(doc)) for doc in self.bm25_docs]
                self.bm25 = rank_bm25.BM25Okapi(tokenized)
            log.info(f"BM25 index built with {len(self.bm25_docs)} documents")
            self._save_bm25()
        else:
            with self._bm25_lock:
                self.bm25 = None
                self.bm25_docs = []
                self.bm25_ids = []
                self._bm25_signature = self._store_signature({"ids": [], "metadatas": []})
            log.info("No documents for BM25 index yet")

    def rebuild_bm25(self):
        """Rebuild BM25 index after document changes."""
        # Clear stale cache
        cache_path = self._bm25_cache_path
        if cache_path.exists():
            cache_path.unlink()
        self._init_bm25()

    def search(self, query: str, top_k: int = None, debug: bool = False) -> list[dict]:
        """Full retrieval pipeline."""
        cfg = self._config.retrieval
        top_k = top_k or cfg.final_top_k

        # Stage 1: Vector search
        vector_results = self.store.search(query, top_k=cfg.vector_top_k)
        if debug:
            log.debug(f"Vector search returned {len(vector_results)} results")

        # Stage 2: BM25 search
        bm25_results = self._bm25_search(query, top_k=cfg.bm25_top_k)
        if debug:
            log.debug(f"BM25 search returned {len(bm25_results)} results")

        # Stage 3: RRF fusion
        fused = self._rrf_fusion(vector_results, bm25_results, cfg.rrf_k)
        if debug:
            log.debug(f"RRF fusion produced {len(fused)} candidates")

        # Trim to max candidates for reranker
        candidates = fused[:cfg.rrf_top_k]

        if not candidates:
            return []

        # Stage 4: Reranker
        reranked = self._rerank(query, candidates, debug)

        # Stage 5: Diversity filtering
        final = self._diversity_filter(reranked, cfg.max_chunks_per_doc, top_k)

        return final

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 keyword search."""
        with self._bm25_lock:
            bm25 = self.bm25
            docs = list(self.bm25_docs)
            ids = list(self.bm25_ids)

        if bm25 is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for i in top_indices:
            if scores[i] > 0:
                try:
                    metadatas = self.store.get_metadatas([ids[i]])
                    doc_meta = metadatas[0] if metadatas else {}
                except Exception:
                    doc_meta = {}

                results.append({
                    "text": docs[i],
                    "meta": doc_meta,
                    "score": float(scores[i]),
                    "source": "bm25",
                })
        return results

    def _rrf_fusion(self, vector_results: list[dict], bm25_results: list[dict],
                     k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion: merge vector and BM25 results."""
        rrf_scores = {}
        all_items = {}

        for rank, r in enumerate(vector_results):
            key = r["text"][:200]
            rrf_scores[key] = rrf_scores.get(key, 0) + r.get("meta", {}).get("weight", 1.0) / (k + rank + 1)
            all_items[key] = r

        for rank, r in enumerate(bm25_results):
            key = r["text"][:200]
            rrf_scores[key] = rrf_scores.get(key, 0) + r.get("meta", {}).get("weight", 1.0) / (k + rank + 1)
            all_items.setdefault(key, r)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [all_items[k] for k, _ in ranked]

    def _rerank(self, query: str, candidates: list[dict], debug: bool = False) -> list[dict]:
        """Apply reranker or fall back to RRF scores."""
        texts = [c.get("text", "") for c in candidates]

        result = self.reranker.rerank(query, texts, top_k=self._config.reranker.max_candidates)

        if result is None:
            if debug:
                log.debug("Reranker unavailable, using RRF order")
            return candidates

        # Reorder candidates by reranker scores
        final = []
        for idx, score in result:
            c = candidates[idx]
            final.append({
                "text": c.get("text", ""),
                "meta": c.get("meta", {}),
                "rerank_score": float(score),
                "source": c.get("source", ""),
                "title": c.get("meta", {}).get("title", ""),
                "source_name": c.get("meta", {}).get("source_name", ""),
                "relative_path": c.get("meta", {}).get("relative_path", ""),
            })
        return final

    def _diversity_filter(self, results: list[dict], max_per_doc: int,
                          top_k: int) -> list[dict]:
        """Ensure no single document dominates results."""
        doc_count = {}
        filtered = []

        for r in results:
            source = r.get("meta", {}).get("source", r.get("source", ""))
            if doc_count.get(source, 0) < max_per_doc:
                filtered.append(r)
                doc_count[source] = doc_count.get(source, 0) + 1
            if len(filtered) >= top_k:
                break

        return filtered
