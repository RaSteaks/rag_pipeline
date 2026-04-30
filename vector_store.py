"""ChromaDB vector store with local Qwen3-Embedding-8B.

Batch upsert with error handling: if a batch of chunks fails to embed,
each chunk is retried individually. Chunks that still fail are skipped
with a warning, rather than crashing the entire sync process.
"""
import chromadb
import threading
from openai import OpenAI
from pathlib import Path
from typing import Optional

from config import get_config
from logger import setup_logger
log = setup_logger("rag")


class RAGVectorStore:
    def __init__(self, config_override=None):
        self._config = config_override or get_config()
        persist_dir = self._config.indexes.chroma_path or str(Path(__file__).parent / "chromadb")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200,
            }
        )
        self._embed_client = None
        self._batch_size = self._config.embedding.batch_size
        self._collection_lock = threading.RLock()

    @property
    def embed_client(self):
        if self._embed_client is None:
            cfg = self._config.embedding
            self._embed_client = OpenAI(
                base_url=f"{cfg.endpoint}/v1",
                api_key="not-needed",
                timeout=cfg.timeout_seconds,
            )
        return self._embed_client

    @staticmethod
    def _visible(meta: Optional[dict]) -> bool:
        return not meta or meta.get("index_state") != "pending"

    @staticmethod
    def _filter_visible(result: dict) -> dict:
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        filtered = {"ids": [], "documents": [], "metadatas": []}
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            if not RAGVectorStore._visible(meta):
                continue
            filtered["ids"].append(chunk_id)
            if docs:
                filtered["documents"].append(docs[idx] if idx < len(docs) else "")
            if metas:
                filtered["metadatas"].append(meta)
        return filtered

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local Qwen3-Embedding-8B."""
        cfg = self._config.embedding
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            resp = self.embed_client.embeddings.create(
                model=cfg.model,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    def add_chunks(self, chunks: list[dict]):
        """Batch upsert chunks into ChromaDB, with error handling per chunk."""
        if not chunks:
            return

        # Process in smaller batches to handle Embedding API errors
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            try:
                embeddings = self.embed_texts(texts)
                with self._collection_lock:
                    self.collection.upsert(
                        ids=[c["chunk_id"] for c in batch],
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=[c["meta"] for c in batch],
                    )
            except Exception as e:
                print(f"Failed to embed batch {i//batch_size}: {e}")
                # Try individual chunks
                for j, chunk in enumerate(batch):
                    try:
                        single_emb = self.embed_texts([chunk["text"]])
                        with self._collection_lock:
                            self.collection.upsert(
                                ids=[chunk["chunk_id"]],
                                embeddings=single_emb,
                                documents=[chunk["text"]],
                                metadatas=[chunk["meta"]],
                            )
                    except Exception as e2:
                        print(f"Skipping chunk {chunk['chunk_id']}: {e2}")

    def delete_by_source(self, source: str):
        """Delete all chunks from a specific source file."""
        try:
            with self._collection_lock:
                self.collection.delete(where={"source": source})
        except Exception:
            pass

    def delete_by_source_name(self, source_name: str):
        """Delete all chunks from a knowledge source."""
        try:
            with self._collection_lock:
                self.collection.delete(where={"source_name": source_name})
        except Exception:
            pass

    def update_metadatas(self, ids: list[str], metadatas: list[dict]):
        """Update metadata for existing chunks through the collection lock."""
        if not ids:
            return
        with self._collection_lock:
            self.collection.update(ids=ids, metadatas=metadatas)

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Vector similarity search."""
        cfg = self._config.retrieval
        top_k = top_k or cfg.vector_top_k
        with self._collection_lock:
            count = self.collection.count()
        if count == 0:
            return []
        top_k = min(top_k, count)

        query_embedding = self.embed_texts([query])[0]
        with self._collection_lock:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        if not results["documents"][0]:
            return []
        return [
            {
                "text": doc,
                "meta": meta,
                "score": 1 - dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
            if self._visible(meta)
        ]

    def get_all_documents(self) -> dict:
        """Get all documents for BM25 index building."""
        with self._collection_lock:
            count = self.collection.count()
            if count == 0:
                return {"ids": [], "documents": [], "metadatas": []}
            return self._filter_visible(self.collection.get(include=["documents", "metadatas"]))

    def get_all_document_markers(self) -> dict:
        """Get stable IDs and metadata for cache consistency checks."""
        with self._collection_lock:
            count = self.collection.count()
            if count == 0:
                return {"ids": [], "metadatas": []}
            markers = self._filter_visible(self.collection.get(include=["metadatas"]))
            markers.pop("documents", None)
            return markers

    def get_metadatas(self, ids: list[str]) -> list[dict]:
        """Fetch metadata for specific IDs through the collection lock."""
        if not ids:
            return []
        with self._collection_lock:
            result = self.collection.get(ids=ids, include=["metadatas"])
        return result.get("metadatas") or []

    def count(self) -> int:
        with self._collection_lock:
            return self.collection.count()
