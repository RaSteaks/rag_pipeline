"""ChromaDB vector store with local Qwen3-Embedding-8B.

Batch upsert with error handling: if a batch of chunks fails to embed,
each chunk is retried individually. Chunks that still fail are skipped
with a warning, rather than crashing the entire sync process.
"""
import chromadb
from openai import OpenAI
from pathlib import Path
from typing import Optional

from config import get_config
from logger import setup_logger
log = setup_logger("rag")


class RAGVectorStore:
    def __init__(self, config_override=None):
        self._config = config_override or get_config()
        persist_dir = self._config.indexes.chroma_path
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
            self.collection.delete(where={"source": source})
        except Exception:
            pass

    def delete_by_source_name(self, source_name: str):
        """Delete all chunks from a knowledge source."""
        try:
            self.collection.delete(where={"source_name": source_name})
        except Exception:
            pass

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Vector similarity search."""
        cfg = self._config.retrieval
        top_k = top_k or cfg.vector_top_k
        count = self.collection.count()
        if count == 0:
            return []
        top_k = min(top_k, count)

        query_embedding = self.embed_texts([query])[0]
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
        ]

    def get_all_documents(self) -> dict:
        """Get all documents for BM25 index building."""
        count = self.collection.count()
        if count == 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        return self.collection.count()