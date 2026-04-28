from config import get_config
from vector_store import RAGVectorStore
from retriever import HybridRetriever
from pathlib import Path

cfg = get_config()
store = RAGVectorStore()
retriever = HybridRetriever(store)
print(f"BM25: {len(retriever.bm25_docs)} docs")

results = retriever.search("CIECAM02", top_k=2)
for r in results:
    title = r.get("title", r.get("source_name", ""))
    score = r.get("rerank_score", r.get("score", 0))
    print(f"  {title} score={score:.4f}")

p = Path(cfg.indexes.bm25_path) / "bm25_index.json"
if p.exists():
    print(f"BM25 cache: {p.stat().st_size/1024:.1f} KB")
else:
    print("BM25 cache: not found")