import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import get_config
from vector_store import RAGVectorStore
from retriever import HybridRetriever

cfg = get_config()
store = RAGVectorStore()
retriever = HybridRetriever(store)

print('=== FULL RAG SEARCH WITH RERANKER ===')
print()

queries = [
    'CIECAM02 色貌模型参数',
    'ARRI Log C4 specification',
    '色域映射 gamut mapping',
]

for q in queries:
    print(f'Query: {q}')
    results = retriever.search(q, top_k=5)
    for i, r in enumerate(results):
        has_rerank = 'rerank_score' in r
        if has_rerank:
            rs = r['rerank_score']
            src = r.get('source_name', '')
            title = r.get('title', '')
            print(f'  [{i+1}] RERANKED score={rs:.4f} src={src} title={title}')
        else:
            score = r.get('score', 0)
            src = r.get('meta', {}).get('source_name', r.get('source', ''))
            title = r.get('meta', {}).get('title', r.get('title', ''))
            print(f'  [{i+1}] RRF score={score:.4f} src={src} title={title}')
    print()