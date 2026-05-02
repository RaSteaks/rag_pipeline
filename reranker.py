"""Reranker client using llama.cpp /v1/rerank endpoint.

NOTE: The llama.cpp server must be started with --rerank flag to enable this endpoint.
The endpoint format is OpenAI-compatible rerank:
  POST /v1/rerank with {"model": ..., "query": ..., "documents": [...]}
  Returns {"results": [{"index": 0, "relevance_score": 0.99}, ...]}

Long texts are truncated to MAX_TEXT_LENGTH chars to avoid 500 errors from context overflow.
Falls back to RRF scores when the reranker is unavailable (timeout/connection error).
"""
import requests
import threading
import time
from config import get_config
from logger import setup_logger
log = setup_logger("rag")


class RerankerClient:
    """Client for Qwen3-Reranker-4B via llama.cpp server."""

    def __init__(self, config_override=None):
        self._config = config_override or get_config()
        self._lock = threading.Lock()
        self._failure_count = 0
        self._circuit_open_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._config.reranker.enabled

    @property
    def endpoint(self) -> str:
        return self._config.reranker.endpoint

    @property
    def fallback_to_rrf(self) -> bool:
        cfg = self._config.reranker
        if cfg.skip_if_unavailable is not None:
            return cfg.skip_if_unavailable
        return cfg.fallback_to_rrf

    @property
    def circuit_state(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "open": self._circuit_open_until > now,
                "open_until": self._circuit_open_until if self._circuit_open_until > now else None,
                "failure_count": self._failure_count,
            }

    def _record_success(self):
        with self._lock:
            self._failure_count = 0
            self._circuit_open_until = 0.0

    def _record_failure(self, reason: str):
        cfg = self._config.reranker
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= max(1, cfg.circuit_breaker_failures):
                self._circuit_open_until = time.time() + max(1, cfg.circuit_breaker_cooldown_seconds)
                log.warning(
                    "Reranker circuit opened for "
                    f"{cfg.circuit_breaker_cooldown_seconds}s after {self._failure_count} failures: {reason}"
                )

    def rerank(self, query: str, texts: list[str],
               top_k: int = None) -> list[tuple[int, float]] | None:
        """
        Rerank texts by relevance to query.
        Returns list of (original_index, score) sorted by score descending.
        Returns None if reranker is unavailable (caller should fall back to RRF).
        """
        if not self.enabled:
            return None

        cfg = self._config.reranker
        max_candidates = top_k or cfg.max_candidates

        # Trim to max candidates and truncate long texts
        # llama.cpp has context limit, truncate texts to avoid 500 errors
        MAX_TEXT_LENGTH = 400  # chars, ~100 tokens, safe for CPU inference
        texts = [t[:MAX_TEXT_LENGTH] if len(t) > MAX_TEXT_LENGTH else t
                 for t in texts[:max_candidates]]

        with self._lock:
            if self._circuit_open_until > time.time():
                log.info("Reranker circuit open, using RRF order")
                return None

        try:
            # llama.cpp rerank endpoint: /v1/rerank (with --rerank flag)
            resp = requests.post(
                f"{self.endpoint}/v1/rerank",
                json={
                    "model": cfg.model,
                    "query": query,
                    "documents": texts,
                },
                timeout=cfg.timeout_seconds,
            )
            resp.raise_for_status()
            # llama.cpp /v1/rerank returns {"results": [{"index": 0, "relevance_score": 0.99}, ...]}
            data = resp.json()
            results = data.get("results", [])
            if not results:
                log.warning("Reranker returned empty results")
                self._record_failure("empty results")
                return None

            # Map relevance scores back to original order
            # results[i]["index"] = original position, results[i]["relevance_score"] = score
            scores = [0.0] * len(texts)
            for r in results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", 0.0)
                if idx < len(texts):
                    scores[idx] = score

            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            self._record_success()
            return ranked

        except requests.exceptions.ConnectionError as e:
            log.info("Reranker service not available, using RRF order")
            self._record_failure(str(e))
            return None
        except requests.exceptions.Timeout as e:
            log.info("Reranker timeout, using RRF order")
            self._record_failure(str(e))
            return None
        except Exception as e:
            log.warning(f"Reranker error, using RRF order: {e}")
            self._record_failure(str(e))
            return None
