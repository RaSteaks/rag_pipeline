"""Incremental indexer with manifest + watchdog support."""
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from chunker import Chunk, chunk_document
from config import AppConfig, KnowledgeSource, get_config
from logger import setup_logger
from parsers import SUPPORT_EXTS_WITH_CONFIG, parse_file
from vector_store import RAGVectorStore

log = setup_logger("rag")


class IncrementalIndexer:
    def __init__(self, store: Optional[RAGVectorStore] = None, config_override: Optional[AppConfig] = None):
        self._config = config_override or get_config()
        self.store = store or RAGVectorStore(config_override=self._config)
        self._lock = threading.Lock()
        self._observer: Optional[Observer] = None
        self.on_index_changed: Optional[Callable[[str], None]] = None
        self._image_jobs: queue.Queue[tuple[dict, str]] = queue.Queue()
        self._image_job_keys: set[tuple[str, str]] = set()
        self._image_jobs_lock = threading.Lock()
        self._image_worker: Optional[threading.Thread] = None
        self._image_worker_stop = threading.Event()

        manifest_path = self._config.indexes.manifest_path or "index_manifest.json"
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.parent != Path("."):
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.hash_cache = self._load_manifest()

    def _notify_index_changed(self, reason: str):
        if not self.on_index_changed:
            return
        try:
            self.on_index_changed(reason)
        except Exception as e:
            log.warning(f"Index change callback failed after {reason}: {e}")

    def _load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to load manifest: {e}")
            return {}

    def _save_manifest(self):
        try:
            self.manifest_path.write_text(
                json.dumps(self.hash_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.error(f"Failed to save manifest: {e}")

    def _source_by_name(self, source_name: str) -> Optional[KnowledgeSource]:
        for source in self._config.knowledge_sources:
            if source.name == source_name:
                return source
        return None

    def _iter_sources(self, source_name: Optional[str] = None) -> list[KnowledgeSource]:
        if source_name:
            source = self._source_by_name(source_name)
            if source is None:
                raise ValueError(f"Source not found: {source_name}")
            return [source]
        return [s for s in self._config.knowledge_sources if s.enabled]

    def _should_exclude(self, path: Path) -> bool:
        normalized = str(path).replace("\\", "/")
        for exc_dir in self._config.exclude.dirs:
            key = exc_dir.strip("/\\")
            if key and f"/{key}/" in f"/{normalized}/":
                return True
        for pattern in self._config.exclude.files:
            if path.match(pattern):
                return True
        return False

    def _collect_source_files(self, source: KnowledgeSource) -> list[Path]:
        root = Path(source.path)
        if not root.exists():
            log.warning(f"Source path does not exist, skipped: {source.path}")
            return []

        all_supported = SUPPORT_EXTS_WITH_CONFIG()
        allowed_exts = {e.lower() for e in source.file_types} & all_supported
        if not allowed_exts:
            allowed_exts = all_supported

        files: list[Path] = []
        for ext in allowed_exts:
            iterator = root.rglob(f"*{ext}") if source.recursive else root.glob(f"*{ext}")
            for p in iterator:
                if p.is_file() and not self._should_exclude(p):
                    files.append(p.resolve())

        # Deduplicate
        return list(dict.fromkeys(files))

    def _remove_deleted_files(self, valid_files: set[str], source_roots: list[Path]) -> int:
        removed = 0
        cache_paths = list(self.hash_cache.keys())
        roots = [r.resolve() for r in source_roots]

        for cached in cache_paths:
            cached_path = Path(cached)
            in_target_sources = any(self._is_under(cached_path, root) for root in roots)
            if not in_target_sources:
                continue
            if cached not in valid_files:
                self.store.delete_by_source(cached)
                with self._lock:
                    self.hash_cache.pop(cached, None)
                removed += 1
        return removed

    def _is_under(self, path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root)
            return True
        except Exception:
            return False

    def _index_file(self, filepath: Path, rebuild: bool = False) -> tuple[bool, int]:
        parsed = parse_file(str(filepath))
        if parsed is None:
            return False, 0

        path_key = str(Path(parsed.source).resolve())
        old_hash = self.hash_cache.get(path_key)
        if not rebuild and old_hash == parsed.hash:
            return False, 0

        if old_hash or rebuild:
            self.store.delete_by_source(path_key)

        chunks = chunk_document(parsed.to_dict())
        if not chunks:
            return False, 0
        self.store.add_chunks([c.to_dict() for c in chunks])

        with self._lock:
            self.hash_cache[path_key] = parsed.hash

        if path_key.lower().endswith(".pdf"):
            self._enqueue_pdf_image_description(parsed.to_dict(), path_key)

        return True, len(chunks)

    def sync(self, source_name: Optional[str] = None, rebuild: bool = False) -> dict:
        started_at = time.time()
        try:
            sources = self._iter_sources(source_name)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        scanned = 0
        indexed = 0
        skipped = 0
        total_chunks = 0
        deleted = 0

        source_roots = [Path(s.path).resolve() for s in sources if Path(s.path).exists()]
        valid_files: set[str] = set()

        if rebuild:
            for source in sources:
                self.store.delete_by_source_name(source.name)
            for cached in list(self.hash_cache.keys()):
                cached_path = Path(cached)
                if any(self._is_under(cached_path, root) for root in source_roots):
                    self.hash_cache.pop(cached, None)

        for source in sources:
            files = self._collect_source_files(source)
            scanned += len(files)
            for filepath in files:
                path_key = str(filepath.resolve())
                valid_files.add(path_key)
                try:
                    changed, chunks = self._index_file(filepath, rebuild=rebuild)
                    if changed:
                        indexed += 1
                        total_chunks += chunks
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    log.error(f"Indexing failed for {filepath}: {e}")

        deleted = self._remove_deleted_files(valid_files, source_roots)
        self._save_manifest()

        return {
            "status": "ok",
            "mode": "rebuild" if rebuild else "incremental",
            "source_name": source_name,
            "scanned_files": scanned,
            "indexed_files": indexed,
            "skipped_files": skipped,
            "deleted_files": deleted,
            "added_chunks": total_chunks,
            "manifest_entries": len(self.hash_cache),
            "elapsed_seconds": round(time.time() - started_at, 3),
        }

    def full_sync(self) -> dict:
        return self.sync()

    def full_rebuild(self) -> dict:
        return self.sync(rebuild=True)

    def rebuild_source(self, source_name: str) -> dict:
        return self.sync(source_name=source_name, rebuild=True)

    def start_watcher(self) -> Optional[Observer]:
        if not self._config.watchdog.enabled:
            log.info("Watchdog disabled in config")
            return None
        if self._observer and self._observer.is_alive():
            return self._observer

        observer = Observer()
        handler = _WatchHandler(self)
        watched = 0
        for source in self._iter_sources():
            root = Path(source.path)
            if root.exists():
                observer.schedule(handler, str(root), recursive=source.recursive)
                watched += 1
                log.info(f"Watching source: {source.name} ({source.path})")

        if watched == 0:
            log.warning("No valid source path for watchdog")
            return None

        observer.start()
        self._observer = observer
        return self._observer

    def stop_watcher(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        self.stop_background_workers()

    @property
    def watcher_running(self) -> bool:
        return bool(self._observer and self._observer.is_alive())

    @property
    def image_jobs_pending(self) -> int:
        return self._image_jobs.qsize()

    def stop_background_workers(self):
        self._image_worker_stop.set()
        if self._image_worker and self._image_worker.is_alive():
            self._image_worker.join(timeout=2)

    def _source_weight(self, source_name: str) -> float:
        try:
            for source in self._config.knowledge_sources:
                if source.name == source_name:
                    return source.weight
        except Exception:
            pass
        return 1.0

    def _start_image_worker(self):
        if self._image_worker and self._image_worker.is_alive():
            return
        self._image_worker_stop.clear()
        self._image_worker = threading.Thread(
            target=self._image_worker_loop,
            name="pdf-image-indexer",
            daemon=True,
        )
        self._image_worker.start()

    def _enqueue_pdf_image_description(self, parsed_doc: dict, filepath: str):
        """Queue PDF image description so text indexing stays responsive."""
        cfg = self._config.image_description
        if not cfg.enabled:
            return

        doc_hash = parsed_doc.get("hash", "")
        job_key = (filepath, doc_hash)
        with self._image_jobs_lock:
            if job_key in self._image_job_keys:
                return
            self._image_job_keys.add(job_key)

        self._start_image_worker()
        self._image_jobs.put((parsed_doc, filepath))
        log.info(f"Queued PDF image description job for {filepath}")

    def _image_worker_loop(self):
        while not self._image_worker_stop.is_set():
            try:
                parsed_doc, filepath = self._image_jobs.get(timeout=0.5)
            except queue.Empty:
                continue

            doc_hash = parsed_doc.get("hash", "")
            try:
                self._describe_pdf_images(parsed_doc, filepath)
            finally:
                with self._image_jobs_lock:
                    self._image_job_keys.discard((filepath, doc_hash))
                self._image_jobs.task_done()

    def _describe_pdf_images(self, parsed_doc, filepath: str):
        """Generate image descriptions for a PDF and add as chunks."""
        try:
            from image_describer import describe_pdf_images
        except ImportError:
            log.warning("image_describer not available, skipping image descriptions")
            return

        cfg = self._config.image_description
        if not cfg.enabled:
            return

        log.info(f"Generating image descriptions for {filepath}")
        try:
            descriptions = describe_pdf_images(pdf_path=filepath)
        except Exception as e:
            log.warning(f"Image description failed for {filepath}: {e}")
            return

        if isinstance(parsed_doc, dict):
            doc_hash = parsed_doc.get("hash", "")
            source = parsed_doc.get("source", filepath)
            doc_format = parsed_doc.get("format", "pdf")
            modified_at = parsed_doc.get("modified_at", 0)
            title = parsed_doc.get("title", "")
            source_name = parsed_doc.get("source_name", "")
            relative_path = parsed_doc.get("relative_path", "")
        else:
            doc_hash = getattr(parsed_doc, "hash", "")
            source = getattr(parsed_doc, "source", filepath)
            doc_format = getattr(parsed_doc, "format", "pdf")
            modified_at = getattr(parsed_doc, "modified_at", 0)
            title = getattr(parsed_doc, "title", "")
            source_name = getattr(parsed_doc, "source_name", "")
            relative_path = getattr(parsed_doc, "relative_path", "")

        with self._lock:
            current_hash = self.hash_cache.get(filepath)
        if current_hash != doc_hash:
            log.info(f"Skipping stale image descriptions for {filepath}")
            return

        weight = self._source_weight(source_name)

        chunks = []
        for desc in descriptions:
            if not desc.description.strip():
                continue
            chunk = Chunk(
                chunk_id=f"{doc_hash}_img_{desc.page_num}",
                source=source,
                text=f"[Page {desc.page_num} 图像描述] {desc.description}",
                meta={
                    "hash": doc_hash,
                    "format": doc_format,
                    "position": desc.page_num,
                    "modified_at": modified_at,
                    "title": title,
                    "source_name": source_name,
                    "relative_path": relative_path,
                    "weight": weight,
                    "content_hash": hashlib.sha256(desc.description.encode()).hexdigest()[:12],
                    "is_image_description": True,
                    "image_page": desc.page_num,
                    "index_state": "pending",
                },
            )
            chunks.append(chunk.to_dict())

        if not chunks:
            return

        with self._lock:
            current_hash = self.hash_cache.get(filepath)
        if current_hash != doc_hash:
            log.info(f"Skipping stale image descriptions for {filepath}")
            return

        try:
            self.store.add_chunks(chunks)
            ready_ids = [c["chunk_id"] for c in chunks]
            ready_metas = []
            for c in chunks:
                meta = dict(c["meta"])
                meta["index_state"] = "ready"
                ready_metas.append(meta)
            self.store.update_metadatas(ready_ids, ready_metas)
            log.info(f"Indexed {len(chunks)} image description chunks for {filepath}")
            self._notify_index_changed("pdf_image_descriptions")
        except Exception as e:
            log.warning(f"Failed to index image descriptions for {filepath}: {e}")


class _WatchHandler(FileSystemEventHandler):
    def __init__(self, indexer: IncrementalIndexer):
        self.indexer = indexer
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        cfg = indexer._config.watchdog
        self._debounce = max(1, cfg.debounce_seconds)
        self._timer: Optional[threading.Timer] = None

    def on_any_event(self, event):
        if event.is_directory:
            return

        p = Path(event.src_path)
        if p.suffix.lower() not in SUPPORT_EXTS_WITH_CONFIG():
            return
        if self.indexer._should_exclude(p):
            return

        with self._lock:
            self._pending[str(p.resolve())] = time.time()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._process)
            self._timer.daemon = True
            self._timer.start()

    def _process(self):
        ready: list[str] = []
        now = time.time()
        with self._lock:
            for path, ts in list(self._pending.items()):
                if now - ts >= self._debounce:
                    ready.append(path)
                    self._pending.pop(path, None)

        for filepath in ready:
            p = Path(filepath)
            if not p.exists():
                self.indexer.store.delete_by_source(filepath)
                with self.indexer._lock:
                    self.indexer.hash_cache.pop(filepath, None)
                    self.indexer._save_manifest()
                self.indexer._notify_index_changed("watchdog_delete")
                continue
            try:
                changed, _ = self.indexer._index_file(p, rebuild=False)
                if changed:
                    self.indexer._save_manifest()
                    self.indexer._notify_index_changed("watchdog_update")
            except Exception as e:
                log.error(f"Watchdog indexing failed for {filepath}: {e}")
