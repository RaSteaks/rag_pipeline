"""Incremental indexer with SHA256 manifest and watchdog file monitoring.

Supports three modes:
  - full_sync(): Detect additions, modifications, deletions via SHA256 hash comparison
  - rebuild_source(name): Rebuild index for a single knowledge source
  - full_rebuild(): Delete all indexes and rebuild from scratch

Config-driven: knowledge sources, exclusion rules, and file types all come from config.yaml.
The manifest (index_manifest.json) persists file hashes for incremental updates.
"""
import json
import threading
import time
from pathlib import Path
from typing import Optional
from fnmatch import fnmatch

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import get_config, reload_config
from logger import setup_logger
log = setup_logger("rag")
from parsers import parse_file, SUPPORT_EXTS_WITH_CONFIG
from chunker import chunk_document
from vector_store import RAGVectorStore


class IncrementalIndexer:
    def __init__(self, store: RAGVectorStore, config_override=None):
        self._config = config_override or get_config()
        self.store = store
        self.hash_cache: dict = {}
        self._lock = threading.Lock()
        self._load_manifest()

    @property
    def manifest_path(self) -> Path:
        return Path(self._config.indexes.manifest_path)

    def _load_manifest(self):
        """Load index manifest from disk."""
        if self.manifest_path.exists():
            try:
                self.hash_cache = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.info("Corrupt manifest, starting fresh")
                self.hash_cache = {}
        else:
            self.hash_cache = {}

    def _save_manifest(self):
        """Persist manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.hash_cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches exclusion rules."""
        exclude = self._config.exclude

        for dir_pattern in exclude.dirs:
            for part in path.parts:
                if fnmatch(part, dir_pattern):
                    return True

        for file_pattern in exclude.files:
            if fnmatch(path.name, file_pattern):
                return True

        return False

    def _collect_files(self) -> set[str]:
        """Collect all files from enabled knowledge sources."""
        current_files = set()
        for source in self._config.knowledge_sources:
            if not source.enabled:
                continue
            sp = Path(source.path)
            if not sp.exists():
                continue
            glob_fn = sp.rglob if source.recursive else sp.glob
            for p in glob_fn("*"):
                if p.is_file() and not self._should_exclude(p):
                    if p.suffix.lower() in set(source.file_types):
                        current_files.add(str(p.absolute()))
        return current_files

    def full_sync(self) -> dict:
        """Full sync: detect additions, modifications, deletions."""
        changed = 0
        deleted = 0
        errors = 0

        current_files = self._collect_files()

        # Detect deletions
        cached_paths = set(self.hash_cache.keys())
        for removed in cached_paths - current_files:
            self.store.delete_by_source(removed)
            del self.hash_cache[removed]
            deleted += 1

        # Detect additions/modifications
        for filepath in current_files:
            parsed = parse_file(filepath)
            if parsed is None:
                continue

            old_hash = self.hash_cache.get(filepath)
            new_hash = parsed.hash

            if old_hash == new_hash:
                continue  # Unchanged

            if old_hash is not None:
                self.store.delete_by_source(filepath)

            try:
                chunks = chunk_document(parsed.to_dict())
                self.store.add_chunks([c.to_dict() for c in chunks])
                self.hash_cache[filepath] = new_hash
                changed += 1
            except Exception as e:
                print(f"Failed to index {filepath}: {e}")
                errors += 1

        self._save_manifest()
        total = len(current_files)
        chunks = self.store.count()
        return {"changed": changed, "deleted": deleted, "errors": errors,
                "total_files": total, "total_chunks": chunks}

    def rebuild_source(self, source_name: str) -> dict:
        """Rebuild index for a specific knowledge source."""
        # Find source config
        source_cfg = None
        for s in self._config.knowledge_sources:
            if s.name == source_name:
                source_cfg = s
                break
        if source_cfg is None:
            return {"error": f"Source '{source_name}' not found"}

        # Delete existing chunks for this source
        self.store.delete_by_source_name(source_name)

        # Re-index
        changed = 0
        errors = 0
        sp = Path(source_cfg.path)
        if not sp.exists():
            return {"error": f"Source path does not exist: {source_cfg.path}"}

        glob_fn = sp.rglob if source_cfg.recursive else sp.glob
        for p in glob_fn("*"):
            if p.is_file() and not self._should_exclude(p):
                if p.suffix.lower() in set(source_cfg.file_types):
                    filepath = str(p.absolute())
                    parsed = parse_file(filepath)
                    if parsed is None:
                        continue
                    try:
                        chunks = chunk_document(parsed.to_dict())
                        self.store.add_chunks([c.to_dict() for c in chunks])
                        self.hash_cache[filepath] = parsed.hash
                        changed += 1
                    except Exception as e:
                        print(f"Failed to index {filepath}: {e}")
                        errors += 1

        self._save_manifest()
        return {"source": source_name, "changed": changed, "errors": errors}

    def full_rebuild(self) -> dict:
        """Delete all indexes and rebuild from scratch."""
        self.hash_cache = {}
        self._save_manifest()

        # Reset ChromaDB collection
        self.store.client.delete_collection("knowledge_base")
        self.store.collection = self.store.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32, "hnsw:construction_ef": 200}
        )

        return self.full_sync()

    def start_watcher(self) -> Optional[Observer]:
        """Start file watcher for incremental updates."""
        if not self._config.watchdog.enabled:
            log.info("Watchdog disabled in config")
            return None

        handler = _WatchHandler(self)
        observer = Observer()

        for source in self._config.knowledge_sources:
            if not source.enabled:
                continue
            sp = Path(source.path)
            if sp.exists():
                observer.schedule(handler, str(sp), recursive=source.recursive)

        observer.daemon = True
        observer.start()
        log.info("File watcher started")
        return observer


class _WatchHandler(FileSystemEventHandler):
    def __init__(self, indexer: IncrementalIndexer):
        self.indexer = indexer
        self._pending: dict[str, float] = {}
        cfg = indexer._config.watchdog
        self._debounce = cfg.debounce_seconds

    def on_any_event(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)

        # Check extension
        exts = SUPPORT_EXTS_WITH_CONFIG()
        if p.suffix.lower() not in exts:
            return

        # Check exclusion
        if self.indexer._should_exclude(p):
            return

        self._pending[event.src_path] = time.time()
        threading.Timer(self._debounce, self._process).start()

    def _process(self):
        now = time.time()
        with self.indexer._lock:
            ready = [p for p, t in self._pending.items() if now - t >= self._debounce]
            for p in ready:
                del self._pending[p]

        for filepath in ready:
            p = Path(filepath)
            if not p.exists():
                # File was deleted
                self.indexer.store.delete_by_source(filepath)
                self.indexer.hash_cache.pop(filepath, None)
                self.indexer._save_manifest()
                continue

            parsed = parse_file(filepath)
            if parsed is None:
                continue

            old_hash = self.indexer.hash_cache.get(filepath)
            if old_hash == parsed.hash:
                continue

            if old_hash:
                self.indexer.store.delete_by_source(filepath)

            try:
                chunks = chunk_document(parsed.to_dict())
                self.indexer.store.add_chunks([c.to_dict() for c in chunks])
                self.indexer.hash_cache[filepath] = parsed.hash
                self.indexer._save_manifest()
            except Exception as e:
                print(f"Watchdog indexing failed for {filepath}: {e}")