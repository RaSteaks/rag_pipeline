import os
import time
import hashlib
import threading
import json
from pathlib import Path
from typing import Optional, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from logger import setup_logger
from config import get_config
from parsers import parse_file, SUPPORT_EXTS_WITH_CONFIG
from chunker import chunk_document, Chunk
from vector_store import VectorStore

log = setup_logger("rag")

class IncrementalIndexer:
    def __init__(self):
        self._config = get_config()
        self.store = VectorStore(self._config)
        self.manifest_path = Path("index_manifest.json")
        self.hash_cache = self._load_manifest()
        self._lock = threading.Lock()
        self._observer: Optional[Observer] = None

    def _load_manifest(self) -> Dict[str, str]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Failed to load manifest: {e}")
        return {}

    def _save_manifest(self):
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.hash_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Failed to save manifest: {e}")

    def _should_exclude(self, path: Path) -> bool:
        path_str = str(path).replace('\\', '/')
        # Check directories
        for exc_dir in self._config.exclude.directories:
            if f"/{exc_dir}/" in f"/{path_str}/":
                return True
        # Check patterns
        for pattern in self._config.exclude.patterns:
            if path.match(pattern):
                return True
        return False

    def sync(self, source_name: Optional[str] = None, rebuild: bool = False):
        """Sync knowledge sources to vector store."""
        sources = self._config.sources
        if source_name:
            if source_name not in sources:
                log.error(f"Source {source_name} not found in config")
                return
            sources_to_process = {source_name: sources[source_name]}
        else:
            sources_to_process = sources

        for name, root_path in sources_to_process.items():
            log.info(f"Syncing source: {name} ({root_path})")
            root = Path(root_path)
            if not root.exists():
                log.warning(f"Source path {root_path} does not exist, skipping")
                continue

            # Identify valid files
            exts = SUPPORT_EXTS_WITH_CONFIG()
            files = []
            for ext in exts:
                files.extend(root.rglob(f"*{ext}"))

            for filepath in files:
                if self._should_exclude(filepath):
                    continue
                
                path_str = str(filepath)
                
                # Parsing
                parsed = parse_file(path_str)
                if parsed is None:
                    continue

                # Check hash
                old_hash = self.hash_cache.get(path_str)
                if not rebuild and old_hash == parsed.hash:
                    continue

                log.info(f"Indexing: {path_str} (rebuild={rebuild})")
                
                # Delete existing if updating
                if old_hash or rebuild:
                    self.store.delete_by_source(path_str)

                try:
                    # Chunks - use object's to_dict() for chunker
                    chunks = chunk_document(parsed.to_dict())
                    self.store.add_chunks([c.to_dict() for c in chunks])
                    
                    # Update cache
                    with self._lock:
                        self.hash_cache[path_str] = parsed.hash
                        self._save_manifest()
                    
                    log.info(f"Indexed {len(chunks)} chunks from {path_str}")

                    # Run image description if PDF
                    if path_str.lower().endswith(".pdf"):
                        self._describe_pdf_images(parsed, path_str)

                except Exception as e:
                    log.error(f"Failed to index {path_str}: {e}")

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

        # Use object properties for safety
        doc_hash = getattr(parsed_doc, 'hash', '')
        source = getattr(parsed_doc, 'source', filepath)
        doc_format = getattr(parsed_doc, 'format', 'pdf')
        total_chunks = getattr(parsed_doc, 'total_chunks', 0)
        modified_at = getattr(parsed_doc, 'modified_at', 0)
        title = getattr(parsed_doc, 'title', '')
        source_name = getattr(parsed_doc, 'source_name', '')
        relative_path = getattr(parsed_doc, 'relative_path', '')
        weight = getattr(parsed_doc, 'weight', 1.0)

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
                    "total_chunks": total_chunks,
                    "modified_at": modified_at,
                    "title": title,
                    "source_name": source_name,
                    "relative_path": relative_path,
                    "weight": weight,
                    "content_hash": hashlib.sha256(desc.description.encode()).hexdigest()[:12],
                    "is_image_description": True,
                    "image_page": desc.page_num,
                },
            )
            try:
                self.store.add_chunks([chunk.to_dict()])
                log.info(f"Added image description for page {desc.page_num}")
            except Exception as e:
                log.warning(f"Failed to index image description for page {desc.page_num}: {e}")

    def start_watchdog(self):
        if self._observer:
            return
        
        self._observer = Observer()
        handler = _WatchHandler(self)
        
        for name, root_path in self._config.sources.items():
            if Path(root_path).exists():
                self._observer.schedule(handler, root_path, recursive=True)
                log.info(f"Watching source: {name} ({root_path})")
            
        self._observer.start()

    def stop_watchdog(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

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
        exts = SUPPORT_EXTS_WITH_CONFIG()
        if p.suffix.lower() not in exts:
            return
        if self.indexer._should_exclude(p):
            return
        with self.indexer._lock:
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
                self.indexer.store.delete_by_source(filepath)
                with self.indexer._lock:
                    self.indexer.hash_cache.pop(filepath, None)
                    self.indexer._save_manifest()
                continue

            parsed = parse_file(filepath)
            if parsed is None:
                continue

            with self.indexer._lock:
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
                    if filepath.lower().endswith(".pdf"):
                        self.indexer._describe_pdf_images(parsed, filepath)
                except Exception as e:
                    log.error(f"Watchdog indexing failed for {filepath}: {e}")
