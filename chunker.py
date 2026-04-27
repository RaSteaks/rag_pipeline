"""Semantic chunker with markdown-aware splitting.

Strategy: Split by markdown headers first, then merge small sections,
then split oversized sections by paragraphs. Add overlap between chunks.
Truncates chunks exceeding MAX_CHUNK_CHARS (12000 chars) to avoid
Embedding API context overflow (500 errors).
"""
import re
from config import get_config


class Chunk:
    __slots__ = ("chunk_id", "source", "text", "meta")

    def __init__(self, chunk_id: str, source: str, text: str, meta: dict):
        self.chunk_id = chunk_id
        self.source = source
        self.text = text
        self.meta = meta

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "meta": self.meta,
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~1.5 char/token for Chinese, ~4 for English."""
    cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = len(text) - cn
    return int(cn / 1.5 + en / 4)


def chunk_document(parsed_doc: dict, chunk_size: int = None,
                   chunk_overlap: int = None, min_chunk: int = None) -> list[Chunk]:
    """Split a parsed document into semantic chunks."""
    config = get_config().chunking
    chunk_size = chunk_size or config.chunk_size_tokens
    chunk_overlap = chunk_overlap or config.chunk_overlap_tokens
    min_chunk = min_chunk or config.min_chunk_tokens

    text = parsed_doc.get("text", "")
    if not text:
        return []

    sections = _split_by_headers(text)
    raw_chunks = _merge_sections(sections, chunk_size, min_chunk)
    overlapped = _add_overlap(raw_chunks, chunk_overlap)
    source_name = parsed_doc.get("source_name", "")

    # Apply source weight
    weight = 1.0
    try:
        for src in get_config().knowledge_sources:
            if src.name == source_name:
                weight = src.weight
                break
    except Exception:
        pass

    # Truncate chunks exceeding max token limit (Embedding context window safety)
    MAX_CHUNK_CHARS = 12000  # ~3000 tokens safety margin for 8B embedding model

    chunks = []
    for i, text in enumerate(overlapped):
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS]
        chunks.append(Chunk(
            chunk_id=f"{parsed_doc['hash']}_{i}",
            source=parsed_doc["source"],
            text=text,
            meta={
                "hash": parsed_doc["hash"],
                "format": parsed_doc.get("format", ""),
                "position": i,
                "total_chunks": len(overlapped),
                "modified_at": parsed_doc.get("modified_at", 0),
                "title": parsed_doc.get("title", ""),
                "source_name": source_name,
                "relative_path": parsed_doc.get("relative_path", ""),
                "weight": weight,
                "content_hash": _content_hash(text),
            },
        ))
    return chunks


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by markdown headers, keeping header with content."""
    parts = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract header level if present
        match = re.match(r'^(#{1,6})\s+(.+)', part, re.MULTILINE)
        level = len(match.group(1)) if match else 0
        result.append((part, level))
    return result


def _merge_sections(sections: list[tuple[str, str]], chunk_size: int,
                    min_chunk: int) -> list[str]:
    """Merge sections into chunks respecting size limits."""
    result = []
    current = ""

    for text, level in sections:
        if estimate_tokens(current + "\n" + text) <= chunk_size:
            current = current + "\n" + text if current else text
        else:
            if current and estimate_tokens(current) >= min_chunk:
                result.append(current.strip())
            elif current:
                # Too small, try to merge with next
                pass

            if estimate_tokens(text) > chunk_size:
                # Section too large, split by paragraphs
                sub_chunks = _split_by_paragraphs(text, chunk_size, min_chunk)
                result.extend(sub_chunks)
                current = ""
            else:
                current = text

    if current and estimate_tokens(current) >= min_chunk:
        result.append(current.strip())

    return result


def _split_by_paragraphs(text: str, chunk_size: int, min_chunk: int) -> list[str]:
    """Split text by paragraphs when sections are too large."""
    paragraphs = re.split(r'\n\s*\n', text)
    result = []
    current = ""

    for para in paragraphs:
        if estimate_tokens(current + "\n\n" + para) <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current and estimate_tokens(current) >= min_chunk:
                result.append(current.strip())
            current = para

    if current and estimate_tokens(current) >= min_chunk:
        result.append(current.strip())

    return result


def _add_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Add overlapping content between consecutive chunks."""
    if not chunks or overlap_tokens <= 0:
        return chunks

    result = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Take last ~overlap_tokens from previous chunk
            prev_tail = chunks[i - 1][-overlap_tokens * 4:]
            chunk = prev_tail + "\n" + chunk
        result.append(chunk)
    return result


def _content_hash(text: str) -> str:
    """Short hash of chunk content for deduplication."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]