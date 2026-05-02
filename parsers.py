"""Multi-format document parser.

Supports: .md, .pdf (PyMuPDF), .docx (python-docx), .html (BeautifulSoup),
.ipynb (Jupyter notebook), .txt, .py, .json, .yaml

Each parsed document includes source_name and relative_path from config.yaml
knowledge_sources, enabling per-source filtering and rebuilding.

PDFs with unsupported annotations (e.g., AGFA_BleedArea) are parsed with
content intact; only the annotation parsing is skipped.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from config import get_config
from logger import setup_logger
log = setup_logger("rag")

SUPPORTED_EXTS = {".md", ".pdf", ".docx", ".doc", ".html", ".htm", ".txt", ".py", ".ipynb"}


@contextmanager
def _suppress_mupdf_stderr():
    """Hide non-fatal MuPDF diagnostics printed directly to stderr."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


class ParsedDoc:
    __slots__ = ("source", "title", "text", "hash", "modified_at", "format",
                 "source_name", "relative_path")

    def __init__(self, source: str, title: str, text: str, hash: str,
                 modified_at: float, format: str, source_name: str = "",
                 relative_path: str = ""):
        self.source = source
        self.title = title
        self.text = text
        self.hash = hash
        self.modified_at = modified_at
        self.format = format
        self.source_name = source_name
        self.relative_path = relative_path

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


def _get_source_info(filepath: str) -> tuple[str, str]:
    """Match filepath to knowledge source name and compute relative path."""
    config = get_config()
    fp = Path(filepath).resolve()
    for source in config.knowledge_sources:
        sp = Path(source.path).resolve()
        try:
            rel = fp.relative_to(sp)
            return source.name, str(rel)
        except ValueError:
            continue
    return "", fp.name


def parse_file(filepath: str) -> Optional[ParsedDoc]:
    """Parse a file and extract text content."""
    p = Path(filepath)
    if not p.exists():
        return None

    suffix = p.suffix.lower()
    all_exts = SUPPORT_EXTS_WITH_CONFIG()
    if suffix not in all_exts:
        return None

    try:
        file_hash = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        stat = p.stat()
        source_name, relative_path = _get_source_info(filepath)

        text = _extract_text(p, suffix)
        if not text or not text.strip():
            return None

        return ParsedDoc(
            source=str(p.absolute()),
            title=p.stem,
            text=text.strip(),
            hash=file_hash,
            modified_at=stat.st_mtime,
            format=suffix.lstrip("."),
            source_name=source_name,
            relative_path=relative_path,
        )
    except Exception as e:
        print(f"Failed to parse {filepath}: {e}")
        return None


def SUPPORT_EXTS_WITH_CONFIG() -> set:
    """Merge default and config-specified file extensions."""
    exts = set(SUPPORTED_EXTS)
    try:
        for source_cfg in get_config().knowledge_sources:
            exts |= set(source_cfg.file_types)
    except Exception:
        pass
    return exts


def _extract_text(p: Path, suffix: str) -> Optional[str]:
    if suffix in (".md", ".txt", ".py"):
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(p)
    if suffix in (".docx", ".doc"):
        return _extract_docx(p)
    if suffix in (".html", ".htm"):
        return _extract_html(p)
    if suffix == ".ipynb":
        return _extract_ipynb(p)
    if suffix in (".json", ".yaml", ".yml"):
        return p.read_text(encoding="utf-8", errors="replace")
    return None


def _extract_pdf(p: Path) -> Optional[str]:
    try:
        import fitz
        with _suppress_mupdf_stderr():
            doc = fitz.open(str(p))
            pages = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    pages.append(f"[Page {page_num + 1}]\n{page_text}")
            doc.close()
        return "\n\n".join(pages) if pages else None
    except ImportError:
        print(f"PyMuPDF not installed, cannot parse PDF: {p}")
        return None


def _extract_docx(p: Path) -> Optional[str]:
    try:
        from docx import Document
        doc = Document(str(p))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs) if paragraphs else None
    except ImportError:
        print(f"python-docx not installed, cannot parse DOCX: {p}")
        return None


def _extract_html(p: Path) -> Optional[str]:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(p.read_text("utf-8", errors="replace"), "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        return p.read_text("utf-8", errors="replace")


def _extract_ipynb(p: Path) -> Optional[str]:
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
        cells = []
        for cell in nb.get("cells", []):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            if source.strip():
                cells.append(f"[{cell_type}]\n{source}")
        return "\n\n".join(cells) if cells else None
    except Exception as e:
        print(f"Failed to parse notebook {p}: {e}")
        return None
