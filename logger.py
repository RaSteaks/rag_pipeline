"""Simple logging system for RAG Pipeline.

Logs to both console and file (D:/Openclaw/rag_pipeline/logs/).
Log files rotate daily.
"""
import logging
import logging.handlers
from pathlib import Path
from config import get_config


def setup_logger(name: str = "rag") -> logging.Logger:
    """Setup and return a logger with console + file handlers."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on reload
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - DEBUG level, daily rotation
    try:
        config = get_config()
        log_dir = Path(config.indexes.chroma_path).parent / "rag_pipeline" / "logs"
    except Exception:
        log_dir = Path(__file__).parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_dir / "rag.log"),
        when="midnight",
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger