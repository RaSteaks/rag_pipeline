"""Configuration loader with Pydantic validation."""
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, validator


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8900


class EmbeddingConfig(BaseModel):
    endpoint: str = ""
    model: str = ""
    timeout_seconds: int = 60
    batch_size: int = 32


class RerankerConfig(BaseModel):
    enabled: bool = True
    endpoint: str = ""
    model: str = ""
    timeout_seconds: int = 20
    max_candidates: int = 20
    fallback_to_rrf: bool = True


class IndexesConfig(BaseModel):
    chroma_path: str = ""
    bm25_path: str = ""
    manifest_path: str = ""


class RetrievalConfig(BaseModel):
    vector_top_k: int = 30
    bm25_top_k: int = 30
    rrf_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    max_chunks_per_doc: int = 2


class ChunkingConfig(BaseModel):
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    min_chunk_tokens: int = 80


class KnowledgeSource(BaseModel):
    name: str
    path: str
    enabled: bool = True
    recursive: bool = True
    weight: float = 1.0
    file_types: List[str] = [".md", ".pdf", ".docx", ".html", ".txt"]

    @validator("file_types")
    def file_types_must_start_with_dot(cls, v):
        return [ft if ft.startswith(".") else f".{ft}" for ft in v]


class ExcludeConfig(BaseModel):
    dirs: List[str] = []
    files: List[str] = []


class WatchdogConfig(BaseModel):
    enabled: bool = True
    debounce_seconds: int = 2
    batch_interval_seconds: int = 10


class ImageDescriptionConfig(BaseModel):
    enabled: bool = False
    backend: str = "local"  # "local" or "server"
    endpoint: str = "http://127.0.0.1:8082"
    model_path: str = ""
    dpi: int = 150
    max_pages_per_pdf: int = 50
    prompt: str = ""


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    indexes: IndexesConfig = IndexesConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    knowledge_sources: List[KnowledgeSource] = []
    exclude: ExcludeConfig = ExcludeConfig()
    watchdog: WatchdogConfig = WatchdogConfig()
    image_description: ImageDescriptionConfig = ImageDescriptionConfig()


def load_config(path: str = "") -> AppConfig:
    """Load and validate config from YAML file."""
    if not path:
        # Look for config.yaml next to this script
        path = str(Path(__file__).parent / "config.yaml")
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = AppConfig(**data)

    # Validate source paths
    for source in config.knowledge_sources:
        if source.enabled and not Path(source.path).exists():
            print(f"Knowledge source path does not exist: {source.path}")

    return config


_config: Optional[AppConfig] = None


def get_config(path: str = "") -> AppConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = load_config(path)
    return _config


def reload_config(path: str = "") -> AppConfig:
    """Force reload config from file."""
    global _config
    _config = load_config(path)
    return _config