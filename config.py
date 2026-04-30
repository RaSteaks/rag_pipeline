"""Configuration loader with Pydantic validation."""
from pathlib import Path
import shutil
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
    skip_if_unavailable: Optional[bool] = None
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_seconds: int = 60


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
    backend: str = "local"  # "local" (llama-cpp-python) or "server" (llama.cpp :8082) or "api" (OpenRouter/OpenAI)
    endpoint: str = "http://127.0.0.1:8082"
    model_path: str = ""
    output_path: str = ""  # 新增
    dpi: int = 150
    max_pages_per_pdf: int = 50
    max_workers: int = 4
    prompt: str = ""
    # API backend config (only used when backend="api")
    # api_key: OpenRouter key (sk-or-v1-xxx) or OpenAI key (sk-xxx)
    api_key: str = ""
    # api_model: model identifier
    #   OpenRouter: openai/gpt-4o, google/gemini-2.5-flash, qwen/qwen-2.5-vl-72b-instruct
    #   OpenAI: gpt-4o, gpt-4o-mini
    api_model: str = "openai/gpt-4o"
    # api_base_url: https://openrouter.ai/api/v1 or https://api.openai.com/v1
    api_base_url: str = "https://openrouter.ai/api/v1"


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


def _config_backup_paths(config_path: Path, max_backups: int) -> list[Path]:
    if max_backups <= 0:
        return []
    paths = [config_path.with_name(f"{config_path.name}.bak")]
    paths.extend(config_path.with_name(f"{config_path.name}.bak.{i}") for i in range(1, max_backups))
    return paths


def backup_config_file(path: str | Path, max_backups: int = 5) -> Optional[Path]:
    """Create a rotating backup before changing config.yaml."""
    config_path = Path(path)
    if max_backups <= 0 or not config_path.exists():
        return None

    backup_paths = _config_backup_paths(config_path, max_backups)
    if not backup_paths:
        return None

    oldest = backup_paths[-1]
    if oldest.exists():
        oldest.unlink()

    for src, dst in reversed(list(zip(backup_paths[:-1], backup_paths[1:]))):
        if src.exists():
            src.replace(dst)

    shutil.copy2(config_path, backup_paths[0])
    return backup_paths[0]


def write_config_with_backup(path: str | Path, text: str, max_backups: int = 5) -> Optional[Path]:
    """Write config text, backing up the existing file when contents change."""
    config_path = Path(path)
    old_text = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    backup_path = None
    if old_text is not None and old_text != text:
        backup_path = backup_config_file(config_path, max_backups=max_backups)

    config_path.write_text(text, encoding="utf-8")
    return backup_path


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
