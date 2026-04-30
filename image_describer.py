"""Image describer using vision models.

All configurations are loaded from config.yaml via config.py.
No hardcoded paths or API keys should exist here.
"""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass

from logger import setup_logger
from config import get_config

log = setup_logger("rag")

@dataclass
class ImageDescription:
    page_num: int
    description: str
    image_path: str
    source: str


def render_pdf_pages(pdf_path: str, output_dir: str,
                     dpi: int = 150, max_pages: int = 50) -> list[str]:
    """Render PDF pages as PNG images using PyMuPDF."""
    import fitz

    doc = fitz.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = str(out / f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths


def describe_image_llamacpp(image_path: str, cfg) -> str:
    """Describe image using llama.cpp server."""
    import requests
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
        payload = {
            "model": "vision-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        resp = requests.post(f"{cfg.endpoint}/v1/chat/completions", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"llama.cpp vision failed: {e}")
        return ""


def describe_image_local(image_path: str, cfg) -> str:
    """Describe image using llama-cpp-python (local mode)."""
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=cfg.model_path, n_ctx=2048, n_gpu_layers=0, verbose=False)
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": cfg.prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]}],
            max_tokens=512,
            temperature=0.3,
        )
        description = response["choices"][0]["message"]["content"]
        del llm
        import gc
        gc.collect()
        return description
    except Exception as e:
        log.warning(f"Local vision failed: {e}")
        return ""


def describe_image_api(image_path: str, cfg) -> str:
    """Describe image using OpenAI-compatible API."""
    import requests
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        suffix = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
        payload = {
            "model": cfg.api_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": cfg.prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]}],
            "max_tokens": 512,
            "temperature": 0.3,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
        resp = requests.post(f"{cfg.api_base_url}/chat/completions", json=payload, headers=headers, timeout=180)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"Vision API failed: {e}")
        return ""


def _describe_single_page(img_path: str, page_num: int, cfg, pdf_path: str) -> ImageDescription | None:
    """Describe one rendered PDF page."""
    if cfg.backend == "api":
        desc = describe_image_api(img_path, cfg)
    elif cfg.backend == "local":
        desc = describe_image_local(img_path, cfg)
    else:
        desc = describe_image_llamacpp(img_path, cfg)

    if not desc:
        return None
    return ImageDescription(
        page_num=page_num,
        description=desc,
        image_path=img_path,
        source=pdf_path,
    )


def describe_pdf_images(pdf_path: str) -> list[ImageDescription]:
    """Full pipeline: render PDF pages -> describe each image.
    All params (except the file to process) loaded from config.
    """
    cfg = get_config().image_description
    if not cfg.enabled:
        return []

    pdf_name = Path(pdf_path).stem
    
    # 路径完全从配置中获取
    output_root = Path(cfg.output_path)
    render_dir = str(output_root / f"{pdf_name}_images")

    # Render pages
    image_paths = render_pdf_pages(pdf_path, render_dir, cfg.dpi, cfg.max_pages_per_pdf)
    if not image_paths:
        return []

    log.info(f"Rendered {len(image_paths)} pages from {pdf_name}")

    descriptions = []
    max_workers = max(1, int(getattr(cfg, "max_workers", 4) or 4))
    max_workers = min(max_workers, len(image_paths))
    log.info(
        f"Describing {len(image_paths)} pages from {pdf_name} "
        f"with {max_workers} workers [backend: {cfg.backend}]"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_describe_single_page, img_path, i + 1, cfg, pdf_path): i + 1
            for i, img_path in enumerate(image_paths)
        }

        for future in as_completed(futures):
            page_num = futures[future]
            try:
                item = future.result()
            except Exception as e:
                log.warning(f"Page {page_num}: image description failed: {e}")
                continue

            if item:
                descriptions.append(item)
                log.debug(f"Page {page_num}: {item.description[:100]}...")
            else:
                log.warning(f"Page {page_num}: description empty, skipping")

    descriptions.sort(key=lambda item: item.page_num)
    log.info(f"Generated {len(descriptions)} image descriptions for {pdf_name}")
    return descriptions
