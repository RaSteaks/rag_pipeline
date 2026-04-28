"""Image describer using InternVL2.5-4B vision model.

Renders PDF pages as images, sends to vision model for description.
Supports two backends:
  - llama.cpp server (recommended, same architecture as Embedding/Reranker)
  - Direct llama-cpp-python (fallback, no separate server needed)

The vision model is NOT a long-running service. It is loaded on-demand
during indexing and released after descriptions are generated.
"""
import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from logger import setup_logger

log = setup_logger("rag")

# Default prompt for image description
DEFAULT_PROMPT = "请详细描述这张图片的内容，包括图表标题、坐标轴标签、数据趋势、关键数值和任何文字标注。如果图片不是图表而是示意图或照片，请描述其主要视觉元素和含义。"


@dataclass
class ImageDescription:
    page_num: int          # 1-indexed page number
    description: str       # Generated description text
    image_path: str        # Path to rendered page image
    source: str            # Source PDF path


def render_pdf_pages(pdf_path: str, output_dir: str, 
                     dpi: int = 150, 
                     max_pages: int = 50) -> list[str]:
    """Render PDF pages as PNG images using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save rendered images
        dpi: Resolution (150 is good balance of quality vs size)
        max_pages: Maximum pages to render (safety limit)
    
    Returns:
        List of image file paths
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        # Render at specified DPI
        zoom = dpi / 72  # 72 is default PDF DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        image_path = str(out / f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    
    doc.close()
    return image_paths


def describe_image_llamacpp(image_path: str, 
                            endpoint: str = "http://127.0.0.1:8082",
                            prompt: str = DEFAULT_PROMPT,
                            timeout: int = 60) -> str:
    """Describe an image using llama.cpp server with vision support.
    
    llama.cpp /v1/chat/completions with image_url content type.
    """
    import requests
    
    # Read and base64 encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Determine mime type
    suffix = Path(image_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
    
    payload = {
        "model": "InternVL2.5-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    
    try:
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        log.warning("Vision model server not running, skipping image description")
        return ""
    except Exception as e:
        log.warning(f"Image description failed: {e}")
        return ""


def describe_image_local(image_path: str,
                         model_path: str = "D:\\models\\InternVL2_5-4B.Q5_K_M.gguf",
                         prompt: str = DEFAULT_PROMPT) -> str:
    """Describe an image using llama-cpp-python (no server needed).
    
    Loads model, generates description, releases memory.
    This is the recommended approach for on-demand indexing.
    """
    try:
        from llama_cpp import Llama
                
        # Load model with vision support
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
        
        # Read image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        suffix = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
        
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.3,
        )
        
        description = response["choices"][0]["message"]["content"]
        
        # Release model memory
        del llm
        import gc
        gc.collect()
        
        return description
        
    except ImportError:
        log.warning("llama-cpp-python not installed, cannot describe images locally")
        return ""
    except Exception as e:
        log.warning(f"Local image description failed: {e}")
        return ""


def describe_pdf_images(pdf_path: str, 
                        output_dir: str,
                        endpoint: str = "http://127.0.0.1:8082",
                        use_local: bool = False,
                        model_path: str = "D:\\models\\InternVL2_5-4B.Q5_K_M.gguf",
                        dpi: int = 150,
                        max_pages: int = 50) -> list[ImageDescription]:
    """Full pipeline: render PDF pages → describe each image.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for rendered images and temp files
        endpoint: llama.cpp vision server endpoint (if use_local=False)
        use_local: Use llama-cpp-python instead of server
        model_path: Path to GGUF model (for local mode)
        dpi: PDF render resolution
        max_pages: Safety limit on pages to process
    
    Returns:
        List of ImageDescription objects
    """
    pdf_name = Path(pdf_path).stem
    render_dir = str(Path(output_dir) / f"{pdf_name}_images")
    
    # Render pages
    image_paths = render_pdf_pages(pdf_path, render_dir, dpi, max_pages)
    if not image_paths:
        return []
    
    log.info(f"Rendered {len(image_paths)} pages from {pdf_name}")
    
    # Describe each page
    descriptions = []
    for i, img_path in enumerate(image_paths):
        page_num = i + 1
        log.info(f"Describing page {page_num}/{len(image_paths)} of {pdf_name}")
        
        if use_local:
            desc = describe_image_local(img_path, model_path=model_path)
        else:
            desc = describe_image_llamacpp(img_path, endpoint=endpoint)
        
        if desc:
            descriptions.append(ImageDescription(
                page_num=page_num,
                description=desc,
                image_path=img_path,
                source=pdf_path,
            ))
            log.debug(f"Page {page_num}: {desc[:100]}...")
        else:
            log.warning(f"Page {page_num}: description empty, skipping")
    
    log.info(f"Generated {len(descriptions)} image descriptions for {pdf_name}")
    return descriptions