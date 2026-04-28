import requests, json, base64
from pathlib import Path

imgs = list(Path("D:/Openclaw/rag_pipeline/test_images").glob("*.png"))
if not imgs:
    print("No images found")
    exit()

img_path = str(imgs[0])
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
print(f"Image size: {len(img_b64)} chars base64")

resp = requests.post("http://127.0.0.1:8082/v1/chat/completions", json={
    "model": "InternVL2.5-4B",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is shown in this image? Describe in Chinese."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }],
    "max_tokens": 200
}, timeout=120)
print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Error: {resp.text[:500]}")
else:
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    print(f"Response: {content[:500]}")