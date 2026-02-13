import os
import requests
from tqdm import tqdm

url = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
cache_dir = os.path.expanduser("~/.cache/clip")
os.makedirs(cache_dir, exist_ok=True)

filepath = os.path.join(cache_dir, "ViT-B-16.pt")

print(f"Downloading CLIP ViT-B/16 model...")
print(f"URL: {url}")
print(f"Saving to: {filepath}")

# Remove partially downloaded file
if os.path.exists(filepath):
    os.remove(filepath)
    print("Removed existing partial download")

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

with open(filepath, 'wb') as file, tqdm(
    desc="Downloading",
    total=total_size,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as pbar:
    for data in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
        size = file.write(data)
        pbar.update(size)

print(f"\nDownload complete! File size: {os.path.getsize(filepath) / (1024**3):.2f} GB")
