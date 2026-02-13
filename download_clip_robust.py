import os
import requests
from tqdm import tqdm
import time

url = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
cache_dir = os.path.expanduser("~/.cache/clip")
os.makedirs(cache_dir, exist_ok=True)

filepath = os.path.join(cache_dir, "ViT-B-16.pt")
temp_filepath = filepath + ".tmp"

print(f"Downloading CLIP ViT-B/16 model...")
print(f"Saving to: {filepath}")

# Check if already downloaded
if os.path.exists(filepath):
    file_size = os.path.getsize(filepath)
    if file_size > 300 * 1024 * 1024:  # > 300MB means likely complete
        print(f"Model already downloaded! Size: {file_size / (1024**3):.2f} GB")
        exit(0)
    else:
        print(f"Found incomplete download ({file_size / (1024**2):.1f} MB), removing...")
        os.remove(filepath)

max_retries = 5
for attempt in range(max_retries):
    try:
        print(f"\nAttempt {attempt + 1}/{max_retries}")
        
        # Use smaller timeout and resume capability
        headers = {}
        resume_pos = 0
        
        if os.path.exists(temp_filepath):
            resume_pos = os.path.getsize(temp_filepath)
            headers['Range'] = f'bytes={resume_pos}-'
            print(f"Resuming from {resume_pos / (1024**2):.1f} MB")
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code in [200, 206]:  # 206 is partial content
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
            
            mode = 'ab' if resume_pos > 0 else 'wb'
            
            with open(temp_filepath, mode) as file, tqdm(
                desc="Downloading",
                initial=resume_pos,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=512*1024):  # 512KB chunks
                    if data:
                        size = file.write(data)
                        pbar.update(size)
            
            # Verify download
            final_size = os.path.getsize(temp_filepath)
            print(f"\nDownload complete! Size: {final_size / (1024**3):.2f} GB")
            
            # Rename temp to final
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            
            print("✓ CLIP model ready!")
            exit(0)
        else:
            print(f"Server returned status code: {response.status_code}")
            
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("\nFailed to download after all retries.")
            print("Please check your internet connection and try again.")
            exit(1)
