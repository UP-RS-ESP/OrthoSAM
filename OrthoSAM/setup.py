import os
import subprocess
import sys
from pathlib import Path
import urllib.request

def main():
    base_dir = Path(__file__).resolve().parent.parent

    meta_dir = base_dir / 'OrthoSAM' / "MetaSAM"
    meta_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ]

    for url in urls:
        target = meta_dir / url.split("/")[-1]
        download_file(url, target)

    update_script = os.path.join(base_dir,'OrthoSAM', "update_config.py")
    subprocess.run([sys.executable, str(update_script)], check=True)

    print("OrthoSAM setup complete")
def download_file(url, target):
    if not target.exists():
        print(f"Downloading {url} → {target}")
        urllib.request.urlretrieve(url, target)