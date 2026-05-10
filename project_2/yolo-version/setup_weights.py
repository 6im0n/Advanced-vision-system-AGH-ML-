"""Download SiamFC AlexNet pretrained weights.

Tries several mirrors. If all fail, instructions printed for manual download.
Target: weights/siamfc_alexnet_e50.pth (huanglianghua/siamfc-pytorch).
"""
from pathlib import Path
import sys
import gdown

WEIGHTS_DIR = Path(__file__).parent / "weights"
OUT = WEIGHTS_DIR / "siamfc_alexnet_e50.pth"

# Mirrors (folder + direct file). Order: try cheapest first.
SOURCES = [
    # Original huanglianghua/siamfc-pytorch shared folder
    ("folder", "https://drive.google.com/drive/folders/14LYwAW28SVkW1B5mJDIUW5DpPSgDC9eI"),
    # Mirror file IDs (community uploads — may rotate)
    ("file", "https://drive.google.com/uc?id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4"),
]


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    if OUT.exists():
        print(f"[ok] already present: {OUT}")
        return

    for kind, url in SOURCES:
        try:
            print(f"[try] {kind}: {url}")
            if kind == "folder":
                gdown.download_folder(url, output=str(WEIGHTS_DIR), quiet=False, use_cookies=False)
            else:
                gdown.download(url, str(OUT), quiet=False)
            if OUT.exists():
                print(f"[ok] saved: {OUT}")
                return
        except Exception as e:
            print(f"[fail] {e}")

    print("\n[manual] Auto-download failed. Get the weights yourself and place at:")
    print(f"   {OUT}")
    print("\nSources:")
    print(" - https://github.com/huanglianghua/siamfc-pytorch  (README → Google Drive)")
    print(" - https://github.com/got-10k/toolkit (SiamFC example weights)")
    print(" - Search 'siamfc_alexnet_e50.pth' on GitHub releases")
    sys.exit(1)


if __name__ == "__main__":
    main()
