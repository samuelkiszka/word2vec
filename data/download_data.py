"""
download_data.py
================
Downloads and unpacks the text8 corpus (~100 MB, ~17M tokens of cleaned Wikipedia).
Creates smaller subsets (10k, 50k, 100k, 1M tokens) for quick testing.

Run:  python download_data.py
"""

import urllib.request, zipfile, pathlib

URL      = "http://mattmahoney.net/dc/text8.zip"
OUT_DIR  = pathlib.Path(__file__).parent
ZIP_PATH = OUT_DIR / "text8.zip"
TXT_PATH = OUT_DIR / "tokens" / "text8.txt"

smaller_datasets = {
    "ten": 10_000,
    "fifty": 50_000,
    "hundred": 100_000,
    "thousand": 1_000_000,
}

def create_smaller_datasets():
    with open(TXT_PATH, "r") as f:
        text = f.read()
    tokens = text.split()

    for name, size in smaller_datasets.items():
        out_path = OUT_DIR / "tokens" / f"{name}.txt"
        if out_path.exists():
            print(f"Already exists: {out_path}")
            continue

        with open(out_path, "w") as f:
            f.write(" ".join(tokens[:size]))
        print(f"Created {out_path} with {size} tokens")


def download():
    if TXT_PATH.exists():
        print(f"Already downloaded: {TXT_PATH}")
        return

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

    print("Unpacking ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(OUT_DIR / "tokens")
        extracted = OUT_DIR / "tokens" / "text8"
        extracted.rename(TXT_PATH)

    create_smaller_datasets()

    ZIP_PATH.unlink()
    print(f"Done → {TXT_PATH}  ({TXT_PATH.stat().st_size // 1_000_000} MB)")


if __name__ == "__main__":
    download()
