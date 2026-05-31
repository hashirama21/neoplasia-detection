"""
Download script — zip /root/outputs and upload to a file hosting service.

Usage in Modal notebook:
    !python {REPO_DIR}/scripts/download.py

Tries providers in order until one succeeds:
  1. file.io      (14-day link, 5 GB max)
  2. transfer.sh  (14-day link, 10 GB max)
  3. 0x0.st       (30-day link, 512 MB max — fallback for small zips)

For Google Drive upload, set GDRIVE_ACCESS_TOKEN:
    import os; os.environ["GDRIVE_ACCESS_TOKEN"] = "ya29.xxx"
    !python {REPO_DIR}/scripts/download.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/outputs")
ZIP_PATH   = "/tmp/rare26_outputs.zip"


def zip_outputs() -> int:
    print(f"Zipping {OUTPUT_DIR} → {ZIP_PATH} ...")
    result = subprocess.run(
        ["zip", "-r", ZIP_PATH, OUTPUT_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("zip error:", result.stderr)
        sys.exit(1)
    size_mb = Path(ZIP_PATH).stat().st_size / 1e6
    print(f"Done — {size_mb:.1f} MB")
    return result.returncode


def upload_fileio() -> str | None:
    print("\nTrying file.io ...")
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{ZIP_PATH}", "https://file.io/?expires=14d"],
        capture_output=True, text=True, timeout=300
    )
    try:
        data = json.loads(result.stdout)
        if data.get("success"):
            return data["link"]
    except Exception:
        pass
    print("  file.io failed:", result.stdout[:200])
    return None


def upload_transfersh() -> str | None:
    print("\nTrying transfer.sh ...")
    result = subprocess.run(
        ["curl", "-s", "--upload-file", ZIP_PATH,
         "https://transfer.sh/rare26_outputs.zip"],
        capture_output=True, text=True, timeout=300
    )
    link = result.stdout.strip()
    if link.startswith("https://"):
        return link
    print("  transfer.sh failed:", result.stdout[:200])
    return None


def upload_0x0() -> str | None:
    print("\nTrying 0x0.st (512 MB limit) ...")
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{ZIP_PATH}", "https://0x0.st"],
        capture_output=True, text=True, timeout=300
    )
    link = result.stdout.strip()
    if link.startswith("https://"):
        return link
    print("  0x0.st failed:", result.stdout[:200])
    return None


def upload_gdrive(token: str) -> str | None:
    print("\nUploading to Google Drive ...")
    try:
        import requests
        with open(ZIP_PATH, "rb") as f:
            r = requests.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                headers={"Authorization": f"Bearer {token}"},
                files={
                    "metadata": (
                        None,
                        json.dumps({"name": "rare26_outputs.zip"}),
                        "application/json",
                    ),
                    "file": ("rare26_outputs.zip", f, "application/zip"),
                },
                timeout=600,
            )
        data = r.json()
        file_id = data.get("id")
        if file_id:
            return f"https://drive.google.com/file/d/{file_id}"
        print("  Drive error:", data)
    except Exception as e:
        print("  Drive exception:", e)
    return None


def main():
    zip_outputs()

    link = None

    # Google Drive first if token provided
    token = os.environ.get("GDRIVE_ACCESS_TOKEN")
    if token:
        link = upload_gdrive(token)

    # Public hosters fallback
    if not link:
        link = upload_fileio()
    if not link:
        link = upload_transfersh()
    if not link:
        link = upload_0x0()

    print("\n" + "=" * 50)
    if link:
        print(f"Download link:\n{link}")
        print("\nIn Colab:")
        print(f'  !wget -O outputs.zip "{link}"')
        print( '  !unzip outputs.zip -d /content/outputs/')
    else:
        print("All upload methods failed.")
        print(f"The zip is at {ZIP_PATH} — download it manually.")
    print("=" * 50)


if __name__ == "__main__":
    main()