"""
Download a ZWM checkpoint from HuggingFace Hub into ./out/

Usage:
    python scripts/hf_model_download.py awwkl/zwm-bvd-170m
    python scripts/hf_model_download.py awwkl/zwm-bvd-170m --filename model.pt

Result:
    ./out/awwkl/zwm-bvd-170m/model.pt
"""

import argparse
import os

from huggingface_hub import hf_hub_download


def find_repo_root(start_path=None):
    """Walk upward from start_path until a .git directory is found."""
    current = start_path or os.getcwd()
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def download(repo_id: str, filename: str = "model.pt", out_root: str = None) -> str:
    """Download filename from repo_id into out_root/repo_id/filename."""
    if out_root is None:
        repo_root = find_repo_root()
        out_root = os.path.join(repo_root or ".", "out")

    target_dir = os.path.join(out_root, repo_id)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading {repo_id}/{filename} → {target_dir}/", flush=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_dir,
    )
    print(f"Saved to {local_path}", flush=True)
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Download a ZWM checkpoint from HuggingFace Hub.")
    parser.add_argument("repo_id", help="HF repo id, e.g. awwkl/zwm-bvd-170m")
    parser.add_argument("--filename", default="model.pt", help="File to download (default: model.pt)")
    parser.add_argument("--out", default=None, help="Override destination root (default: <repo>/out)")
    args = parser.parse_args()
    download(args.repo_id, args.filename, args.out)


if __name__ == "__main__":
    main()