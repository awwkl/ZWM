"""
Upload ZWM checkpoints to HuggingFace Hub.

Edit the UPLOADS list below with (local_path, hf_dest) pairs, where hf_dest
has the form "<org>/<repo_name>/<filename>". Then run:

    python scripts/hf_model_upload.py

Prerequisites:
    huggingface-cli login        # or set HF_TOKEN env var
"""

import argparse
import os

from huggingface_hub import HfApi, create_repo


UPLOADS = [
    # (local_path, "<org>/<repo>/<filename-in-repo>")
    (
        "out/ZWM170M_RGB_BigVideo_200k/model_00200000.pt",
        "awwkl/zwm-bvd-170m/model.pt",
    ),
    (
        "out/ZWM1B_RGB_BigVideo_200k/model_00200000.pt",
        "awwkl/zwm-bvd-1b/model.pt",
    ),
    (
        "out/ZWM170M_RGB_Kinetics_200k/model_00200000.pt",
        "awwkl/zwm-kinetics-170m/model.pt",
    ),
    (
        "out/ZWM1B_RGB_Kinetics_200k/model_00200000.pt",
        "awwkl/zwm-kinetics-1b/model.pt",
    ),
    (
        "out/ZWM170M_RGB_Babyview_200k/model_00200000.pt",
        "awwkl/zwm-babyview-170m/model.pt",
    ),
    (
        "out/ZWM1B_RGB_Babyview_200k/model_00200000.pt",
        "awwkl/zwm-babyview-1b/model.pt",
    ),
]


def parse_dest(hf_dest: str):
    """Split 'awwkl/zwm-bvd-170m/model.pt' → ('awwkl/zwm-bvd-170m', 'model.pt')."""
    parts = hf_dest.split("/", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid hf_dest '{hf_dest}'. Expected '<org>/<repo>/<filename>'.")
    repo_id = f"{parts[0]}/{parts[1]}"
    path_in_repo = parts[2] if len(parts) > 2 else "model.pt"
    return repo_id, path_in_repo


def upload(local_path: str, hf_dest: str, private: bool = False):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    repo_id, path_in_repo = parse_dest(hf_dest)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {local_path} ({size_mb:.1f} MB)")
    print(f"      →  {repo_id} : {path_in_repo}")

    create_repo(repo_id=repo_id, exist_ok=True, private=private)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
    )
    api.super_squash_history(repo_id=repo_id, branch="main")
    print(f"Done: https://huggingface.co/{repo_id}/blob/main/{path_in_repo}\n")


def main():
    parser = argparse.ArgumentParser(description="Upload ZWM checkpoints to HuggingFace Hub.")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LOCAL_PATH:HF_DEST",
        help="Override UPLOADS list. Format: /path/to/model.pt:org/repo/filename.pt "
             "(may be passed multiple times).",
    )
    parser.add_argument("--private", action="store_true", help="Create repos as private.")
    args = parser.parse_args()

    if args.pair:
        uploads = [tuple(p.split(":", 1)) for p in args.pair]
    else:
        uploads = UPLOADS

    if not uploads:
        parser.error("No uploads specified. Edit UPLOADS in this file or pass --pair.")

    for local_path, hf_dest in uploads:
        upload(local_path, hf_dest, private=args.private)


if __name__ == "__main__":
    main()