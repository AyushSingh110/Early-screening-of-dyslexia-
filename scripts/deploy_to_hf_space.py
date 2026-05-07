"""Deploy the project folder to a Hugging Face Docker Space.

Usage:
    python scripts/deploy_to_hf_space.py --repo-id AImRs/dyslexia-early-screening-system
"""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi


IGNORE_PATTERNS = [
    ".git/*",
    "__pycache__/*",
    "*.pyc",
    ".env",
    ".env.*",
    "data/*",
    "logs/*",
    "figures/*",
    "notebooks/*",
    "gradcam.ipynb",
    "DOCUMENTATION.md",
    "screening_history.db",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    ".venv/*",
    "venv/*",
    "env/*",
    ".conda/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces.")
    parser.add_argument(
        "--repo-id",
        default="AImRs/dyslexia-early-screening-system",
        help="Space repo id, for example username/space-name.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private if it does not already exist.",
    )
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=".",
        commit_message="Deploy dyslexia screening Streamlit app",
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
