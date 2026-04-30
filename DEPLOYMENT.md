# Deployment Guide

This project is organized for a Hugging Face Spaces Docker deployment.

## Local Run

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

The app expects the inference checkpoint at:

```text
models/resnet50_dyslexia_finetuned.pth
```

You can override model, data, log, and history paths with environment variables:

```bash
DYSLEXIA_MODELS_DIR=/path/to/models
DYSLEXIA_DATA_DIR=/path/to/data
DYSLEXIA_LOG_DIR=/path/to/logs
DYSLEXIA_HISTORY_DB=/path/to/screening_history.db
```

## Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose **Docker** as the SDK.
3. Set the app port to `8501`.
4. Push this repository to the Space.
5. Upload `models/resnet50_dyslexia_finetuned.pth` to the Space repo using Git LFS.
6. Wait for the Docker build to finish.

Recommended commands:

```bash
git lfs install
git lfs track "models/*.pth"
git add .gitattributes models/resnet50_dyslexia_finetuned.pth
git commit -m "Add deployment model checkpoint"
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

If you want screening history to persist across Space restarts, add persistent storage in the Space settings and set this runtime variable:

```text
DYSLEXIA_HISTORY_DB=/data/screening_history.db
```

## Notes

- `frontend/app.py` is the Streamlit UI.
- `backend/` contains configuration, training, evaluation, model inference, OCR, NLP, Grad-CAM, reports, and history.
- `data/` is intentionally excluded from Docker builds.
- Model files are not committed unless you explicitly add them with Git LFS.
