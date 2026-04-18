"""
Dyslexia Screening System — Streamlit application.

Analyzes a handwriting image using:
  1. Patch-based ResNet-50 vision model (handwriting risk)
  2. OCR + linguistic feature analysis (language risk)
  3. Weighted multimodal fusion to produce a final risk score
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import streamlit as st

import config
from language_model.inference import predict_language_risk
from utils.gradcam import generate_gradcam_visualization
from utils.ocr import extract_text_from_image
from utils.patchify import split_into_patches
from utils.predict import load_model
from utils.preprocess import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RISK_HIGH     = 0.5
RISK_MODERATE = 0.3
MIN_PATCHES   = 5
MAX_FILE_MB   = 10
MIN_OCR_WORDS = 20

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Page background ── */
        .stApp { background-color: #f8f9fc; }

        /* ── Main header ── */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .main-header h1 { color: white; font-size: 2rem; margin: 0 0 0.4rem 0; }
        .main-header p  { color: #a8b2d8; font-size: 1rem; margin: 0; }

        /* ── Metric cards ── */
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #0f3460;
            margin-bottom: 1rem;
        }
        .metric-card .label {
            font-size: 0.78rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .metric-card .value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #1a1a2e;
        }

        /* ── Risk banner ── */
        .risk-banner {
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
            text-align: center;
        }
        .risk-high     { background: #fef2f2; border: 2px solid #ef4444; }
        .risk-moderate { background: #fffbeb; border: 2px solid #f59e0b; }
        .risk-low      { background: #f0fdf4; border: 2px solid #22c55e; }

        .risk-banner .risk-label {
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }
        .risk-high     .risk-label { color: #dc2626; }
        .risk-moderate .risk-label { color: #d97706; }
        .risk-low      .risk-label { color: #16a34a; }

        .risk-banner .risk-msg {
            font-size: 0.95rem;
            margin-top: 0.4rem;
            color: #374151;
        }

        /* ── Score bar ── */
        .score-bar-wrap {
            background: #e5e7eb;
            border-radius: 999px;
            height: 14px;
            margin: 0.5rem 0 0.2rem 0;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        /* ── Section divider ── */
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1a1a2e;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 0.4rem;
            margin: 1.5rem 0 1rem 0;
        }

        /* ── Sidebar tweaks ── */
        section[data-testid="stSidebar"] { background-color: #1a1a2e; }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        section[data-testid="stSidebar"] .stMarkdown a { color: #60a5fa !important; }

        /* ── Upload zone ── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 1rem;
            background: white;
        }

        /* ── Disclaimer ── */
        .disclaimer {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            font-size: 0.85rem;
            color: #1e40af;
            margin-top: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
for key, default in [("model", None), ("device", None), ("enable_gradcam", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# System initialisation
# ---------------------------------------------------------------------------

def initialize_system() -> Tuple[torch.nn.Module, torch.device]:
    try:
        device = torch.device(config.DEVICE)
        with st.spinner("Loading AI model…"):
            model = load_model()
        logger.info("Model loaded on %s", device)
        return model, device
    except Exception as exc:
        logger.error("System init failed: %s", exc)
        st.error(f"System initialisation failed: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def validate_image(uploaded_file) -> bool:
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        st.error(f"File size ({size_mb:.1f} MB) exceeds the {MAX_FILE_MB} MB limit.")
        return False
    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        return True
    except Exception as exc:
        st.error(f"Invalid image file: {exc}")
        return False


# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------

def classify_risk(score: float) -> Tuple[str, str, str]:
    if score >= RISK_HIGH:
        return (
            "HIGH RISK",
            "The handwriting shows significant patterns associated with dyslexia. "
            "We strongly recommend a professional assessment by an educational psychologist.",
            "high",
        )
    if score >= RISK_MODERATE:
        return (
            "MODERATE RISK",
            "Some dyslexia-related patterns were detected. Consider consulting an "
            "educational specialist for further evaluation.",
            "moderate",
        )
    return (
        "LOW RISK",
        "Handwriting appears typical. No strong dyslexia indicators detected.",
        "low",
    )


# ---------------------------------------------------------------------------
# Score bar helper
# ---------------------------------------------------------------------------

def _score_bar(score: float, color: str) -> str:
    pct = int(score * 100)
    colors = {"high": "#ef4444", "moderate": "#f59e0b", "low": "#22c55e"}
    bar_color = colors.get(color, "#6b7280")
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{bar_color};"></div>'
        f"</div>"
        f'<div style="font-size:0.8rem;color:#6b7280;text-align:right;">{pct}%</div>'
    )


# ---------------------------------------------------------------------------
# Vision-model inference
# ---------------------------------------------------------------------------

def run_inference(
    patches: List[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    enable_gradcam: bool = False,
) -> Optional[Dict]:
    dyslexic_count = 0
    predictions: List[float] = []
    gradcam_vis: List[Dict] = []

    progress = st.progress(0)
    status   = st.empty()

    try:
        for idx, patch in enumerate(patches):
            progress.progress((idx + 1) / len(patches))
            status.text(f"Analysing patch {idx + 1} / {len(patches)} …")

            patch_img    = Image.fromarray(patch).convert("RGB")
            input_tensor = transform(patch_img).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(input_tensor).item()

            predictions.append(prob)
            if prob > 0.5:
                dyslexic_count += 1

            if enable_gradcam and (prob > 0.6 or prob < 0.4):
                try:
                    cam, overlay = generate_gradcam_visualization(
                        model, patch, input_tensor, device
                    )
                    if overlay is not None:
                        gradcam_vis.append(
                            {
                                "patch_idx":  idx,
                                "confidence": prob,
                                "prediction": "Dyslexic" if prob > 0.5 else "Normal",
                                "original":   patch,
                                "overlay":    overlay,
                                "cam":        cam,
                            }
                        )
                except Exception as exc:
                    logger.warning("Grad-CAM failed for patch %d: %s", idx, exc)

    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        st.error(f"Analysis failed: {exc}")
        return None
    finally:
        progress.empty()
        status.empty()

    preds = np.array(predictions)
    return {
        "total_patches":          len(patches),
        "dyslexic_patches":       dyslexic_count,
        "normal_patches":         len(patches) - dyslexic_count,
        "risk_score":             dyslexic_count / len(patches),
        "avg_confidence":         float(preds.mean()),
        "max_confidence":         float(preds.max()),
        "min_confidence":         float(preds.min()),
        "predictions":            predictions,
        "gradcam_visualizations": gradcam_vis,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _render_gradcam(results: Dict) -> None:
    st.markdown('<div class="section-title">AI Explainability — What the Model Sees</div>', unsafe_allow_html=True)

    vis_data = results.get("gradcam_visualizations", [])
    if not vis_data:
        st.warning(
            "No Grad-CAM visualisations generated. All patches had moderate confidence (0.40–0.60). "
            "Try a higher-contrast image."
        )
        return

    st.info(
        "**Grad-CAM heatmaps** show which parts of the handwriting most influenced "
        "the model's prediction. Warmer colours (red/yellow) = high attention."
    )

    dyslexic_vis = sorted(
        [v for v in vis_data if v["prediction"] == "Dyslexic"],
        key=lambda x: x["confidence"], reverse=True,
    )
    normal_vis = sorted(
        [v for v in vis_data if v["prediction"] == "Normal"],
        key=lambda x: x["confidence"],
    )

    c1, c2 = st.columns(2)
    c1.metric("Dyslexic Patches Visualised", len(dyslexic_vis))
    c2.metric("Normal Patches Visualised",   len(normal_vis))

    for section_label, vis_list in [
        ("High Dyslexia Confidence Patches", dyslexic_vis),
        ("Low Dyslexia Confidence Patches (Normal)", normal_vis),
    ]:
        if not vis_list:
            continue
        st.markdown(f"**{section_label}**")
        for i, vis in enumerate(vis_list[:3]):
            label = f"Patch {vis['patch_idx'] + 1}  —  Dyslexia probability: {vis['confidence']:.1%}"
            with st.expander(label, expanded=(i == 0)):
                col_orig, col_cam = st.columns(2)
                col_orig.image(vis["original"], caption="Original Patch",   width="stretch")
                col_cam.image(vis["overlay"],   caption="Grad-CAM Overlay", width="stretch")
                st.caption(
                    f"Model assigned **{vis['confidence']:.1%}** dyslexia probability. "
                    "Red/yellow regions highlight features that drove the prediction."
                )


def display_results(
    results: Dict,
    handwriting_risk: float,
    language_risk: Optional[float],
    final_risk: float,
    enable_gradcam: bool,
) -> None:
    risk_label, message, level = classify_risk(final_risk)

    # ── Risk banner ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="risk-banner risk-{level}">'
        f'<div class="risk-label">{risk_label}</div>'
        f'<div class="risk-msg">{message}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Score bar ────────────────────────────────────────────────────────────
    st.markdown(
        f"**Overall Screening Score: {final_risk:.1%}**"
        + _score_bar(final_risk, level),
        unsafe_allow_html=True,
    )

    # ── Top metric cards ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Multimodal Risk Breakdown</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Handwriting Risk</div>'
            f'<div class="value">{handwriting_risk:.1%}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        lang_val = f"{language_risk:.1%}" if language_risk is not None else "N/A"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Language Risk</div>'
            f'<div class="value">{lang_val}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="label">Patches Analysed</div>'
            f'<div class="value">{results["total_patches"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if language_risk is None:
        st.caption(
            "Language risk unavailable — fewer than 20 words extracted by OCR. "
            "Final score is based on handwriting only."
        )

    # ── Detailed stats ───────────────────────────────────────────────────────
    with st.expander("Detailed Patch Analysis"):
        ca, cb = st.columns(2)
        with ca:
            st.write("**Patch classification**")
            st.write(f"- Dyslexia-like : {results['dyslexic_patches']} patches")
            st.write(f"- Typical       : {results['normal_patches']} patches")
        with cb:
            st.write("**Per-patch confidence**")
            st.write(f"- Mean    : {results['avg_confidence']:.1%}")
            st.write(f"- Highest : {results['max_confidence']:.1%}")
            st.write(f"- Lowest  : {results['min_confidence']:.1%}")
        st.write("**Confidence distribution across patches**")
        st.bar_chart(results["predictions"])

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    if enable_gradcam:
        _render_gradcam(results)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        "<strong>Important notice</strong> — This tool provides <em>early screening support only</em> "
        "and is <strong>not</strong> a medical diagnosis. Always consult a qualified educational "
        "psychologist or medical professional for a comprehensive assessment."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Dyslexia Screening System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">'
        "<h1>🧠 AI-Based Dyslexia Screening</h1>"
        "<p>Upload a handwriting sample to receive an instant, AI-powered dyslexia risk assessment "
        "— combining deep learning vision analysis with linguistic pattern detection.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## About")
        st.markdown(
            "This system uses a fine-tuned **ResNet-50** model trained on **128,902** "
            "handwriting images to screen for dyslexia indicators."
        )
        st.markdown("---")
        st.markdown(
            "**Model performance**\n"
            "- Test accuracy : **75.9%**\n"
            "- ROC-AUC       : **84.9%**\n"
            "- Dyslexic recall: **79.9%**\n"
            "- Architecture  : ResNet-50\n"
            f"- Device        : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}"
        )
        st.markdown("---")
        st.markdown("**Key features**")
        st.markdown(
            "- Patch-based handwriting analysis\n"
            "- OCR + linguistic risk scoring\n"
            "- Multimodal fusion (vision + NLP)\n"
            "- Grad-CAM explainability heatmaps"
        )
        st.markdown("---")

        enable_gradcam = st.checkbox(
            "Enable Grad-CAM Visualisation",
            value=False,
            help="Show attention heatmaps — adds a few seconds of processing time",
        )
        st.session_state.enable_gradcam = enable_gradcam
        if enable_gradcam:
            st.success("Grad-CAM enabled")
            st.caption("Heatmaps shown for patches with confidence > 60% or < 40%")
        else:
            st.info("Enable Grad-CAM to see what the model focuses on.")

        st.markdown("---")
        st.markdown(
            "**Upload guidelines**\n"
            "- Clear, well-lit handwriting scan\n"
            "- Full page gives more reliable results\n"
            f"- Max {MAX_FILE_MB} MB  ·  PNG / JPG / JPEG"
        )

    # ── Model load ────────────────────────────────────────────────────────────
    if st.session_state.model is None:
        st.session_state.model, st.session_state.device = initialize_system()

    model  = st.session_state.model
    device = st.session_state.device

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Upload Handwriting Sample</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a handwriting image (full page recommended for best results)",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear scan or photo of handwritten text.",
    )

    if uploaded is None:
        # Show a friendly placeholder
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
                <div style="font-size:3rem;">📄</div>
                <div style="font-size:1rem;margin-top:0.5rem;">
                    Upload a handwriting image above to begin screening
                </div>
                <div style="font-size:0.85rem;margin-top:0.3rem;">
                    Supports full-page scans · PNG, JPG, JPEG · up to 10 MB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not validate_image(uploaded):
        return

    try:
        image    = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, caption="Uploaded Handwriting Sample", width="stretch")
        with col_info:
            st.markdown('<div class="section-title">Image Info</div>', unsafe_allow_html=True)
            st.markdown(
                f"- **Width**  : {image.size[0]} px\n"
                f"- **Height** : {image.size[1]} px\n"
                f"- **Mode**   : {image.mode}"
            )
            st.markdown("---")
            st.markdown(
                "**How it works**\n\n"
                "1. Image is split into 224×224 patches\n"
                "2. Each patch is scored by ResNet-50\n"
                "3. OCR extracts text for language analysis\n"
                "4. Scores are fused into a final risk level"
            )

        if not st.button("🔍 Analyse Handwriting", type="primary", use_container_width=True):
            return

        # ── Patch extraction ─────────────────────────────────────────────────
        with st.spinner("Extracting patches…"):
            patches = split_into_patches(
                image_np,
                patch_size=config.IMAGE_SIZE,
                stride=config.IMAGE_SIZE,
            )

        if len(patches) == 0:
            st.error("No valid patches found. The image may be blank or too small (minimum ~224×224 px).")
            return

        if len(patches) < MIN_PATCHES:
            st.warning(
                f"Only {len(patches)} patches detected (recommended ≥ {MIN_PATCHES}). "
                "Results may be less reliable. Try uploading a larger image."
            )

        st.success(f"✅ Detected **{len(patches)}** handwriting patches.")

        # ── Vision inference ─────────────────────────────────────────────────
        results = run_inference(patches, model, device, enable_gradcam=st.session_state.enable_gradcam)
        if results is None:
            return

        # ── Language analysis ────────────────────────────────────────────────
        with st.spinner("Analysing written content (language patterns)…"):
            extracted_text = extract_text_from_image(image)
            word_count     = len(extracted_text.split())

            if word_count < MIN_OCR_WORDS:
                st.warning(
                    f"Only {word_count} words extracted by OCR (minimum {MIN_OCR_WORDS} required). "
                    "Language-based analysis skipped — handwriting risk only."
                )
                language_risk = None
            else:
                language_risk = predict_language_risk(extracted_text)

        # ── Multimodal fusion ────────────────────────────────────────────────
        handwriting_risk = results["risk_score"]

        if language_risk is None:
            final_risk = handwriting_risk
        else:
            final_risk = (
                0.60 * handwriting_risk
                + 0.25 * language_risk
                + 0.15 * max(language_risk - handwriting_risk, 0.0)
            )

        # Conservative cap: if handwriting looks clearly normal, don't let
        # language noise push the score into the moderate band.
        if handwriting_risk < 0.2:
            final_risk = min(final_risk, 0.45)

        # ── Display ──────────────────────────────────────────────────────────
        display_results(results, handwriting_risk, language_risk, final_risk, st.session_state.enable_gradcam)

        logger.info(
            "Analysis complete — handwriting=%.2f  language=%s  final=%.2f  patches=%d",
            handwriting_risk,
            f"{language_risk:.2f}" if language_risk is not None else "N/A",
            final_risk,
            results["total_patches"],
        )

    except Exception as exc:
        logger.error("Processing error: %s", exc, exc_info=True)
        st.error(f"An error occurred during processing: {exc}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;margin-top:2rem;'>"
        "For educational and screening purposes only &nbsp;·&nbsp; "
        "Not a substitute for professional diagnosis &nbsp;·&nbsp; "
        f"v{getattr(config, 'VERSION', '1.0.0')}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
