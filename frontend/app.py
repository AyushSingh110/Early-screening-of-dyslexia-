"""
Dyslexia Screening System — Streamlit application v2.

New in v2:
  - Letter reversal detection  (utils/reversal_detector.py)
  - Writing regularity analysis (utils/regularity.py)
  - PDF report download         (utils/report.py)
  - Screening history tab       (utils/history.py)
  - Batch upload (ZIP of images)
"""

import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.language_model.inference import predict_language_risk
from backend.utils.gradcam import generate_gradcam_visualization
from backend.utils.history import clear_history, export_csv, get_history, save_screening
from backend.utils.ocr import extract_text_from_image
from backend.utils.patchify import split_into_patches
from backend.utils.predict import load_model, predict_patch_tta
from backend.utils.preprocess import transform
from backend.utils.regularity import analyze_regularity
from backend.utils.report import generate_pdf_report
from backend.utils.reversal_detector import detect_reversals, reversal_risk_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_HIGH     = 0.5
RISK_MODERATE = 0.3
MIN_PATCHES   = 5
MAX_FILE_MB   = 10
MIN_OCR_WORDS = 20

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Clinical Design System ── */
    :root {
        --bg:        #f0f4f8;
        --surface:   #ffffff;
        --border:    #dce4ed;
        --ink:       #0d1b2a;
        --ink2:      #3d5166;
        --ink3:      #7a92a8;
        --primary:   #1a56db;
        --primary-h: #1545b8;
        --teal:      #0891b2;
        --green:     #059669;
        --amber:     #b45309;
        --red:       #b91c1c;
        --green-bg:  #ecfdf5;
        --amber-bg:  #fffbeb;
        --red-bg:    #fef2f2;
        --green-bd:  #6ee7b7;
        --amber-bd:  #fcd34d;
        --red-bd:    #fca5a5;
    }

    /* Page */
    .stApp { background-color: var(--bg); color: var(--ink); }
    .block-container { max-width: 1160px; padding-top: 1.2rem; padding-bottom: 3rem; }
    h1,h2,h3,h4,h5,h6 { color: var(--ink); letter-spacing: -.01em; }

    /* ── Header ── */
    .main-header {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 4px solid var(--primary);
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .main-header .hdr-icon {
        font-size: 2.6rem;
        line-height: 1;
        flex-shrink: 0;
    }
    .main-header .hdr-text h1 {
        color: var(--ink);
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 .25rem 0;
        letter-spacing: -.02em;
    }
    .main-header .hdr-text p { color: var(--ink2); font-size: .93rem; margin: 0; }
    .main-header .hdr-badge {
        margin-left: auto;
        flex-shrink: 0;
        background: #eff6ff;
        color: var(--primary);
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: .3rem .9rem;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 3px solid var(--primary);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 96px;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .metric-card .label {
        font-size: .7rem;
        color: var(--ink3);
        text-transform: uppercase;
        letter-spacing: .07em;
        font-weight: 700;
        margin-bottom: .4rem;
    }
    .metric-card .value { font-size: 1.5rem; font-weight: 800; color: var(--ink); }

    /* ── Risk banners ── */
    .risk-banner {
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        display: flex;
        flex-direction: column;
        gap: .3rem;
    }
    .risk-high     { background: var(--red-bg);   border-color: var(--red);   border: 1px solid var(--red-bd);   border-left: 5px solid var(--red); }
    .risk-moderate { background: var(--amber-bg); border-color: var(--amber); border: 1px solid var(--amber-bd); border-left: 5px solid var(--amber); }
    .risk-low      { background: var(--green-bg); border-color: var(--green); border: 1px solid var(--green-bd); border-left: 5px solid var(--green); }
    .risk-banner .risk-label { font-size: 1.25rem; font-weight: 800; }
    .risk-high     .risk-label { color: var(--red); }
    .risk-moderate .risk-label { color: var(--amber); }
    .risk-low      .risk-label { color: var(--green); }
    .risk-banner .risk-msg { font-size: .88rem; color: var(--ink2); line-height: 1.5; }

    /* ── Score bar ── */
    .score-bar-wrap {
        background: #e8edf3;
        border-radius: 999px;
        height: 10px;
        margin: .5rem 0 .15rem;
        overflow: hidden;
    }
    .score-bar-fill { height: 100%; border-radius: 999px; }

    /* ── Section titles ── */
    .section-title {
        font-size: .95rem;
        font-weight: 700;
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: .06em;
        border-bottom: 2px solid var(--border);
        padding-bottom: .4rem;
        margin: 1.5rem 0 .9rem;
    }

    /* ── Tabs ── */
    div[data-testid="stTabs"] button {
        font-weight: 600;
        color: var(--ink2);
        border-radius: 6px 6px 0 0;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--primary);
        background: #eff6ff;
        font-weight: 700;
    }

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] {
        background: var(--surface);
        border: 2px dashed #c4d4e4;
        border-radius: 10px;
        padding: .8rem;
    }
    div[data-testid="stFileUploader"]:hover { border-color: var(--primary); }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 7px;
        font-weight: 600;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--ink);
    }
    .stButton > button[kind="primary"] {
        background: var(--primary);
        border-color: var(--primary);
        color: #ffffff;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover { background: var(--primary-h); }
    .stDownloadButton > button {
        border-radius: 7px;
        font-weight: 600;
        background: var(--surface);
        border: 1px solid var(--primary);
        color: var(--primary);
    }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* ── Data table ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #1e3a5f !important;
        border-right: 1px solid #162d4a;
    }
    section[data-testid="stSidebar"] * { color: #e2eaf4 !important; }
    section[data-testid="stSidebar"] .stMarkdown a { color: #93c5fd !important; }
    section[data-testid="stSidebar"] hr { border-color: #2d4f73 !important; }
    section[data-testid="stSidebar"] .stCheckbox label { font-size: .9rem !important; }
    .sidebar-badge {
        display: inline-block;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 5px;
        padding: .15rem .55rem;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .04em;
        color: #bde0ff;
        margin-bottom: .6rem;
    }
    .sidebar-metric {
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 7px;
        padding: .55rem .8rem;
        margin-bottom: .4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .sidebar-metric .sm-label { font-size: .78rem; color: #93b4d4; }
    .sidebar-metric .sm-val   { font-size: .88rem; font-weight: 700; color: #e8f2ff; }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: .9rem 1.1rem;
        font-size: .82rem;
        color: #1e40af;
        margin-top: 1.5rem;
        line-height: 1.55;
    }

    /* ── Info / success / warning overrides ── */
    div[data-testid="stAlert"] { border-radius: 8px !important; }

    /* ── Mobile ── */
    @media (max-width: 720px) {
        .block-container { padding-left: .8rem; padding-right: .8rem; }
        .main-header { flex-direction: column; align-items: flex-start; gap: .6rem; }
        .main-header .hdr-badge { margin-left: 0; }
        .main-header .hdr-text h1 { font-size: 1.4rem; }
        .metric-card { min-height: auto; }
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
for _k, _v in [("model", None), ("device", None), ("enable_gradcam", False),
                ("use_tta", False), ("spell_checker", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_spell_checker():
    if st.session_state.spell_checker is None:
        try:
            import enchant
            st.session_state.spell_checker = enchant.Dict("en_US")
        except Exception:
            st.session_state.spell_checker = False
    return st.session_state.spell_checker if st.session_state.spell_checker is not False else None


def classify_risk(score: float) -> Tuple[str, str, str]:
    if score >= RISK_HIGH:
        return ("HIGH RISK",
                "Significant dyslexia-related patterns detected. Professional assessment strongly recommended.",
                "high")
    if score >= RISK_MODERATE:
        return ("MODERATE RISK",
                "Some dyslexia-related patterns detected. Consider consulting an educational specialist.",
                "moderate")
    return ("LOW RISK", "Handwriting appears typical. No strong dyslexia indicators detected.", "low")


def _score_bar(score: float, level: str) -> str:
    pct = int(score * 100)
    col = {"high": "#b91c1c", "moderate": "#b45309", "low": "#059669"}.get(level, "#64748b")
    return (f'<div class="score-bar-wrap">'
            f'<div class="score-bar-fill" style="width:{pct}%;background:{col};"></div></div>'
            f'<div style="font-size:.8rem;color:#6b7280;text-align:right;">{pct}%</div>')


def _card(label: str, value: str) -> str:
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>')


def initialize_system() -> Tuple[torch.nn.Module, torch.device]:
    try:
        device = torch.device(config.DEVICE)
        with st.spinner("Loading AI model…"):
            model = load_model()
        return model, device
    except Exception as exc:
        st.error(f"System initialisation failed: {exc}")
        st.stop()


def validate_image(f) -> bool:
    if f.size / 1024 / 1024 > MAX_FILE_MB:
        st.error(f"File too large ({f.size/1024/1024:.1f} MB). Max {MAX_FILE_MB} MB.")
        return False
    try:
        Image.open(f).verify(); f.seek(0)
        return True
    except Exception as exc:
        st.error(f"Invalid image: {exc}")
        return False


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def run_vision(patches, model, device, enable_gradcam, use_tta) -> Optional[Dict]:
    dyslexic_count = 0
    predictions: List[float] = []
    gradcam_vis: List[Dict]  = []

    progress = st.progress(0)
    status   = st.empty()

    try:
        for idx, patch in enumerate(patches):
            progress.progress((idx + 1) / len(patches))
            status.text(f"Analysing patch {idx + 1}/{len(patches)}…")

            pil = Image.fromarray(patch).convert("RGB")

            if use_tta:
                prob = predict_patch_tta(model, pil, device, n_augments=5)
            else:
                tensor = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = model(tensor).item()

            # Model outputs P(normal). Dyslexic = P(normal) < 0.5
            dyslexia_prob = 1.0 - prob
            predictions.append(dyslexia_prob)
            if dyslexia_prob > 0.5:
                dyslexic_count += 1

            if enable_gradcam and (dyslexia_prob > 0.6 or dyslexia_prob < 0.4):
                try:
                    t = transform(pil).unsqueeze(0).to(device)
                    cam, overlay = generate_gradcam_visualization(model, patch, t, device)
                    if overlay is not None:
                        gradcam_vis.append({"patch_idx": idx, "confidence": dyslexia_prob,
                                            "prediction": "Dyslexic" if dyslexia_prob > 0.5 else "Normal",
                                            "original": patch, "overlay": overlay})
                except Exception as exc:
                    logger.warning("Grad-CAM patch %d: %s", idx, exc)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return None
    finally:
        progress.empty(); status.empty()

    arr = np.array(predictions)
    return {
        "total_patches":   len(patches),
        "dyslexic_patches": dyslexic_count,
        "normal_patches":  len(patches) - dyslexic_count,
        "risk_score":      dyslexic_count / len(patches),
        "avg_confidence":  float(arr.mean()),
        "max_confidence":  float(arr.max()),
        "min_confidence":  float(arr.min()),
        "predictions":     predictions,
        "gradcam_visualizations": gradcam_vis,
    }


def fuse_risks(hw: float, lang: Optional[float], rev: float, reg: float) -> float:
    """
    Weighted fusion of all four risk signals.
      60% handwriting (vision model)
      20% language (OCR + NLP)
      10% letter reversals
      10% writing regularity
    """
    if lang is None:
        score = 0.70 * hw + 0.15 * rev + 0.15 * reg
    else:
        score = (0.60 * hw + 0.20 * lang
                 + 0.15 * max(lang - hw, 0.0)   # differential bonus
                 + 0.10 * rev + 0.10 * reg)

    if hw < 0.2:
        score = min(score, 0.45)           # conservative cap when vision says normal
    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _render_gradcam(results: Dict) -> None:
    st.markdown('<div class="section-title">AI Explainability — Grad-CAM</div>',
                unsafe_allow_html=True)
    vis = results.get("gradcam_visualizations", [])
    if not vis:
        st.warning("No Grad-CAM visualisations — all patches had moderate confidence.")
        return
    st.info("**Grad-CAM heatmaps** — warmer colours (red/yellow) = high model attention.")
    for grp, lst in [("Dyslexic patches",
                       sorted([v for v in vis if v["prediction"]=="Dyslexic"],
                              key=lambda x: x["confidence"], reverse=True)),
                     ("Normal patches",
                       sorted([v for v in vis if v["prediction"]=="Normal"],
                              key=lambda x: x["confidence"]))]:
        if not lst:
            continue
        st.markdown(f"**{grp}**")
        for i, v in enumerate(lst[:3]):
            with st.expander(f"Patch {v['patch_idx']+1}  —  {v['confidence']:.1%}", expanded=(i==0)):
                c1, c2 = st.columns(2)
                c1.image(v["original"], caption="Original",      width="stretch")
                c2.image(v["overlay"],  caption="Grad-CAM",      width="stretch")


def display_results(
    results: Dict, hw: float, lang: Optional[float],
    final: float, rev_info: Dict, reg_info: Dict,
    enable_gradcam: bool, filename: str,
    image,
) -> None:
    label, msg, level = classify_risk(final)

    # Risk banner
    st.markdown(
        f'<div class="risk-banner risk-{level}">'
        f'<div class="risk-label">{label}</div>'
        f'<div class="risk-msg">{msg}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Overall Screening Score: {final:.1%}**" + _score_bar(final, level),
                unsafe_allow_html=True)

    # Four risk signals
    st.markdown('<div class="section-title">Multimodal Risk Breakdown</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_card("Handwriting Risk", f"{hw:.1%}"), unsafe_allow_html=True)
    c2.markdown(_card("Language Risk",
                       f"{lang:.1%}" if lang is not None else "N/A"), unsafe_allow_html=True)
    c3.markdown(_card("Reversal Risk",
                       f"{reversal_risk_score(rev_info):.1%}"), unsafe_allow_html=True)
    c4.markdown(_card("Regularity Risk",
                       f"{reg_info.get('regularity_risk', 0):.1%}"), unsafe_allow_html=True)

    # Reversal detail
    if rev_info.get("total_words", 0) > 0:
        with st.expander(
            f"Letter Reversal Analysis — {rev_info['reversal_count']} reversals "
            f"({rev_info['reversal_ratio']:.1%})"
        ):
            st.write(f"**Words analysed:** {rev_info['total_words']}")
            st.write(f"**Reversals found:** {rev_info['reversal_count']} "
                     f"({rev_info['reversal_ratio']:.1%})")
            if rev_info.get("reversal_words"):
                examples = " · ".join(f"`{w}` → `{c}`"
                                      for w, c in rev_info["reversal_words"])
                st.write(f"**Examples:** {examples}")
            st.caption("Typical dyslexic reversals: b↔d, p↔q, n↔u, m↔w")

    # Regularity detail
    with st.expander("Writing Regularity Analysis"):
        ra, rb = st.columns(2)
        ra.metric("Baseline Straightness",
                  f"{reg_info.get('baseline_straightness', 0):.2f}",
                  help="1.0 = perfectly straight lines")
        rb.metric("Letter Size Variance",
                  f"{reg_info.get('letter_size_variance', 0):.2f}",
                  help="Higher = more inconsistent letter sizes")

    # Patch detail
    with st.expander("Detailed Patch Analysis"):
        ca, cb = st.columns(2)
        ca.write(f"- Dyslexia-like: **{results['dyslexic_patches']}** patches")
        ca.write(f"- Typical:       **{results['normal_patches']}** patches")
        cb.write(f"- Mean confidence:    **{results['avg_confidence']:.1%}**")
        cb.write(f"- Highest confidence: **{results['max_confidence']:.1%}**")
        st.write("**Confidence distribution**")
        st.bar_chart(results["predictions"])

    if enable_gradcam:
        _render_gradcam(results)

    # PDF download
    st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
    try:
        pdf_bytes = generate_pdf_report(
            filename=filename,
            results=results,
            handwriting_risk=hw,
            language_risk=lang,
            final_risk=final,
            risk_label=label,
            reversal_info=rev_info,
            regularity_info=reg_info,
        )
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"dyslexia_screening_{filename.split('.')[0]}.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.info("Install `reportlab` to enable PDF reports: `pip install reportlab`")
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer"><strong>Important</strong> — This tool provides '
        '<em>early screening support only</em> and is <strong>not</strong> a medical '
        'diagnosis. Always consult a qualified educational psychologist.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Single image analysis
# ---------------------------------------------------------------------------

def analyse_single(uploaded, model, device, enable_gradcam, use_tta) -> None:
    if not validate_image(uploaded):
        return

    image    = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col_img, col_info = st.columns([2, 1])
    with col_img:
        st.image(image, caption="Uploaded Handwriting Sample", width="stretch")
    with col_info:
        st.markdown('<div class="section-title">Image Info</div>', unsafe_allow_html=True)
        st.markdown(f"- **Width**  : {image.size[0]} px\n"
                    f"- **Height** : {image.size[1]} px\n"
                    f"- **Mode**   : {image.mode}")
        st.markdown("---")
        st.markdown("**Pipeline**\n1. Patch extraction (224×224)\n"
                    "2. ResNet-50 per-patch scoring\n"
                    "3. OCR + language analysis\n"
                    "4. Reversal & regularity check\n"
                    "5. Multimodal fusion")

    if not st.button("🔍 Analyse Handwriting", type="primary", use_container_width=True):
        return

    # Patch extraction
    with st.spinner("Extracting patches…"):
        patches = split_into_patches(image_np, patch_size=config.IMAGE_SIZE,
                                     stride=config.IMAGE_SIZE)
    if not patches:
        st.error("No valid patches found. Image may be blank or too small.")
        return
    if len(patches) < MIN_PATCHES:
        st.warning(f"Only {len(patches)} patches (recommended ≥ {MIN_PATCHES}). "
                   "Results may be less reliable.")
    st.success(f"✅ Detected **{len(patches)}** handwriting patches.")

    # Vision model
    results = run_vision(patches, model, device, enable_gradcam, use_tta)
    if results is None:
        return

    # OCR + language
    with st.spinner("OCR + language analysis…"):
        text       = extract_text_from_image(image)
        word_count = len(text.split())
        if word_count < MIN_OCR_WORDS:
            st.warning(f"Only {word_count} words extracted — language analysis skipped.")
            lang_risk = None
        else:
            lang_risk = predict_language_risk(text)

    # Reversal detection
    checker  = _get_spell_checker()
    rev_info = detect_reversals(text, spell_checker=checker)

    # Regularity
    with st.spinner("Analysing handwriting regularity…"):
        reg_info = analyze_regularity(image_np)

    # Fuse
    hw    = results["risk_score"]
    rev_r = reversal_risk_score(rev_info)
    reg_r = reg_info.get("regularity_risk", 0.5)
    final = fuse_risks(hw, lang_risk, rev_r, reg_r)

    label, _, _ = classify_risk(final)

    # Persist to history
    save_screening(
        filename=uploaded.name,
        final_risk=final,
        risk_label=label,
        handwriting_risk=hw,
        language_risk=lang_risk,
        reversal_ratio=rev_info.get("reversal_ratio", 0.0),
        regularity_risk=reg_r,
        total_patches=results["total_patches"],
        dyslexic_patches=results["dyslexic_patches"],
    )

    display_results(results, hw, lang_risk, final, rev_info, reg_info,
                    enable_gradcam, uploaded.name, image)

    logger.info("Analysis — hw=%.2f lang=%s rev=%.2f reg=%.2f final=%.2f patches=%d",
                hw, f"{lang_risk:.2f}" if lang_risk else "N/A",
                rev_r, reg_r, final, results["total_patches"])


# ---------------------------------------------------------------------------
# Batch analysis (ZIP of images)
# ---------------------------------------------------------------------------

def analyse_batch(zip_file, model, device, use_tta) -> None:
    st.markdown('<div class="section-title">Batch Results</div>', unsafe_allow_html=True)

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_file.read()))
    except Exception as exc:
        st.error(f"Could not read ZIP file: {exc}")
        return

    image_names = [n for n in zf.namelist()
                   if n.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_names:
        st.error("No PNG/JPG images found inside the ZIP file.")
        return

    st.info(f"Found **{len(image_names)}** images in ZIP. Analysing…")

    rows = []
    prog = st.progress(0)

    for i, name in enumerate(image_names):
        prog.progress((i + 1) / len(image_names))
        try:
            data     = zf.read(name)
            image    = Image.open(io.BytesIO(data)).convert("RGB")
            image_np = np.array(image)

            patches = split_into_patches(image_np, config.IMAGE_SIZE, config.IMAGE_SIZE)
            if not patches:
                rows.append({"File": name, "Risk Score": "—",
                             "Risk Level": "No patches", "Patches": 0})
                continue

            preds = []
            for patch in patches:
                pil = Image.fromarray(patch).convert("RGB")
                if use_tta:
                    preds.append(predict_patch_tta(model, pil, device))
                else:
                    t = transform(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        preds.append(model(t).item())

            # Model outputs P(normal); dyslexic = P(normal) < 0.5
            hw    = sum(1 for p in preds if p < 0.5) / len(preds)
            label, _, _ = classify_risk(hw)
            rows.append({
                "File":       name,
                "Risk Score": f"{hw:.1%}",
                "Risk Level": label,
                "Patches":    len(patches),
                "Avg Conf":   f"{np.mean(preds):.1%}",
            })
        except Exception as exc:
            rows.append({"File": name, "Risk Score": "Error",
                         "Risk Level": str(exc), "Patches": 0})

    prog.empty()

    import pandas as pd
    df = pd.DataFrame(rows)
    # Sort by risk descending
    risk_order = {"HIGH RISK": 0, "MODERATE RISK": 1, "LOW RISK": 2}
    if "Risk Level" in df.columns:
        df["_ord"] = df["Risk Level"].map(risk_order).fillna(3)
        df = df.sort_values("_ord").drop(columns=["_ord"])

    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download batch results CSV", csv_bytes,
                       "batch_screening.csv", "text/csv")


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

def show_history() -> None:
    st.markdown('<div class="section-title">Screening History</div>',
                unsafe_allow_html=True)
    history = get_history(limit=100)
    if not history:
        st.info("No screenings recorded yet. Run an analysis first.")
        return

    import pandas as pd
    df = pd.DataFrame(history)
    # Friendly column names
    df = df.rename(columns={
        "timestamp":       "Time",
        "filename":        "File",
        "final_risk":      "Final Risk",
        "risk_label":      "Level",
        "handwriting_risk":"HW Risk",
        "language_risk":   "Lang Risk",
        "reversal_ratio":  "Reversal %",
        "regularity_risk": "Reg Risk",
        "total_patches":   "Patches",
        "dyslexic_patches":"Dyslexic",
    })
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    csv_bytes = export_csv()
    c1.download_button("📥 Export history CSV", csv_bytes,
                       "screening_history.csv", "text/csv")
    if c2.button("🗑 Clear history"):
        clear_history()
        st.success("History cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Dyslexia Screening System",
        page_icon="🧠", layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    # Header
    st.markdown(
        '<div class="main-header">'
        '<div class="hdr-icon">🧠</div>'
        '<div class="hdr-text">'
        '<h1>Dyslexia Early Screening System</h1>'
        '<p>AI-powered handwriting analysis combining deep learning, OCR, letter reversal '
        'detection, and writing regularity — for early identification support only.</p>'
        '</div>'
        '<div class="hdr-badge">Screening Tool · Not a Diagnosis</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-badge">AI Screening System</div>', unsafe_allow_html=True)
        st.markdown("### Model Info")
        st.markdown(
            '<div class="sidebar-metric"><span class="sm-label">Architecture</span>'
            '<span class="sm-val">ResNet-50</span></div>'
            '<div class="sidebar-metric"><span class="sm-label">Training images</span>'
            '<span class="sm-val">128,902</span></div>'
            '<div class="sidebar-metric"><span class="sm-label">Test accuracy</span>'
            '<span class="sm-val">77.0 %</span></div>'
            '<div class="sidebar-metric"><span class="sm-label">ROC-AUC</span>'
            '<span class="sm-val">86.0 %</span></div>'
            '<div class="sidebar-metric"><span class="sm-label">Dyslexic F1</span>'
            '<span class="sm-val">79.3 %</span></div>'
            f'<div class="sidebar-metric"><span class="sm-label">Device</span>'
            f'<span class="sm-val">{"GPU" if torch.cuda.is_available() else "CPU"}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Analysis Settings")
        enable_gradcam = st.checkbox("Enable Grad-CAM",
                                     help="Show attention heatmaps — highlights regions the model "
                                          "focuses on (adds ~2 s per image)")
        use_tta        = st.checkbox("Enable TTA",
                                     help="Test-Time Augmentation — averages 5 augmented passes "
                                          "per patch for +1–2% accuracy")
        st.session_state.enable_gradcam = enable_gradcam
        st.session_state.use_tta        = use_tta
        if use_tta:
            st.info("TTA active — 5× augmented predictions averaged per patch.")
        st.markdown("---")
        st.markdown("### Upload Requirements")
        st.markdown(
            "- Clear, well-lit scan or photo\n"
            "- Cursive or print handwriting\n"
            "- Full page preferred\n"
            f"- PNG · JPG · JPEG · max {MAX_FILE_MB} MB"
        )
        st.markdown("---")
        st.markdown(
            '<div style="font-size:.75rem;color:#7a9bbf;line-height:1.6;">'
            'This tool is intended for <strong style="color:#aac8e8;">early screening '
            'support only</strong>. Results must be interpreted by a qualified '
            'educational psychologist or clinician.</div>',
            unsafe_allow_html=True,
        )

    # Load model
    if st.session_state.model is None:
        st.session_state.model, st.session_state.device = initialize_system()

    model  = st.session_state.model
    device = st.session_state.device

    # Tabs
    tab_single, tab_batch, tab_history = st.tabs(
        ["📄 Single Image", "📦 Batch Upload (ZIP)", "📋 History"]
    )

    with tab_single:
        st.markdown('<div class="section-title">Upload Handwriting Sample</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose a handwriting image",
                                    type=["png", "jpg", "jpeg"])
        if uploaded:
            analyse_single(uploaded, model, device,
                           st.session_state.enable_gradcam,
                           st.session_state.use_tta)
        else:
            st.markdown("""
            <div style="background:#ffffff;border:1px solid #dce4ed;border-radius:10px;
                        text-align:center;padding:2.8rem 1.5rem;margin-top:.5rem;">
                <div style="font-size:2.8rem;margin-bottom:.6rem;">📋</div>
                <div style="font-size:1.05rem;font-weight:700;color:#0d1b2a;margin-bottom:.35rem;">
                    Upload a handwriting sample to begin
                </div>
                <div style="font-size:.85rem;color:#7a92a8;max-width:380px;margin:0 auto;
                            line-height:1.6;">
                    The system will extract patches, run the AI model, perform OCR,
                    detect letter reversals, and produce a multimodal risk score.
                </div>
                <div style="margin-top:1rem;font-size:.78rem;color:#a0b4c8;">
                    PNG · JPG · JPEG &nbsp;·&nbsp; up to 10 MB &nbsp;·&nbsp;
                    Full page preferred
                </div>
            </div>""", unsafe_allow_html=True)

    with tab_batch:
        st.markdown('<div class="section-title">Batch Analysis</div>',
                    unsafe_allow_html=True)
        st.info("Upload a ZIP file containing multiple handwriting images. "
                "Each image is analysed and results are shown in a ranked table.")
        zip_file = st.file_uploader("Upload ZIP file of handwriting images", type=["zip"])
        if zip_file:
            if st.button("🔍 Analyse All Images", type="primary"):
                analyse_batch(zip_file, model, device, st.session_state.use_tta)

    with tab_history:
        show_history()

    # Footer
    st.markdown(
        "<div style='text-align:center;color:#9aafc4;font-size:.78rem;margin-top:2.5rem;"
        "border-top:1px solid #dce4ed;padding-top:1rem;'>"
        "For early screening support only &nbsp;·&nbsp; Not a substitute for professional "
        f"clinical assessment &nbsp;·&nbsp; v{config.VERSION}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
