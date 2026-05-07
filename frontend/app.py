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
    /* ── Page background ── */
    .stApp { background-color: #f4f6f9; }

    /* ── Header ── */
    .main-header {
        background: #ffffff;
        border-top: 5px solid #2563eb;
        border-radius: 0 0 12px 12px;
        padding: 1.8rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(37,99,235,.10);
    }
    .main-header h1 { color:#1e3a5f; font-size:1.85rem; margin:0 0 .4rem 0; }
    .main-header p  { color:#4b6a8a; font-size:.95rem; margin:0; }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff; border-radius: 10px;
        padding: 1.1rem 1.4rem;
        box-shadow: 0 1px 6px rgba(0,0,0,.07);
        border-left: 4px solid #2563eb;
        margin-bottom: .8rem;
    }
    .metric-card .label { font-size:.75rem; color:#64748b;
        text-transform:uppercase; letter-spacing:.06em; margin-bottom:.3rem; }
    .metric-card .value { font-size:1.6rem; font-weight:700; color:#1e3a5f; }

    /* ── Risk banners ── */
    .risk-banner { border-radius:10px; padding:1.4rem 2rem; margin:1rem 0; text-align:center; }
    .risk-high     { background:#fff5f5; border:2px solid #f87171; }
    .risk-moderate { background:#fffbf0; border:2px solid #fbbf24; }
    .risk-low      { background:#f0fdf6; border:2px solid #34d399; }
    .risk-banner .risk-label { font-size:1.5rem; font-weight:800; }
    .risk-high     .risk-label { color:#b91c1c; }
    .risk-moderate .risk-label { color:#b45309; }
    .risk-low      .risk-label { color:#059669; }
    .risk-banner .risk-msg { font-size:.93rem; margin-top:.4rem; color:#374151; }

    /* ── Progress bar ── */
    .score-bar-wrap { background:#e2e8f0; border-radius:999px; height:13px;
        margin:.5rem 0 .2rem; overflow:hidden; }
    .score-bar-fill { height:100%; border-radius:999px; }

    /* ── Section titles ── */
    .section-title { font-size:1.05rem; font-weight:700; color:#1e3a5f;
        border-bottom:2px solid #dde3ea; padding-bottom:.4rem; margin:1.5rem 0 .9rem; }

    /* ── Sidebar — light academic grey ── */
    section[data-testid="stSidebar"] { background-color: #eef2f7 !important; }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }
    section[data-testid="stSidebar"] .stMarkdown a { color:#2563eb !important; }

    /* ── Disclaimer ── */
    .disclaimer { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
        padding:1rem 1.2rem; font-size:.85rem; color:#1e40af; margin-top:1.5rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    :root {
        --ink: #eef6ff;
        --muted: #a9b7c9;
        --line: #263548;
        --surface: #111c2d;
        --surface-soft: #16243a;
        --blue: #60a5fa;
        --teal: #2dd4bf;
        --green: #34d399;
        --amber: #fbbf24;
        --red: #f87171;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(45,212,191,.18), transparent 340px),
            linear-gradient(180deg, #07111f 0%, #0b1423 46%, #101827 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2.4rem;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span {
        color: inherit;
        letter-spacing: 0;
    }

    .main-header {
        background:
            linear-gradient(135deg, rgba(17,28,45,.98), rgba(15,40,60,.94));
        border: 1px solid rgba(96,165,250,.18);
        border-left: 6px solid var(--teal);
        border-radius: 10px;
        padding: 1.55rem 1.8rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 18px 44px rgba(0,0,0,.28);
    }
    .main-header h1 {
        color: #f8fbff;
        font-size: 1.95rem;
        line-height: 1.15;
        margin: 0 0 .45rem 0;
        font-weight: 800;
    }
    .main-header p {
        color: var(--muted);
        font-size: .98rem;
        margin: 0;
        max-width: 820px;
    }

    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        color: var(--muted);
        font-weight: 650;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--teal);
        background: rgba(45,212,191,.11);
    }

    div[data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #3b5876;
        border-radius: 10px;
        padding: .9rem;
        box-shadow: 0 10px 28px rgba(0,0,0,.22);
    }
    div[data-testid="stFileUploader"] section {
        background: #0d1a2b;
        border-color: #2a4058;
    }

    .metric-card {
        background: linear-gradient(180deg, #142238, #101b2d);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        border: 1px solid #263548;
        border-top: 4px solid var(--teal);
        box-shadow: 0 12px 30px rgba(0,0,0,.22);
        min-height: 104px;
    }
    .metric-card .label {
        font-size: .72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: .38rem;
        font-weight: 700;
    }
    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fbff;
    }

    .risk-banner {
        border-radius: 10px;
        padding: 1.35rem 1.65rem;
        margin: 1rem 0;
        text-align: left;
        box-shadow: 0 14px 34px rgba(0,0,0,.24);
    }
    .risk-high {
        background:#2a1218;
        border:1px solid rgba(248,113,113,.32);
        border-left:6px solid var(--red);
    }
    .risk-moderate {
        background:#2a2110;
        border:1px solid rgba(251,191,36,.30);
        border-left:6px solid var(--amber);
    }
    .risk-low {
        background:#10251d;
        border:1px solid rgba(52,211,153,.30);
        border-left:6px solid var(--green);
    }
    .risk-banner .risk-label { font-size:1.42rem; font-weight:850; }
    .risk-high .risk-label { color:var(--red); }
    .risk-moderate .risk-label { color:var(--amber); }
    .risk-low .risk-label { color:var(--green); }
    .risk-banner .risk-msg { color:#d7e1ee; }

    .score-bar-wrap {
        background:#1e2b3d;
        border-radius:999px;
        height:12px;
        margin:.55rem 0 .2rem;
        overflow:hidden;
        border: 1px solid rgba(148,163,184,.22);
    }
    .score-bar-fill {
        height:100%;
        border-radius:999px;
        transition: width .25s ease;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #f8fbff;
        border-bottom: 1px solid var(--line);
        padding-bottom: .45rem;
        margin: 1.35rem 0 .9rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #0a1321 0%, #101c2f 100%) !important;
        border-right: 1px solid #253348;
    }
    section[data-testid="stSidebar"] * { color: #d8e3f1 !important; }
    section[data-testid="stSidebar"] .stMarkdown a { color: var(--blue) !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        font-size: .92rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 750;
        border: 1px solid rgba(96,165,250,.20);
        box-shadow: 0 10px 24px rgba(0,0,0,.24);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, var(--teal));
        border: 0;
        color: white;
    }
    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid #263548;
        border-radius: 8px;
        box-shadow: 0 10px 24px rgba(0,0,0,.18);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #263548;
        border-radius: 8px;
        overflow: hidden;
    }
    .disclaimer {
        background:#0f2238;
        border:1px solid #25496d;
        border-left: 5px solid var(--blue);
        border-radius:8px;
        padding:1rem 1.15rem;
        font-size:.85rem;
        color:#dbeafe;
        margin-top:1.5rem;
    }

    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .main-header { padding: 1.2rem; }
        .main-header h1 { font-size: 1.55rem; }
        .metric-card { min-height: auto; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    :root {
        --ink: #172033;
        --muted: #5b677a;
        --line: #d9e1ea;
        --surface: #ffffff;
        --blue: #2563eb;
        --teal: #0f9f8f;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
    }

    .stApp {
        background:
            linear-gradient(180deg, #eef6ff 0%, #f8fbfd 250px, #f6f8fb 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2.4rem;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span {
        letter-spacing: 0;
    }

    .main-header {
        background:
            linear-gradient(135deg, rgba(255,255,255,.98), rgba(244,250,255,.98));
        border: 1px solid rgba(37, 99, 235, .14);
        border-left: 6px solid var(--blue);
        border-radius: 10px;
        padding: 1.55rem 1.8rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 14px 36px rgba(29, 78, 216, .10);
    }
    .main-header h1 {
        color: var(--ink);
        font-size: 1.95rem;
        line-height: 1.15;
        margin: 0 0 .45rem 0;
        font-weight: 800;
    }
    .main-header p {
        color: var(--muted);
        font-size: .98rem;
        margin: 0;
        max-width: 820px;
    }

    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        color: var(--muted);
        font-weight: 650;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--blue);
        background: rgba(37,99,235,.07);
    }

    div[data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #a8bfdc;
        border-radius: 10px;
        padding: .9rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .04);
    }
    div[data-testid="stFileUploader"] section {
        background: #f8fbff;
        border-color: #d7e5f5;
    }

    .metric-card {
        background: var(--surface);
        border-radius: 8px;
        padding: 1rem 1.15rem;
        border: 1px solid #e3eaf2;
        border-top: 4px solid var(--blue);
        box-shadow: 0 8px 24px rgba(15, 23, 42, .055);
        min-height: 104px;
    }
    .metric-card .label {
        font-size: .72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: .38rem;
        font-weight: 700;
    }
    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--ink);
    }

    .risk-banner {
        border-radius: 10px;
        padding: 1.35rem 1.65rem;
        margin: 1rem 0;
        text-align: left;
        box-shadow: 0 10px 28px rgba(15, 23, 42, .06);
    }
    .risk-high {
        background:#fff7f7;
        border:1px solid #fecaca;
        border-left:6px solid var(--red);
    }
    .risk-moderate {
        background:#fffbeb;
        border:1px solid #fde68a;
        border-left:6px solid var(--amber);
    }
    .risk-low {
        background:#f0fdf6;
        border:1px solid #bbf7d0;
        border-left:6px solid var(--green);
    }
    .risk-banner .risk-label { font-size:1.42rem; font-weight:850; }
    .risk-high .risk-label { color:var(--red); }
    .risk-moderate .risk-label { color:var(--amber); }
    .risk-low .risk-label { color:var(--green); }

    .score-bar-wrap {
        background:#e5edf5;
        border-radius:999px;
        height:12px;
        margin:.55rem 0 .2rem;
        overflow:hidden;
        border: 1px solid rgba(148,163,184,.35);
    }
    .score-bar-fill {
        height:100%;
        border-radius:999px;
        transition: width .25s ease;
    }

    .section-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: var(--ink);
        border-bottom: 1px solid var(--line);
        padding-bottom: .45rem;
        margin: 1.35rem 0 .9rem;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #f8fbff 0%, #eef4f8 100%) !important;
        border-right: 1px solid #dbe6ef;
    }
    section[data-testid="stSidebar"] * { color: #1e293b !important; }
    section[data-testid="stSidebar"] .stMarkdown a { color: var(--blue) !important; }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        font-size: .92rem;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 750;
        border: 1px solid rgba(37, 99, 235, .2);
        box-shadow: 0 8px 18px rgba(37, 99, 235, .12);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--blue), var(--teal));
        border: 0;
    }
    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid #e3eaf2;
        border-radius: 8px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, .035);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e3eaf2;
        border-radius: 8px;
        overflow: hidden;
    }
    .disclaimer {
        background:#eff6ff;
        border:1px solid #bfdbfe;
        border-left: 5px solid var(--blue);
        border-radius:8px;
        padding:1rem 1.15rem;
        font-size:.85rem;
        color:#1e40af;
        margin-top:1.5rem;
    }

    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .main-header { padding: 1.2rem; }
        .main-header h1 { font-size: 1.55rem; }
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
    col = {"high": "#ef4444", "moderate": "#f59e0b", "low": "#22c55e"}.get(level, "#6b7280")
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
        "<h1>🧠 AI-Based Dyslexia Screening</h1>"
        "<p>Upload a handwriting sample for instant AI-powered dyslexia risk assessment — "
        "combining vision analysis, OCR, letter reversal detection, and writing regularity.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.markdown("Fine-tuned **ResNet-50** trained on **128,902** handwriting images.")
        st.markdown("---")
        st.markdown("**Model performance**\n"
                    "- Test accuracy : **75.9%**\n"
                    "- ROC-AUC       : **84.9%**\n"
                    "- Dyslexic recall: **79.9%**\n"
                    f"- Device        : {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.markdown("---")
        st.markdown("**Settings**")
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=False,
                                     help="Show attention heatmaps (+processing time)")
        use_tta        = st.checkbox("Enable TTA", value=False,
                                     help="Test-time augmentation — 5× passes, +1-2% accuracy")
        st.session_state.enable_gradcam = enable_gradcam
        st.session_state.use_tta        = use_tta

        if use_tta:
            st.success("TTA enabled — averaging 5 augmented predictions per patch")
        st.markdown("---")
        st.markdown("**Upload guidelines**\n"
                    "- Clear, well-lit scan\n"
                    "- Full page preferred\n"
                    f"- Max {MAX_FILE_MB} MB · PNG/JPG/JPEG")

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
            <div style="text-align:center;padding:3rem 1rem;color:#64748b;">
                <div style="font-size:3rem;">📄</div>
                <div style="font-size:1rem;margin-top:.5rem;color:#334155;">
                    Upload a handwriting image above to begin screening
                </div>
                <div style="font-size:.85rem;margin-top:.3rem;color:#64748b;">
                    Supports full-page scans · PNG, JPG, JPEG · up to 10 MB
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
        "<div style='text-align:center;color:#64748b;font-size:.8rem;margin-top:2rem;"
        "border-top:1px solid #dde3ea;padding-top:1rem;'>"
        "For educational and screening purposes only &nbsp;·&nbsp; "
        f"Not a substitute for professional diagnosis &nbsp;·&nbsp; v{config.VERSION}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
