"""
PDF report generation for the Dyslexia Screening System.

Requires: pip install reportlab
Generates a professional A4 report with risk scores, tables, and disclaimer.
"""

import io
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _risk_color(label: str):
    from reportlab.lib import colors
    return {
        'HIGH RISK':     colors.HexColor('#dc2626'),
        'MODERATE RISK': colors.HexColor('#d97706'),
        'LOW RISK':      colors.HexColor('#16a34a'),
    }.get(label, colors.grey)


def _table_style(header_color):
    from reportlab.platypus import TableStyle
    from reportlab.lib import colors
    return TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), header_color),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f8f9fc')]),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf_report(
    filename: str,
    results: Dict,
    handwriting_risk: float,
    language_risk: Optional[float],
    final_risk: float,
    risk_label: str,
    reversal_info: Optional[Dict] = None,
    regularity_info: Optional[Dict] = None,
) -> bytes:
    """
    Generate a professional A4 PDF screening report.

    Parameters
    ----------
    filename         : original uploaded filename (shown in header)
    results          : dict from run_inference() (patch stats)
    handwriting_risk : vision-model risk score [0,1]
    language_risk    : linguistic risk score [0,1] or None
    final_risk       : fused final score [0,1]
    risk_label       : 'LOW RISK' / 'MODERATE RISK' / 'HIGH RISK'
    reversal_info    : output of detect_reversals()
    regularity_info  : output of analyze_regularity()

    Returns
    -------
    bytes — raw PDF content, ready to pass to st.download_button
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.enums import TA_CENTER
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable,
        )
    except ImportError:
        raise ImportError(
            "reportlab is required for PDF reports.\n"
            "Install it with:  pip install reportlab"
        )

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=22 * mm,   bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()

    # ── Custom paragraph styles ──────────────────────────────────────────────
    H1 = ParagraphStyle('H1', parent=styles['Title'],
                        fontSize=20, textColor=colors.HexColor('#1a1a2e'),
                        spaceAfter=4)
    SUB = ParagraphStyle('Sub', parent=styles['Normal'],
                         fontSize=9, textColor=colors.grey, spaceAfter=14)
    SEC = ParagraphStyle('Sec', parent=styles['Heading2'],
                         fontSize=12, textColor=colors.HexColor('#0f3460'),
                         spaceBefore=12, spaceAfter=6)
    BODY = ParagraphStyle('Body2', parent=styles['Normal'],
                          fontSize=10, leading=14, spaceAfter=4)
    DISC = ParagraphStyle('Disc', parent=styles['Normal'],
                          fontSize=8, textColor=colors.HexColor('#374151'),
                          leading=12)

    NAVY   = colors.HexColor('#1a1a2e')
    BLUE   = colors.HexColor('#0f3460')
    r_col  = _risk_color(risk_label)
    elems  = []

    # ── Title block ──────────────────────────────────────────────────────────
    elems.append(Paragraph("Dyslexia Screening Report", H1))
    elems.append(Paragraph(
        f"File: <b>{filename}</b> &nbsp;|&nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; "
        "AI-Assisted Tool — Not a Medical Diagnosis",
        SUB,
    ))
    elems.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=10))

    # ── Risk verdict banner ──────────────────────────────────────────────────
    verdict_data = [[
        Paragraph(
            f"<b>{risk_label}</b>",
            ParagraphStyle('V', fontSize=20, textColor=r_col, alignment=TA_CENTER),
        ),
        Paragraph(
            f"Final Risk Score<br/><b>{final_risk:.1%}</b>",
            ParagraphStyle('S', fontSize=14, alignment=TA_CENTER,
                           textColor=colors.HexColor('#1a1a2e')),
        ),
    ]]
    v_table = Table(verdict_data, colWidths=['50%', '50%'])
    v_table.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
        ('BOX',          (0, 0), (-1, -1), 1.5, colors.HexColor('#e5e7eb')),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',   (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 16),
    ]))
    elems.append(v_table)
    elems.append(Spacer(1, 10))

    # ── Risk score breakdown ─────────────────────────────────────────────────
    elems.append(Paragraph("Risk Score Breakdown", SEC))
    lang_str = f"{language_risk:.1%}" if language_risk is not None else "N/A (text too short)"
    score_rows = [
        ["Component",                 "Score", "Weight"],
        ["Handwriting (Vision Model)",f"{handwriting_risk:.1%}", "60%"],
        ["Language (Linguistics)",    lang_str,                  "25%"],
        ["Final Fused Score",         f"{final_risk:.1%}",       "—"],
    ]
    st = Table(score_rows, colWidths=['55%', '25%', '20%'])
    st.setStyle(_table_style(BLUE))
    st.setStyle(TableStyle([
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#eff6ff')),
        ('FONTNAME',   (0, -1), (-1, -1), 'Helvetica-Bold'),
    ]))
    elems.append(st)
    elems.append(Spacer(1, 8))

    # ── Patch statistics ────────────────────────────────────────────────────
    elems.append(Paragraph("Handwriting Patch Analysis", SEC))
    patch_rows = [
        ["Metric",                    "Value"],
        ["Total patches analysed",    str(results['total_patches'])],
        ["Dyslexia-like patches",
         f"{results['dyslexic_patches']} ({results['risk_score']:.1%})"],
        ["Typical patches",           str(results['normal_patches'])],
        ["Mean patch confidence",     f"{results['avg_confidence']:.1%}"],
        ["Highest patch confidence",  f"{results['max_confidence']:.1%}"],
        ["Lowest patch confidence",   f"{results['min_confidence']:.1%}"],
    ]
    pt = Table(patch_rows, colWidths=['60%', '40%'])
    pt.setStyle(_table_style(NAVY))
    elems.append(pt)
    elems.append(Spacer(1, 8))

    # ── Letter reversal analysis ─────────────────────────────────────────────
    if reversal_info and reversal_info.get('total_words', 0) > 0:
        elems.append(Paragraph("Letter Reversal Analysis", SEC))
        rev_rows = [
            ["Metric",                "Value"],
            ["Words analysed",        str(reversal_info['total_words'])],
            ["Reversal errors found", str(reversal_info['reversal_count'])],
            ["Reversal error rate",   f"{reversal_info['reversal_ratio']:.1%}"],
        ]
        if reversal_info.get('reversal_words'):
            examples = ', '.join(
                f"'{w}' → '{c}'"
                for w, c in reversal_info['reversal_words'][:5]
            )
            rev_rows.append(["Example reversals", examples])
        rt = Table(rev_rows, colWidths=['45%', '55%'])
        rt.setStyle(_table_style(BLUE))
        elems.append(rt)
        elems.append(Spacer(1, 8))

    # ── Writing regularity ──────────────────────────────────────────────────
    if regularity_info:
        elems.append(Paragraph("Writing Regularity Analysis", SEC))
        reg_rows = [
            ["Metric",                "Score", "Interpretation"],
            ["Baseline Straightness",
             f"{regularity_info['baseline_straightness']:.2f}",
             "Higher = straighter writing lines"],
            ["Letter Size Variance",
             f"{regularity_info['letter_size_variance']:.2f}",
             "Higher = more inconsistent letter sizes"],
            ["Overall Regularity Risk",
             f"{regularity_info['regularity_risk']:.2f}",
             "Higher = more irregular writing (risk signal)"],
        ]
        regt = Table(reg_rows, colWidths=['38%', '18%', '44%'])
        regt.setStyle(_table_style(NAVY))
        elems.append(regt)
        elems.append(Spacer(1, 8))

    # ── Recommendation ───────────────────────────────────────────────────────
    elems.append(Paragraph("Screening Recommendation", SEC))
    recs = {
        'HIGH RISK':
            "The analysis detected significant patterns associated with dyslexia across "
            "multiple indicators. <b>A comprehensive assessment by a qualified educational "
            "psychologist is strongly recommended.</b>",
        'MODERATE RISK':
            "Some dyslexia-related patterns were detected. Consider arranging a follow-up "
            "assessment with an educational specialist to explore further.",
        'LOW RISK':
            "No strong dyslexia indicators were detected. Continue to monitor writing "
            "development and consult a professional if concerns arise.",
    }
    elems.append(Paragraph(recs.get(risk_label, ""), BODY))
    elems.append(Spacer(1, 14))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    elems.append(HRFlowable(width="100%", thickness=1,
                            color=colors.lightgrey, spaceAfter=8))
    elems.append(Paragraph(
        "<b>Disclaimer:</b> This report was generated by an AI-based early screening tool "
        "and does <b>not</b> constitute a medical or psychological diagnosis. The system "
        "identifies handwriting and linguistic patterns that may be associated with dyslexia "
        "and is intended to support, not replace, professional evaluation. All findings should "
        "be interpreted by a qualified educational psychologist or medical professional. "
        "A positive screening result indicates the need for further professional assessment.",
        DISC,
    ))

    doc.build(elems)
    return buf.getvalue()
