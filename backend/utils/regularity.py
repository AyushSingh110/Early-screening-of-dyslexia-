"""
Handwriting regularity analysis using computer vision.

Irregular baseline, inconsistent letter sizes, and uneven word spacing
are well-documented visual markers of dyslexic handwriting.

Metrics computed:
  baseline_straightness : how straight the writing lines are (0=wavy, 1=straight)
  letter_size_variance  : coefficient of variation of letter heights (normalised)
  regularity_score      : combined score (0=irregular, 1=very regular)
  regularity_risk       : inverse of regularity_score (for fusion with other risks)
"""

import logging
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _baseline_straightness(gray: np.ndarray) -> float:
    """
    Estimate how straight the writing baseline is using Hough line detection.

    Returns a score in [0, 1] where 1 = perfectly straight lines.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=80, minLineLength=40, maxLineGap=20,
    )

    if lines is None or len(lines) < 3:
        return 0.5  # neutral when not enough lines detected

    # Keep only near-horizontal lines (angle < 20°)
    y_mids = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 20 or angle > 160:
            y_mids.append((y1 + y2) / 2.0)

    if len(y_mids) < 3:
        return 0.5

    # Sort y-positions and measure how evenly spaced the text lines are.
    # Low variance in consecutive gaps → straight, evenly-spaced lines.
    y_sorted = sorted(y_mids)
    gaps = np.diff(y_sorted)
    if gaps.mean() < 1e-6:
        return 0.5

    cv_gaps = gaps.std() / gaps.mean()                    # coefficient of variation
    score = max(0.0, 1.0 - min(cv_gaps / 1.5, 1.0))     # normalise to [0,1]
    return round(float(score), 3)


def _letter_size_consistency(gray: np.ndarray) -> float:
    """
    Measure consistency of letter heights via connected-component analysis.

    Returns a score in [0, 1] where 1 = very consistent letter sizes.
    """
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    heights = []
    img_h = gray.shape[0]
    for i in range(1, num_labels):                         # skip background (label 0)
        h    = int(stats[i, cv2.CC_STAT_HEIGHT])
        w    = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Heuristic filter: discard noise (too small) and large blobs (words/lines)
        if (img_h * 0.02 < h < img_h * 0.15) and (3 < w < 120) and area > 25:
            heights.append(h)

    if len(heights) < 5:
        return 0.5  # not enough components

    cv_h  = np.std(heights) / (np.mean(heights) + 1e-6)  # coefficient of variation
    score = max(0.0, 1.0 - min(cv_h / 0.7, 1.0))
    return round(float(score), 3)


def analyze_regularity(image_np: np.ndarray) -> Dict:
    """
    Analyse handwriting regularity from a full-page image array.

    Parameters
    ----------
    image_np : H×W×3 uint8 RGB array (or grayscale H×W)

    Returns
    -------
    dict
      baseline_straightness  : [0, 1]  higher = straighter lines
      letter_size_variance   : [0, 1]  higher = MORE variance (bad)
      regularity_score       : [0, 1]  higher = more regular writing
      regularity_risk        : [0, 1]  higher = more irregular (risk signal)
    """
    try:
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()

        # Denoise before analysis
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        baseline   = _baseline_straightness(gray)
        size_cons  = _letter_size_consistency(gray)
        size_var   = round(1.0 - size_cons, 3)            # higher = more variance

        regularity_score = round(0.5 * baseline + 0.5 * size_cons, 3)
        regularity_risk  = round(1.0 - regularity_score, 3)

        logger.debug(
            "Regularity — baseline=%.2f  size_consistency=%.2f  risk=%.2f",
            baseline, size_cons, regularity_risk,
        )

        return {
            'baseline_straightness': baseline,
            'letter_size_variance':  size_var,
            'regularity_score':      regularity_score,
            'regularity_risk':       regularity_risk,
        }

    except Exception as exc:
        logger.warning("Regularity analysis failed: %s", exc)
        return {
            'baseline_straightness': 0.5,
            'letter_size_variance':  0.5,
            'regularity_score':      0.5,
            'regularity_risk':       0.5,
        }
