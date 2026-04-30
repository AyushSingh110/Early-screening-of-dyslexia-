"""
OCR utilities for extracting text from handwriting images.

Preprocessing steps (grayscale → denoise → adaptive binarization) are
applied before Tesseract runs; this significantly improves recognition
accuracy on low-contrast or unevenly-lit handwriting scans.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Tesseract config: OEM 3 = LSTM engine, PSM 6 = uniform block of text
_TESS_CONFIG = "--oem 3 --psm 6"


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Convert a PIL image to a binarized, denoised grayscale image
    that Tesseract can read more accurately.
    """
    img = np.array(image.convert("RGB"))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Light denoising (preserve ink strokes)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding handles uneven lighting / scanner artifacts
    binarized = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    return Image.fromarray(binarized)


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract plain text from a handwriting image using Tesseract OCR.

    The image is preprocessed (grayscale → denoise → binarize) before
    being passed to Tesseract to improve recognition quality.

    Returns an empty string on failure rather than raising, so the app
    can degrade gracefully to handwriting-only analysis.
    """
    try:
        preprocessed = _preprocess_for_ocr(image)
        text = pytesseract.image_to_string(
            preprocessed, lang="eng", config=_TESS_CONFIG
        )
        return text.strip()
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""
