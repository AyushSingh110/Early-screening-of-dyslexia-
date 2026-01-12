# utils/ocr.py

import pytesseract
from PIL import Image

def extract_text_from_image(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image, lang="eng")
    return text.strip()
