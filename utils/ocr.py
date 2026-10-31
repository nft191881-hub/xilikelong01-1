"""Image -> text via Tesseract.

System requirement:  apt-get install tesseract-ocr tesseract-ocr-chi-sim
Python requirement:  pytesseract, Pillow
"""
from functools import lru_cache

from PIL import Image, ImageOps

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

MIN_WIDTH = 1200  # upscale small screenshots; Tesseract does better on larger text


@lru_cache(maxsize=1)
def _pick_lang() -> str:
    """Use Chinese+English if the Chinese model is installed, otherwise English."""
    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    if {"chi_sim", "eng"} <= langs:
        return "chi_sim+eng"
    return "eng"


def _preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")
    if img.width < MIN_WIDTH:
        scale = MIN_WIDTH / img.width
        img = img.resize((MIN_WIDTH, int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def extract_text_from_image(path: str) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract 未安装：pip install pytesseract Pillow")
    try:
        with Image.open(path) as img:
            prepared = _preprocess(img)
        text = pytesseract.image_to_string(prepared, lang=_pick_lang(), config="--psm 6")
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError("未找到 Tesseract：请安装 tesseract-ocr（Linux: apt-get install tesseract-ocr tesseract-ocr-chi-sim）")
    return text.strip()
