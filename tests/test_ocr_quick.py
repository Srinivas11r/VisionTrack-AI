"""Unit tests for OCR handler utility behavior."""

import numpy as np

from backend.utils.ocr_handler import OCRHandler


def test_clean_plate_text_keeps_alnum_uppercase():
    handler = OCRHandler(use_easyocr=False)
    assert handler.clean_plate_text(" ts-09 ab 1234 ") == "TS09AB1234"


def test_valid_plate_format_requires_letters_and_numbers():
    handler = OCRHandler(use_easyocr=False)
    assert handler.is_valid_plate("AB1234") is True
    assert handler.is_valid_plate("123456") is False
    assert handler.is_valid_plate("ABCDEF") is False


def test_detect_plate_returns_unknown_when_ocr_unavailable():
    handler = OCRHandler(use_easyocr=False)
    handler.ocr_available = False
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert handler.detect_plate(image) == "Unknown"
