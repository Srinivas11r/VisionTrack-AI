"""
OCR handler for license plate detection.
Supports EasyOCR with fallback to Tesseract.
"""

import logging
import re

import cv2

logger = logging.getLogger("app.ocr")


class OCRHandler:
    """Handler for OCR operations, primarily license plate detection."""

    def __init__(self, use_easyocr=True):
        """
        Initialize OCR handler.

        Args:
            use_easyocr: If True, try EasyOCR first, else use Tesseract
        """
        self.reader = None
        self.ocr_available = False
        self.use_easyocr = use_easyocr

        if use_easyocr:
            try:
                import easyocr  # type: ignore

                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.ocr_available = True
                logger.info("EasyOCR initialized successfully")
            except ImportError:
                logger.warning("EasyOCR not installed. Install with: pip install easyocr")
                self._try_tesseract()
            except Exception as e:
                logger.warning("EasyOCR initialization failed: %s", e)
                self._try_tesseract()
        else:
            self._try_tesseract()

    def _try_tesseract(self):
        """Try to initialize Tesseract OCR as fallback."""
        try:
            import pytesseract

            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.reader = pytesseract
            self.ocr_available = True
            self.use_easyocr = False
            logger.info("Tesseract OCR initialized successfully")
        except ImportError:
            logger.warning("Pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            logger.warning("Tesseract not available: %s", e)
            logger.warning("Install Tesseract: https://github.com/tesseract-ocr/tesseract")

    def preprocess_plate_region(self, plate_image):
        """
        Preprocess license plate region for better OCR accuracy.

        Args:
            plate_image: Cropped plate region

        Returns:
            Preprocessed image
        """
        if plate_image is None or plate_image.size == 0:
            return None

        try:
            # Convert to grayscale
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image

            # Resize for better OCR (minimum height of 50 pixels)
            h, w = gray.shape
            if h < 50:
                scale = 50 / h
                new_w = int(w * scale)
                gray = cv2.resize(gray, (new_w, 50), interpolation=cv2.INTER_CUBIC)

            # Apply bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)

            # Adaptive thresholding for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Try both normal and inverted (some plates are dark on light)
            inverted = cv2.bitwise_not(binary)

            return binary, inverted

        except Exception as e:
            logger.debug("Preprocessing error: %s", e)
            return plate_image, None

    def detect_plate(self, plate_region, confidence_threshold=0.5):
        """
        Detect and extract license plate number from image region.

        Args:
            plate_region: Cropped vehicle region (likely containing plate)
            confidence_threshold: Minimum confidence for valid detection

        Returns:
            License plate string or "Unknown"
        """
        if not self.ocr_available:
            return "Unknown"

        if plate_region is None or plate_region.size == 0:
            return "Unknown"

        try:
            # Preprocess image
            processed = self.preprocess_plate_region(plate_region)
            if processed is None:
                return "Unknown"

            binary, inverted = processed if isinstance(processed, tuple) else (processed, None)

            # Try OCR on both versions
            texts = []

            if self.use_easyocr and self.reader:
                # EasyOCR
                for img in [binary, inverted] if inverted is not None else [binary]:
                    results = self.reader.readtext(img, detail=1)
                    for bbox, text, conf in results:
                        if conf >= confidence_threshold:
                            texts.append((text, conf))
            else:
                # Tesseract
                import pytesseract

                for img in [binary, inverted] if inverted is not None else [binary]:
                    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    text = pytesseract.image_to_string(img, config=config).strip()
                    if text:
                        texts.append((text, 0.8))  # Tesseract doesn't provide confidence easily

            # Filter and format results
            if texts:
                # Sort by confidence
                texts.sort(key=lambda x: x[1], reverse=True)

                for text, conf in texts:
                    # Clean the text
                    cleaned = self.clean_plate_text(text)

                    # Validate plate format (basic check)
                    if self.is_valid_plate(cleaned):
                        logger.debug("License plate detected: %s (confidence: %.2f)", cleaned, conf)
                        return cleaned

            return "Unknown"

        except Exception as e:
            logger.debug("OCR detection error: %s", e)
            return "Unknown"

    def clean_plate_text(self, text):
        """
        Clean and format plate text.

        Args:
            text: Raw OCR output

        Returns:
            Cleaned plate string
        """
        # Remove spaces and special characters
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned

    def is_valid_plate(self, text):
        """
        Basic validation of license plate format.

        Args:
            text: Plate text

        Returns:
            True if valid format
        """
        if not text or len(text) < 3:
            return False

        # Should have both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)

        # Length should be reasonable (3-10 characters)
        if 3 <= len(text) <= 10 and has_letter and has_number:
            return True

        return False
