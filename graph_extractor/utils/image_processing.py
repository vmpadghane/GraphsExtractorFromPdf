"""
Image Processing Utilities

Handles image extraction, saving, and processing operations.
"""

import logging
from pathlib import Path
from typing import Tuple
from PIL import Image
import cv2
import numpy as np


class ImageProcessor:
    """Utility class for image processing operations."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_region(self, image: Image.Image, bbox: Tuple[float, ...]) -> Image.Image:
        """
        Extract region from image using bounding box.

        Args:
            image: PIL image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Cropped PIL image
        """
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def save_image(self, image: Image.Image, path: Path) -> None:
        """
        Save image to file.

        Args:
            image: PIL image
            path: Output path
        """
        format_type = self.config['image']['format']
        quality = self.config['image'].get('quality', 95)

        try:
            if format_type.upper() == 'PNG':
                image.save(path, format_type)
            else:
                image.save(path, format_type, quality=quality)

            self.logger.debug(f"Saved image to {path}")
        except Exception as e:
            self.logger.error(f"Error saving image {path}: {e}")
            raise

    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.

        Args:
            image: PIL image

        Returns:
            Processed PIL image
        """
        # Convert to grayscale
        gray = image.convert('L')

        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)

        # Convert to numpy for OpenCV operations
        img_array = np.array(enhanced)

        # Apply thresholding
        _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(thresh)