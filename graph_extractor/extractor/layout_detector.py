"""
Layout Detector Module

Uses OpenCV for basic layout detection as fallback.
"""

import logging
from typing import List, Dict, Any
from PIL import Image
import cv2
import numpy as np


class LayoutDetector:
    """Detects layout blocks using OpenCV."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect_layout(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect basic layout blocks using OpenCV.

        Args:
            image: PIL image

        Returns:
            List of detected blocks with coordinates and types
        """
        self.logger.debug("Detecting layout blocks with OpenCV")

        # Convert PIL to numpy array
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Simple thresholding to find text regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        height, width = gray.shape

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Filter small contours
            if w > 50 and h > 20:
                # Classify as Text or Figure based on size
                area = w * h
                self.logger.debug(f"Block: w={w}, h={h}, area={area}")
                if area > 5000:  # Lower threshold for figures
                    block_type = "Figure"
                else:
                    block_type = "Text"
                block_type = "Text"
                blocks.append({
                    'type': block_type,
                    'bbox': (x, y, x + w, y + h),
                    'score': 0.5  # Default confidence
                })

        self.logger.debug(f"Detected {len(blocks)} layout blocks")
        # Debug: print block types
        types = [block['type'] for block in blocks]
        self.logger.info(f"Block types found: {types}")
        return blocks