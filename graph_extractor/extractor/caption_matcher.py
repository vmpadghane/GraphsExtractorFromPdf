"""
Caption Matcher Module

Extracts OCR text and finds caption candidates near graph regions.
"""

import logging
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

# OCR imports - conditional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class CaptionMatcher:
    """Matches captions to graph regions using OCR and spatial logic."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize OCR engine
        ocr_engine = config['ocr']['engine']
        if ocr_engine == 'tesseract' and TESSERACT_AVAILABLE:
            self.ocr_engine = 'tesseract'
            pytesseract.pytesseract.tesseract_cmd = config['ocr'].get('tesseract_cmd', 'tesseract')
        elif ocr_engine == 'paddle' and PADDLE_AVAILABLE:
            self.ocr_engine = 'paddle'
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        else:
            self.logger.warning(f"OCR engine {ocr_engine} not available, using mock OCR")
            self.ocr_engine = 'mock'

    def extract_ocr(self, image: Image.Image, layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract OCR text from text blocks.

        Args:
            image: PIL image
            layout_blocks: Layout blocks

        Returns:
            List of OCR text blocks with coordinates
        """
        self.logger.debug("Extracting OCR text")

        img_array = np.array(image)
        ocr_texts = []

        # Focus on Text blocks
        text_blocks = [block for block in layout_blocks if block['type'] == 'Text']

        for block in text_blocks:
            bbox = block['bbox']
            region = self._extract_region(img_array, bbox)

            text = self._perform_ocr(region)
            if text.strip():
                ocr_texts.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': block['score']
                })

        self.logger.debug(f"Extracted {len(ocr_texts)} OCR text blocks")
        return ocr_texts

    def find_candidates(self, graph_bbox: Tuple[float, ...], ocr_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find candidate captions near the graph region.

        Args:
            graph_bbox: Graph bounding box
            ocr_texts: OCR text blocks

        Returns:
            List of candidate captions with distances
        """
        candidates = []
        max_distance = self.config['caption']['max_distance']
        keywords = self.config['caption']['keywords']

        gx1, gy1, gx2, gy2 = graph_bbox

        for ocr_block in ocr_texts:
            tx1, ty1, tx2, ty2 = ocr_block['bbox']

            # Calculate distance (simplified - vertical distance if overlapping horizontally)
            if tx1 < gx2 and tx2 > gx1:  # Horizontal overlap
                distance = min(abs(ty1 - gy2), abs(ty2 - gy1))
            else:
                # Euclidean distance between centers
                g_center = ((gx1 + gx2) / 2, (gy1 + gy2) / 2)
                t_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
                distance = np.sqrt((g_center[0] - t_center[0])**2 + (g_center[1] - t_center[1])**2)

            if distance <= max_distance:
                # Check for keywords
                text_lower = ocr_block['text'].lower()
                has_keyword = any(keyword.lower() in text_lower for keyword in keywords)

                if has_keyword:
                    candidates.append({
                        'text': ocr_block['text'],
                        'distance': distance,
                        'bbox': ocr_block['bbox'],
                        'confidence': ocr_block['confidence']
                    })

        # Sort by distance
        candidates.sort(key=lambda x: x['distance'])
        return candidates[:5]  # Top 5 candidates

    def _extract_region(self, image: np.ndarray, bbox: Tuple[float, ...]) -> np.ndarray:
        """Extract region from image."""
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    def _perform_ocr(self, region: np.ndarray) -> str:
        """Perform OCR on image region."""
        if self.ocr_engine == 'tesseract':
            config = self.config['ocr']['config']
            return pytesseract.image_to_string(region, config=config)
        elif self.ocr_engine == 'paddle':
            result = self.ocr.ocr(region, cls=True)
            if result and result[0]:
                return ' '.join([line[1][0] for line in result[0]])
            return ""
        elif self.ocr_engine == 'mock':
            # Mock OCR - return some sample text
            return "Sample caption text for graph"
        else:
            return ""