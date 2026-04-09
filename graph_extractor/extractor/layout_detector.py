"""
Layout Detector Module

Uses LayoutParser to detect layout blocks in images.
"""

import logging
from typing import List, Dict, Any
from PIL import Image
import layoutparser as lp
import numpy as np


class LayoutDetector:
    """Detects layout blocks using LayoutParser."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize LayoutParser model
        model_path = config['layout']['model_path']
        self.model = lp.Detectron2LayoutModel(
            model_path,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", config['layout']['threshold']],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )

    def detect_layout(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect layout blocks in the image.

        Args:
            image: PIL image

        Returns:
            List of detected blocks with coordinates and types
        """
        self.logger.debug("Detecting layout blocks")

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Detect layout
        layout = self.model.detect(img_array)

        blocks = []
        for block in layout:
            blocks.append({
                'type': block.type,
                'bbox': block.coordinates,  # (x1, y1, x2, y2)
                'score': block.score
            })

        self.logger.debug(f"Detected {len(blocks)} layout blocks")
        return blocks