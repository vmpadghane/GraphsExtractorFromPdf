"""
Graph Detector Module

Detects graph regions using aspect ratio, edge density, and axis detection.
"""

import logging
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image


class GraphDetector:
    """Detects potential graph regions in images."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def detect_graphs(self, image: Image.Image, layout_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect graph regions from layout blocks.

        Args:
            image: PIL image
            layout_blocks: Detected layout blocks

        Returns:
            List of graph regions with coordinates
        """
        self.logger.debug("Detecting graph regions")

        img_array = np.array(image)
        graphs = []

        # Focus on Figure blocks from layout detection
        figure_blocks = [block for block in layout_blocks if block['type'] == 'Figure']

        for block in figure_blocks:
            bbox = block['bbox']
            region = self._extract_region(img_array, bbox)

            if self._is_graph_region(region):
                graphs.append({
                    'bbox': bbox,
                    'confidence': block['score']
                })

        self.logger.info(f"Detected {len(graphs)} potential graph regions")
        return graphs

    def _extract_region(self, image: np.ndarray, bbox: Tuple[float, ...]) -> np.ndarray:
        """Extract region from image using bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    def _is_graph_region(self, region: np.ndarray) -> bool:
        """
        Check if region is likely a graph using heuristics.

        Args:
            region: Image region as numpy array

        Returns:
            True if likely a graph
        """
        if region.size == 0:
            return False

        # Check aspect ratio
        h, w = region.shape[:2]
        aspect_ratio = w / h if h > 0 else 0

        min_ar = self.config['graph']['min_aspect_ratio']
        max_ar = self.config['graph']['max_aspect_ratio']

        if not (min_ar <= aspect_ratio <= max_ar):
            return False

        # Check edge density
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        if edge_density < self.config['graph']['min_edge_density']:
            return False

        # Check for axes (simple line detection)
        if not self._has_axes(region):
            return False

        return True

    def _has_axes(self, region: np.ndarray) -> bool:
        """
        Check for axis lines in the region.

        Args:
            region: Image region

        Returns:
            True if axes detected
        """
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            gray,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return False

        # Check for horizontal/vertical lines that could be axes
        h, w = gray.shape
        axis_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly horizontal or vertical
            if abs(x2 - x1) > abs(y2 - y1):  # horizontal
                if min(y1, y2) < h * 0.1 or max(y1, y2) > h * 0.9:
                    axis_lines += 1
            else:  # vertical
                if min(x1, x2) < w * 0.1 or max(x1, x2) > w * 0.9:
                    axis_lines += 1

        return axis_lines >= 2  # At least x and y axes