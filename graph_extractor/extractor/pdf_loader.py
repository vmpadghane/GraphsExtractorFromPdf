"""
PDF Loader Module

Handles loading PDF pages as images using pdf2image and pypdf.
"""

import logging
from pathlib import Path
from typing import List
import pypdf
from pdf2image import convert_from_path
from PIL import Image


class PDFLoader:
    """Loads PDF pages as images."""

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_pdf(self, pdf_path: str) -> List[Image.Image]:
        """
        Load PDF pages as PIL images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL images, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.logger.info(f"Loading PDF: {pdf_path}")

        # Use pdf2image for high-quality image conversion
        dpi = self.config['pdf'].get('dpi', 300)
        poppler_path = self.config['pdf'].get('poppler_path')

        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path
            )
            self.logger.info(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {e}")
            raise

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get number of pages in PDF using pypdf.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        with open(pdf_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            return len(pdf.pages)