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

        # Try pdf2image first
        try:
            dpi = self.config['pdf'].get('dpi', 300)
            poppler_path = self.config['pdf'].get('poppler_path')

            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path
            )
            self.logger.info(f"Converted {len(images)} pages to images")
            return images
        except Exception as e:
            self.logger.warning(f"pdf2image failed: {e}, trying alternative method")

            # Fallback: use pypdf to extract images
            try:
                with open(pdf_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    images = []

                    for page_num in range(len(pdf.pages)):
                        page = pdf.pages[page_num]

                        # Try to extract images from the page
                        for image_key in page.images:
                            image_obj = page.images[image_key]
                            image_data = image_obj.data

                            # Convert to PIL Image
                            from io import BytesIO
                            img = Image.open(BytesIO(image_data))
                            images.append(img)

                    if images:
                        self.logger.info(f"Extracted {len(images)} images from PDF")
                        return images
                    else:
                        raise Exception("No images found in PDF")

            except Exception as e2:
                self.logger.error(f"Alternative method also failed: {e2}")
                raise Exception(f"Could not load PDF: {e} and {e2}")

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