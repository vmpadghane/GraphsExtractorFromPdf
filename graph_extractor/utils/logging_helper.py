"""
Logging Helper Module

Sets up logging configuration for the application.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)