"""AIRT-Engine Logging"""
import logging
import sys
from typing import Optional

def setup_logger(name: str = "airt", level: Optional[str] = None) -> logging.Logger:
    """Setup and return a logger instance."""
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    else:
        logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Global logger
log = setup_logger()