"""
DotsOCR Client Library

Example:
    from api.client import DotsOCRClient
    
    client = DotsOCRClient("http://localhost:5000")
    result = client.ocr_file("document.pdf", detect_stamps=True)
"""

from .dots_ocr_client import DotsOCRClient, EnhancedDotsOCRClient

__all__ = ['DotsOCRClient', 'EnhancedDotsOCRClient']

