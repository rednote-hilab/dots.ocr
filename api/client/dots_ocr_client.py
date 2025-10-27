"""
DotsOCR Client for external projects integration

Example usage:
    from api.client.dots_ocr_client import DotsOCRClient
    
    client = DotsOCRClient("http://localhost:5000")
    result = client.ocr_file("document.pdf", detect_stamps=True)
    
    print(f"Text: {result['text']}")
    print(f"Stamps found: {len(result['stamps'])}")
"""

import requests
import base64
import json
from typing import Dict, List, Optional
from pathlib import Path


class DotsOCRClient:
    """Client for dots.ocr API"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        """
        Initialize client
        
        Args:
            api_url: Base URL of dots.ocr API
        """
        self.api_url = api_url.rstrip('/')
        self._check_health()
    
    def _check_health(self):
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Connected to {self.api_url}")
            else:
                print(f"⚠️  API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API: {e}")
            raise
    
    def ocr_file(
        self, 
        file_path: str, 
        detect_stamps: bool = True,
        prompt_type: str = "prompt_layout_all_en"
    ) -> Dict:
        """
        OCR file and extract information
        
        Args:
            file_path: Path to file (PDF/Image)
            detect_stamps: Whether to detect stamps/seals
            prompt_type: Type of OCR prompt to use
        
        Returns:
            {
                'layout': List[dict],  # All elements with bboxes
                'text': str,           # Full text
                'stamps': List[dict],  # Detected stamps
                'tables': List[dict],  # Tables
                'formulas': List[dict] # Formulas
            }
        """
        # Read file
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Send request
        response = requests.post(
            f"{self.api_url}/ocr",
            json={
                "image": image_b64,
                "image_format": "base64",
                "prompt_type": prompt_type
            },
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Parse response
        parsed = self._parse_response(result)
        
        # Detect stamps if needed
        if detect_stamps:
            parsed['stamps'] = self._detect_stamps(parsed['layout'])
        
        return parsed
    
    def _parse_response(self, response: Dict) -> Dict:
        """Parse JSON response from API"""
        try:
            data = json.loads(response['response'])
        except (json.JSONDecodeError, KeyError):
            raise Exception("Invalid API response format")
        
        layout_elements = []
        text_parts = []
        tables = []
        formulas = []
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            data = [data]
        
        for element in data:
            bbox = element.get('bbox', [])
            category = element.get('category', '')
            text = element.get('text', '')
            
            layout_elements.append({
                'bbox': bbox,  # [x1, y1, x2, y2]
                'category': category,
                'text': text
            })
            
            # Extract tables
            if category == 'Table':
                tables.append({
                    'bbox': bbox,
                    'html': text
                })
            
            # Extract formulas
            elif category == 'Formula':
                formulas.append({
                    'bbox': bbox,
                    'latex': text
                })
            
            # Collect text (skip headers/footers/pictures)
            elif category not in ['Picture', 'Page-header', 'Page-footer']:
                if text.strip():
                    text_parts.append(text)
        
        return {
            'layout': layout_elements,
            'text': '\n'.join(text_parts),
            'tables': tables,
            'formulas': formulas,
            'stamps': []  # Will be filled if detect_stamps=True
        }
    
    def _detect_stamps(self, layout_elements: List[Dict]) -> List[Dict]:
        """
        Detect stamps/seals from layout elements
        
        Heuristics:
        - Circular/oval shapes (aspect ratio ~ 1:1)
        - Contains stamp keywords
        - Small size (< 300px)
        """
        stamps = []
        
        # Keywords for stamp detection
        stamp_keywords = [
            # Russian
            'ПЕЧАТЬ', 'УТВЕРЖДЕНО', 'СОГЛАСОВАНО', 'М.П.', 'ПЕЧ',
            # English
            'STAMP', 'SEAL', 'APPROVED', 'CERTIFIED', 
            # Chinese
            '印章', '公章', '盖章',
        ]
        
        for element in layout_elements:
            text = element.get('text', '').upper()
            bbox = element['bbox']
            
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                continue
            
            aspect_ratio = width / height
            
            # Check for stamp keywords
            has_keyword = any(keyword in text for keyword in stamp_keywords)
            
            # Round/square shape (0.7 < ratio < 1.3)
            is_round = 0.7 < aspect_ratio < 1.3
            
            # Small size (stamps are usually < 300px)
            is_small = width < 300 and height < 300
            
            # Detect stamp
            if (has_keyword and is_small) or (is_round and is_small):
                confidence = 'high' if has_keyword else 'medium'
                
                stamps.append({
                    'bbox': bbox,
                    'text': element['text'],
                    'confidence': confidence,
                    'category': element.get('category', '')
                })
        
        return stamps


class EnhancedDotsOCRClient(DotsOCRClient):
    """
    Enhanced client with visual stamp detection
    Requires: opencv-python, numpy
    """
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        super().__init__(api_url)
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
        except ImportError:
            raise ImportError(
                "Enhanced client requires: pip install opencv-python numpy"
            )
    
    def ocr_file(
        self, 
        file_path: str, 
        detect_stamps: bool = True,
        visual_detection: bool = True,
        prompt_type: str = "prompt_layout_all_en"
    ) -> Dict:
        """
        OCR with enhanced stamp detection
        
        Args:
            file_path: Path to file
            detect_stamps: Use text-based detection
            visual_detection: Use color-based detection
            prompt_type: OCR prompt type
        """
        # Get OCR result
        result = super().ocr_file(file_path, detect_stamps=False, prompt_type=prompt_type)
        
        if detect_stamps:
            # Text-based detection
            text_stamps = self._detect_stamps(result['layout'])
            
            # Visual detection
            if visual_detection:
                visual_stamps = self._detect_stamps_visual(file_path)
                # Merge results
                result['stamps'] = self._merge_detections(text_stamps, visual_stamps)
            else:
                result['stamps'] = text_stamps
        
        return result
    
    def _detect_stamps_visual(self, image_path: str) -> List[Dict]:
        """Detect stamps by color (red/blue)"""
        img = self.cv2.imread(str(image_path))
        if img is None:
            return []
        
        img_hsv = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2HSV)
        
        stamps = []
        
        # Red color ranges
        lower_red1 = self.np.array([0, 100, 100])
        upper_red1 = self.np.array([10, 255, 255])
        lower_red2 = self.np.array([160, 100, 100])
        upper_red2 = self.np.array([180, 255, 255])
        
        mask_red = self.cv2.inRange(img_hsv, lower_red1, upper_red1) | \
                   self.cv2.inRange(img_hsv, lower_red2, upper_red2)
        
        # Blue color range
        lower_blue = self.np.array([100, 100, 100])
        upper_blue = self.np.array([130, 255, 255])
        mask_blue = self.cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        # Find contours for both colors
        for mask, color in [(mask_red, 'red'), (mask_blue, 'blue')]:
            contours, _ = self.cv2.findContours(
                mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                x, y, w, h = self.cv2.boundingRect(contour)
                
                # Filter by size
                if 30 < w < 300 and 30 < h < 300:
                    aspect_ratio = w / h
                    
                    # Circular/square shapes
                    if 0.7 < aspect_ratio < 1.3:
                        stamps.append({
                            'bbox': [x, y, x+w, y+h],
                            'color': color,
                            'confidence': 'high',
                            'detection_method': 'visual'
                        })
        
        return stamps
    
    def _merge_detections(
        self, 
        text_stamps: List[Dict], 
        visual_stamps: List[Dict]
    ) -> List[Dict]:
        """Merge text and visual detections"""
        merged = []
        
        for vstamp in visual_stamps:
            # Find overlapping text stamp
            text_info = ''
            for tstamp in text_stamps:
                if self._bbox_iou(vstamp['bbox'], tstamp['bbox']) > 0.3:
                    text_info = tstamp['text']
                    break
            
            merged.append({
                'bbox': vstamp['bbox'],
                'color': vstamp.get('color', ''),
                'text': text_info,
                'confidence': 'very_high' if text_info else vstamp['confidence'],
                'detection_method': 'combined' if text_info else 'visual'
            })
        
        # Add non-overlapping text stamps
        for tstamp in text_stamps:
            overlaps = any(
                self._bbox_iou(tstamp['bbox'], v['bbox']) > 0.3 
                for v in visual_stamps
            )
            if not overlaps:
                merged.append({
                    'bbox': tstamp['bbox'],
                    'text': tstamp['text'],
                    'confidence': tstamp['confidence'],
                    'detection_method': 'text'
                })
        
        return merged
    
    def _bbox_iou(self, bbox1: List, bbox2: List) -> float:
        """Calculate IoU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dots_ocr_client.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Basic client
    print("Using basic client...")
    client = DotsOCRClient("http://localhost:5000")
    result = client.ocr_file(file_path, detect_stamps=True)
    
    print(f"\n✓ Text extracted: {len(result['text'])} characters")
    print(f"✓ Layout elements: {len(result['layout'])}")
    print(f"✓ Stamps detected: {len(result['stamps'])}")
    print(f"✓ Tables found: {len(result['tables'])}")
    print(f"✓ Formulas found: {len(result['formulas'])}")
    
    if result['stamps']:
        print("\nStamps:")
        for i, stamp in enumerate(result['stamps'], 1):
            print(f"  {i}. {stamp['bbox']} - {stamp.get('text', 'N/A')[:50]}")

