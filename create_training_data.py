#!/usr/bin/env python3
"""
Training data preparation script for dots.ocr
Converts PAGEXML + JPEG pairs to dots.ocr training format

Usage:
    python create_training_data.py --input_dir /path/to/data --output_file training_data.jsonl
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import re


class PageXMLParser:
    """Parser for PAGEXML files to extract layout and text information"""
    
    def __init__(self):
        # Mapping from PAGEXML region types to dots.ocr categories
        self.category_mapping = {
            'TextRegion': 'Text',
            'TitleRegion': 'Title',
            'TableRegion': 'Table',
            'ImageRegion': 'Picture',
            'MathsRegion': 'Formula',
            'GraphicRegion': 'Picture',
            'LineDrawingRegion': 'Picture',
            'ChartRegion': 'Picture',
            'SeparatorRegion': 'Text',
            'NoiseRegion': 'Text',
            'UnknownRegion': 'Text',
            'HeaderRegion': 'Page-header',
            'FooterRegion': 'Page-footer',
            'CaptionRegion': 'Caption',
            'FootnoteRegion': 'Footnote',
            'ListRegion': 'List-item',
            # Add more mappings based on your PAGEXML schema
        }
        
        self.namespaces = {
            'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
            'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
        }
    
    def parse_pagexml(self, xml_file: str) -> List[Dict]:
        """
        Parse PAGEXML file and extract layout elements with text
        
        Args:
            xml_file: Path to PAGEXML file
            
        Returns:
            List of layout elements with bbox, category, and text
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Try to detect namespace automatically if not in predefined list
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0][1:]
                self.namespaces['page'] = namespace
                self.namespaces['pc'] = namespace
            
            elements = []
            
            # Find all regions (TextRegion, ImageRegion, etc.)
            for region in root.findall('.//page:TextRegion', self.namespaces):
                elements.extend(self._parse_text_region(region))
            
            for region in root.findall('.//page:ImageRegion', self.namespaces):
                elements.extend(self._parse_image_region(region))
                
            for region in root.findall('.//page:TableRegion', self.namespaces):
                elements.extend(self._parse_table_region(region))
                
            # Try alternative namespace if no elements found
            if not elements:
                elements = self._parse_with_fallback_namespace(root)
            
            # Sort by reading order (top-to-bottom, left-to-right)
            elements = self._sort_by_reading_order(elements)
            
            return elements
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return []
    
    def _parse_text_region(self, region) -> List[Dict]:
        """Parse a TextRegion element"""
        elements = []
        
        # Get region coordinates
        coords = region.find('page:Coords', self.namespaces)
        if coords is None:
            return elements
            
        bbox = self._parse_coordinates(coords)
        if not bbox:
            return elements
        
        # Determine category based on region type or custom attributes
        region_type = region.get('type', 'TextRegion')
        custom_type = region.get('custom', '')
        
        # Parse custom attributes for more specific categorization
        category = self._determine_category(region_type, custom_type)
        
        # Extract text content
        text_content = self._extract_text_content(region)
        
        if text_content.strip():  # Only add if there's actual text
            elements.append({
                'bbox': bbox,
                'category': category,
                'text': text_content
            })
        
        return elements
    
    def _parse_image_region(self, region) -> List[Dict]:
        """Parse an ImageRegion element"""
        coords = region.find('page:Coords', self.namespaces)
        if coords is None:
            return []
            
        bbox = self._parse_coordinates(coords)
        if not bbox:
            return []
        
        # Pictures don't have text content in dots.ocr format
        return [{
            'bbox': bbox,
            'category': 'Picture'
        }]
    
    def _parse_table_region(self, region) -> List[Dict]:
        """Parse a TableRegion element"""
        coords = region.find('page:Coords', self.namespaces)
        if coords is None:
            return []
            
        bbox = self._parse_coordinates(coords)
        if not bbox:
            return []
        
        # Extract table text and format as HTML
        text_content = self._extract_table_as_html(region)
        
        return [{
            'bbox': bbox,
            'category': 'Table',
            'text': text_content
        }]
    
    def _parse_coordinates(self, coords_element) -> Optional[List[int]]:
        """Parse coordinates from Coords element"""
        points_attr = coords_element.get('points', '')
        if not points_attr:
            return None
        
        try:
            # Parse points "x1,y1 x2,y2 x3,y3 x4,y4"
            points = []
            for point_str in points_attr.split():
                if ',' in point_str:
                    x, y = map(int, point_str.split(','))
                    points.append((x, y))
            
            if len(points) < 2:
                return None
            
            # Calculate bounding box [x1, y1, x2, y2]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing coordinates '{points_attr}': {e}")
            return None
    
    def _extract_text_content(self, region) -> str:
        """Extract text content from a region (combining all TextLine elements)"""
        text_parts = []
        
        # First try to get TextEquiv at region level
        text_equiv = region.find('page:TextEquiv/page:Unicode', self.namespaces)
        if text_equiv is not None and text_equiv.text:
            return text_equiv.text.strip()
        
        # Otherwise, combine text from all TextLine elements
        for text_line in region.findall('.//page:TextLine', self.namespaces):
            line_text = self._extract_line_text(text_line)
            if line_text:
                text_parts.append(line_text)
        
        return '\n'.join(text_parts)
    
    def _extract_line_text(self, text_line) -> str:
        """Extract text from a TextLine element"""
        # Try TextEquiv first
        text_equiv = text_line.find('page:TextEquiv/page:Unicode', self.namespaces)
        if text_equiv is not None and text_equiv.text:
            return text_equiv.text.strip()
        
        # Fall back to combining Word elements
        word_texts = []
        for word in text_line.findall('.//page:Word', self.namespaces):
            word_equiv = word.find('page:TextEquiv/page:Unicode', self.namespaces)
            if word_equiv is not None and word_equiv.text:
                word_texts.append(word_equiv.text.strip())
        
        return ' '.join(word_texts)
    
    def _extract_table_as_html(self, table_region) -> str:
        """Extract table content and format as HTML"""
        # This is a simplified version - you might need to enhance based on your PAGEXML structure
        text_content = self._extract_text_content(table_region)
        
        if not text_content:
            return "<table></table>"
        
        # Basic conversion - split by lines and create simple table
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        if not lines:
            return "<table></table>"
        
        html_parts = ["<table>"]
        for line in lines:
            # Split by tabs or multiple spaces to detect columns
            cells = re.split(r'\t+|\s{2,}', line)
            if len(cells) > 1:
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell.strip()}</td>")
                html_parts.append("</tr>")
            else:
                # Single column
                html_parts.append(f"<tr><td>{line}</td></tr>")
        
        html_parts.append("</table>")
        return "".join(html_parts)
    
    def _determine_category(self, region_type: str, custom_attr: str) -> str:
        """Determine dots.ocr category from PAGEXML region type and custom attributes"""
        
        # Parse custom attributes for specific types
        if custom_attr:
            if 'header' in custom_attr.lower():
                return 'Page-header'
            elif 'footer' in custom_attr.lower():
                return 'Page-footer'
            elif 'title' in custom_attr.lower() or 'heading' in custom_attr.lower():
                return 'Title'
            elif 'caption' in custom_attr.lower():
                return 'Caption'
            elif 'footnote' in custom_attr.lower():
                return 'Footnote'
            elif 'list' in custom_attr.lower():
                return 'List-item'
            elif 'formula' in custom_attr.lower() or 'math' in custom_attr.lower():
                return 'Formula'
        
        # Fall back to region type mapping
        return self.category_mapping.get(region_type, 'Text')
    
    def _sort_by_reading_order(self, elements: List[Dict]) -> List[Dict]:
        """Sort elements by reading order (top-to-bottom, left-to-right)"""
        def reading_order_key(element):
            bbox = element['bbox']
            # Primary sort by top coordinate, secondary by left coordinate
            return (bbox[1], bbox[0])
        
        return sorted(elements, key=reading_order_key)
    
    def _parse_with_fallback_namespace(self, root) -> List[Dict]:
        """Try parsing without namespace if the standard approach fails"""
        elements = []
        
        # Find all elements that might be regions
        for elem in root.iter():
            if 'Region' in elem.tag:
                coords = elem.find('.//Coords') or elem.find('Coords')
                if coords is not None:
                    bbox = self._parse_coordinates(coords)
                    if bbox:
                        # Determine category from tag name
                        tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        category = self.category_mapping.get(tag_name, 'Text')
                        
                        if category != 'Picture':
                            text_content = self._extract_text_fallback(elem)
                            if text_content:
                                elements.append({
                                    'bbox': bbox,
                                    'category': category,
                                    'text': text_content
                                })
                        else:
                            elements.append({
                                'bbox': bbox,
                                'category': category
                            })
        
        return self._sort_by_reading_order(elements)
    
    def _extract_text_fallback(self, element) -> str:
        """Fallback text extraction without namespace"""
        text_parts = []
        
        # Look for Unicode text elements
        for unicode_elem in element.iter():
            if 'Unicode' in unicode_elem.tag and unicode_elem.text:
                text_parts.append(unicode_elem.text.strip())
        
        return '\n'.join(text_parts)


class DotsOCRTrainingDataCreator:
    """Creates training data in dots.ocr format"""
    
    def __init__(self):
        self.parser = PageXMLParser()
        
        # Training prompts
        self.layout_all_prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""

        self.layout_only_prompt = """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."""
    
    def create_training_sample(self, image_path: str, xml_path: str, include_text: bool = True) -> Optional[Dict]:
        """
        Create a single training sample from image and XML pair
        
        Args:
            image_path: Path to JPEG image
            xml_path: Path to PAGEXML file
            include_text: Whether to include text content (vs layout only)
            
        Returns:
            Training sample in chat format or None if failed
        """
        try:
            # Verify files exist
            if not os.path.exists(image_path) or not os.path.exists(xml_path):
                print(f"Missing files: {image_path} or {xml_path}")
                return None
            
            # Parse PAGEXML
            elements = self.parser.parse_pagexml(xml_path)
            if not elements:
                print(f"No elements found in {xml_path}")
                return None
            
            # Choose prompt based on whether we want text
            prompt = self.layout_all_prompt if include_text else self.layout_only_prompt
            
            # For layout-only mode, remove text fields
            if not include_text:
                elements = [{k: v for k, v in elem.items() if k != 'text'} for elem in elements]
            
            # Create training sample
            training_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": os.path.abspath(image_path)},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(elements, ensure_ascii=False)
                    }
                ]
            }
            
            return training_sample
            
        except Exception as e:
            print(f"Error creating training sample for {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir: str, output_file: str, include_text: bool = True):
        """
        Process all XML/JPEG pairs in a directory
        
        Args:
            input_dir: Directory containing XML and JPEG files
            output_file: Output JSONL file path
            include_text: Whether to include text content
        """
        input_path = Path(input_dir)
        
        # Find all XML files
        xml_files = list(input_path.glob("*.xml"))
        
        if not xml_files:
            print(f"No XML files found in {input_dir}")
            return
        
        print(f"Found {len(xml_files)} XML files")
        
        training_samples = []
        processed = 0
        skipped = 0
        
        for xml_file in xml_files:
            # Find corresponding JPEG
            base_name = xml_file.stem
            jpeg_file = None
            
            # Try different image extensions
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
                candidate = xml_file.parent / (base_name + ext)
                if candidate.exists():
                    jpeg_file = candidate
                    break
            
            if jpeg_file is None:
                print(f"No corresponding JPEG found for {xml_file}")
                skipped += 1
                continue
            
            # Create training sample
            sample = self.create_training_sample(str(jpeg_file), str(xml_file), include_text)
            if sample:
                training_samples.append(sample)
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} samples...")
            else:
                skipped += 1
        
        # Save to JSONL
        if training_samples:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in training_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"\n✅ Successfully created training data:")
            print(f"   📁 Input directory: {input_dir}")
            print(f"   📄 Output file: {output_file}")
            print(f"   ✅ Processed: {processed} samples")
            print(f"   ⏭️  Skipped: {skipped} samples")
            print(f"   📊 Total training samples: {len(training_samples)}")
        else:
            print("❌ No training samples were created")


def main():
    parser = argparse.ArgumentParser(description='Create dots.ocr training data from PAGEXML + JPEG pairs')
    parser.add_argument('--input_dir', required=True, help='Directory containing XML and JPEG files')
    parser.add_argument('--output_file', required=True, help='Output JSONL file path')
    parser.add_argument('--layout_only', action='store_true', 
                       help='Create layout detection only data (no text content)')
    parser.add_argument('--both', action='store_true',
                       help='Create both layout+text and layout-only datasets')
    
    args = parser.parse_args()
    
    creator = DotsOCRTrainingDataCreator()
    
    if args.both:
        # Create both versions
        base_name = os.path.splitext(args.output_file)[0]
        
        print("Creating layout + text dataset...")
        creator.process_directory(args.input_dir, f"{base_name}_with_text.jsonl", include_text=True)
        
        print("\nCreating layout-only dataset...")
        creator.process_directory(args.input_dir, f"{base_name}_layout_only.jsonl", include_text=False)
    else:
        include_text = not args.layout_only
        creator.process_directory(args.input_dir, args.output_file, include_text=include_text)


if __name__ == "__main__":
    main()