#!/usr/bin/env python3
"""
Example usage of the training data creation script
"""

import os
from create_training_data import DotsOCRTrainingDataCreator

# Example 1: Process a directory of XML/JPEG pairs
def example_basic_usage():
    """Basic usage example"""
    creator = DotsOCRTrainingDataCreator()
    
    # Process directory and create training data with text
    input_directory = "/path/to/your/pagexml_jpeg_data"
    output_file = "dots_ocr_training_data.jsonl"
    
    creator.process_directory(
        input_dir=input_directory,
        output_file=output_file,
        include_text=True  # Include HTR text content
    )

# Example 2: Create both layout+text and layout-only datasets
def example_both_datasets():
    """Create both types of training data"""
    creator = DotsOCRTrainingDataCreator()
    
    input_directory = "/path/to/your/data"
    
    # Create layout + text recognition dataset
    creator.process_directory(
        input_dir=input_directory,
        output_file="training_layout_and_text.jsonl",
        include_text=True
    )
    
    # Create layout detection only dataset
    creator.process_directory(
        input_dir=input_directory,
        output_file="training_layout_only.jsonl",
        include_text=False
    )

# Example 3: Process a single file pair
def example_single_file():
    """Process a single XML/JPEG pair"""
    creator = DotsOCRTrainingDataCreator()
    
    sample = creator.create_training_sample(
        image_path="document_001.jpg",
        xml_path="document_001.xml",
        include_text=True
    )
    
    if sample:
        print("Training sample created successfully!")
        print(f"Number of messages: {len(sample['messages'])}")
        print(f"Assistant response preview: {sample['messages'][1]['content'][:100]}...")
    else:
        print("Failed to create training sample")

# Example 4: Validate your data format
def validate_pagexml_structure(xml_file: str):
    """Helper function to understand your PAGEXML structure"""
    import xml.etree.ElementTree as ET
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        print(f"Root tag: {root.tag}")
        print(f"Namespaces: {root.attrib}")
        
        # Find all unique element types
        element_types = set()
        for elem in root.iter():
            clean_tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            element_types.add(clean_tag)
        
        print(f"Found element types: {sorted(element_types)}")
        
        # Look for region types
        regions = []
        for elem in root.iter():
            if 'Region' in elem.tag:
                region_type = elem.get('type', 'Unknown')
                custom = elem.get('custom', '')
                regions.append((elem.tag, region_type, custom))
        
        print(f"Found {len(regions)} regions:")
        for tag, region_type, custom in regions[:5]:  # Show first 5
            print(f"  - {tag}: type='{region_type}', custom='{custom}'")
            
    except Exception as e:
        print(f"Error analyzing XML: {e}")

if __name__ == "__main__":
    print("Example usage of dots.ocr training data creator")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # example_basic_usage()
    # example_both_datasets()
    # example_single_file()
    
    # To understand your PAGEXML structure:
    # validate_pagexml_structure("path/to/your/sample.xml")