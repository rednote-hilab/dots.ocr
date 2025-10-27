#!/usr/bin/env python3
"""
Test script for the training data creation functionality
"""

import os
import json
import tempfile
from pathlib import Path
from PIL import Image
from create_training_data import DotsOCRTrainingDataCreator, PageXMLParser

def create_sample_pagexml(output_path: str):
    """Create a sample PAGEXML file for testing"""
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page imageFilename="test.jpg" imageWidth="1000" imageHeight="1200">
        <TextRegion id="r1" type="paragraph">
            <Coords points="100,100 400,100 400,200 100,200"/>
            <TextEquiv>
                <Unicode>This is a sample text line in the document.</Unicode>
            </TextEquiv>
        </TextRegion>
        <TextRegion id="r2" type="heading" custom="title">
            <Coords points="100,50 400,50 400,90 100,90"/>
            <TextEquiv>
                <Unicode>Sample Document Title</Unicode>
            </TextEquiv>
        </TextRegion>
        <TableRegion id="r3">
            <Coords points="100,250 500,250 500,400 100,400"/>
            <TextLine id="l1">
                <Coords points="100,250 500,250 500,280 100,280"/>
                <TextEquiv>
                    <Unicode>Name	Age	City</Unicode>
                </TextEquiv>
            </TextLine>
            <TextLine id="l2">
                <Coords points="100,280 500,280 500,310 100,310"/>
                <TextEquiv>
                    <Unicode>John	25	New York</Unicode>
                </TextEquiv>
            </TextLine>
        </TableRegion>
        <ImageRegion id="r4">
            <Coords points="600,100 800,100 800,300 600,300"/>
        </ImageRegion>
    </Page>
</PcGts>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)

def create_sample_image(output_path: str, width: int = 1000, height: int = 1200):
    """Create a sample image for testing"""
    # Create a simple test image
    img = Image.new('RGB', (width, height), color='white')
    img.save(output_path)

def test_pagexml_parser():
    """Test the PAGEXML parser"""
    print("Testing PAGEXML parser...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        xml_path = os.path.join(temp_dir, "test.xml")
        create_sample_pagexml(xml_path)
        
        parser = PageXMLParser()
        elements = parser.parse_pagexml(xml_path)
        
        print(f"Parsed {len(elements)} elements:")
        for i, elem in enumerate(elements):
            print(f"  {i+1}. Category: {elem['category']}, BBox: {elem['bbox']}")
            if 'text' in elem:
                print(f"      Text: {elem['text'][:50]}...")
        
        assert len(elements) > 0, "Should parse at least one element"
        assert any(elem['category'] == 'Title' for elem in elements), "Should find title element"
        assert any(elem['category'] == 'Text' for elem in elements), "Should find text element"
        assert any(elem['category'] == 'Table' for elem in elements), "Should find table element"
        assert any(elem['category'] == 'Picture' for elem in elements), "Should find picture element"
        
        print("✅ PAGEXML parser test passed!")

def test_training_data_creator():
    """Test the training data creator"""
    print("\nTesting training data creator...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        xml_path = os.path.join(temp_dir, "test.xml")
        img_path = os.path.join(temp_dir, "test.jpg")
        
        create_sample_pagexml(xml_path)
        create_sample_image(img_path)
        
        creator = DotsOCRTrainingDataCreator()
        
        # Test single sample creation
        sample = creator.create_training_sample(img_path, xml_path, include_text=True)
        
        assert sample is not None, "Should create training sample"
        assert 'messages' in sample, "Should have messages key"
        assert len(sample['messages']) == 2, "Should have user and assistant messages"
        
        user_msg = sample['messages'][0]
        assistant_msg = sample['messages'][1]
        
        assert user_msg['role'] == 'user', "First message should be user"
        assert assistant_msg['role'] == 'assistant', "Second message should be assistant"
        assert len(user_msg['content']) == 2, "User content should have image and text"
        
        # Check that assistant content is valid JSON
        try:
            assistant_data = json.loads(assistant_msg['content'])
            assert isinstance(assistant_data, list), "Assistant content should be JSON list"
            assert len(assistant_data) > 0, "Should have layout elements"
            
            # Check first element structure
            first_elem = assistant_data[0]
            assert 'bbox' in first_elem, "Element should have bbox"
            assert 'category' in first_elem, "Element should have category"
            assert len(first_elem['bbox']) == 4, "BBox should have 4 coordinates"
            
        except json.JSONDecodeError:
            assert False, "Assistant content should be valid JSON"
        
        print("✅ Training data creator test passed!")

def test_directory_processing():
    """Test processing a directory of files"""
    print("\nTesting directory processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test file pairs
        for i in range(3):
            xml_path = os.path.join(temp_dir, f"doc_{i:03d}.xml")
            img_path = os.path.join(temp_dir, f"doc_{i:03d}.jpg")
            
            create_sample_pagexml(xml_path)
            create_sample_image(img_path)
        
        # Process directory
        output_file = os.path.join(temp_dir, "training_data.jsonl")
        creator = DotsOCRTrainingDataCreator()
        creator.process_directory(temp_dir, output_file, include_text=True)
        
        # Check output file
        assert os.path.exists(output_file), "Should create output file"
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 3, f"Should have 3 training samples, got {len(lines)}"
        
        # Check each line is valid JSON
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                assert 'messages' in data, f"Line {i} should have messages"
            except json.JSONDecodeError:
                assert False, f"Line {i} should be valid JSON"
        
        print("✅ Directory processing test passed!")

def test_layout_only_mode():
    """Test layout-only mode (no text content)"""
    print("\nTesting layout-only mode...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        xml_path = os.path.join(temp_dir, "test.xml")
        img_path = os.path.join(temp_dir, "test.jpg")
        
        create_sample_pagexml(xml_path)
        create_sample_image(img_path)
        
        creator = DotsOCRTrainingDataCreator()
        sample = creator.create_training_sample(img_path, xml_path, include_text=False)
        
        assert sample is not None, "Should create training sample"
        
        # Check that text fields are removed
        assistant_data = json.loads(sample['messages'][1]['content'])
        for elem in assistant_data:
            if elem['category'] != 'Picture':
                assert 'text' not in elem, f"Element {elem} should not have text field in layout-only mode"
        
        print("✅ Layout-only mode test passed!")

def main():
    """Run all tests"""
    print("Running training script tests...")
    print("=" * 50)
    
    try:
        test_pagexml_parser()
        test_training_data_creator()
        test_directory_processing()
        test_layout_only_mode()
        
        print("\n🎉 All tests passed! The training script is working correctly.")
        print("\nYou can now use it with your PAGEXML + JPEG data:")
        print("python create_training_data.py --input_dir /path/to/data --output_file training.jsonl")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()