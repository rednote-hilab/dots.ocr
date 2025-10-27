# Training Data Creation for dots.ocr

This directory contains scripts to convert PAGEXML + JPEG pairs into training data for the dots.ocr model.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r training_requirements.txt
```

### 2. Prepare Your Data
Organize your data so that XML and JPEG files have the same name:
```
data/
├── document_001.xml
├── document_001.jpg
├── document_002.xml
├── document_002.jpg
└── ...
```

### 3. Create Training Data
```bash
# Create full layout + text training data
python create_training_data.py --input_dir /path/to/data --output_file training_data.jsonl

# Create layout detection only data
python create_training_data.py --input_dir /path/to/data --output_file layout_only.jsonl --layout_only

# Create both types
python create_training_data.py --input_dir /path/to/data --output_file training --both
```

## Script Features

### Supported PAGEXML Elements
- **TextRegion** → Text
- **TitleRegion** → Title  
- **TableRegion** → Table
- **ImageRegion** → Picture
- **MathsRegion** → Formula
- **HeaderRegion** → Page-header
- **FooterRegion** → Page-footer
- **CaptionRegion** → Caption
- **FootnoteRegion** → Footnote
- **ListRegion** → List-item

### Automatic Text Extraction
The script extracts text from:
1. Region-level `TextEquiv/Unicode` elements
2. Line-level `TextLine/TextEquiv/Unicode` elements  
3. Word-level `Word/TextEquiv/Unicode` elements

### Output Formats
- **Layout + Text**: Full layout detection and text recognition
- **Layout Only**: Bounding boxes and categories only
- **Reading Order**: Automatically sorted top-to-bottom, left-to-right

## Output Format

Each line in the JSONL file contains a training sample:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "PROMPT_TEXT"}
      ]
    },
    {
      "role": "assistant", 
      "content": "[{\"bbox\": [x1, y1, x2, y2], \"category\": \"Text\", \"text\": \"HTR content\"}]"
    }
  ]
}
```

## Customization

### Adding New Region Types
Edit the `category_mapping` dictionary in `PageXMLParser.__init__()`:

```python
self.category_mapping = {
    'YourCustomRegion': 'Text',  # or appropriate category
    # ... existing mappings
}
```

### Custom Text Processing
Override `_extract_text_content()` method for custom text extraction logic.

### Custom Reading Order
Override `_sort_by_reading_order()` method for custom sorting logic.

## Troubleshooting

### No elements found
- Check your PAGEXML namespace
- Use `validate_pagexml_structure()` from `example_usage.py`
- Verify coordinate format

### Missing text content
- Ensure `TextEquiv/Unicode` elements exist
- Check text extraction path in your PAGEXML
- Verify HTR ground truth is embedded in XML

### Wrong categories
- Review and update `category_mapping`
- Check custom attributes in your PAGEXML
- Add custom categorization logic

## Example Usage

See `example_usage.py` for detailed usage examples and helper functions.

## Integration with dots.ocr Training

The generated JSONL files can be used directly with vision-language model training frameworks like:
- Transformers fine-tuning
- LLaMA-Factory
- Custom training loops

Make sure to:
1. Set appropriate learning rates for vision encoder vs. language model
2. Use proper image preprocessing (resize to <11M pixels)
3. Handle long sequences (JSON outputs can be lengthy)
4. Consider multi-task training with different prompts