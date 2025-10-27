# DotsOCR API Server Interface Documentation

This document describes the request and response formats for the Flask-based DotsOCR API server.

**Due to the token segmentation of VLLM, the pixel dimensions of the image's length and width should preferably be multiples of 28! Otherwise, there may be slight misalignment in the recognition coordinates. When converting documents to images, it is recommended to use 200 dpi.**

## Server Information

- **Base URL**: `http://localhost:5000`
- **Protocol**: HTTP/1.1
- **Content Type**: application/json

## API Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check server status and model loading status

**Request Format**:
```http
GET /health HTTP/1.1
Host: localhost:5000
```

**Response Format**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Response Fields**:
- `status`: Server status, fixed as "healthy"
- `model_loaded`: Boolean value indicating whether the OCR model is loaded

---

### 2. OCR Processing

**Endpoint**: `POST /ocr`

**Description**: Process images and return OCR results

#### Request Format

**HTTP Headers**:
```http
POST /ocr HTTP/1.1
Host: localhost:5000
Content-Type: application/json
```

**Request Body**:
```json
{
  "image": "image data",
  "image_format": "path|url|base64",
  "prompt_type": "prompt_layout_all_en",
  "temperature": 0.1,
  "top_p": 1.0,
  "max_new_tokens": 12000,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | - | Image data, format depends on `image_format` |
| `image_format` | string | No | "path" | Image format: `path`(file path), `url`(web address), `base64`(Base64 encoding) |
| `prompt_type` | string | No | "prompt_layout_all_en" | Prompt type, determines OCR processing mode |
| `temperature` | float | No | 0.1 | Generation temperature parameter |
| `top_p` | float | No | 1.0 | Nucleus sampling parameter |
| `max_new_tokens` | integer | No | 12000 | Maximum number of new tokens to generate |
| `stream` | boolean | No | false | Whether to use streaming response |

#### Supported prompt_type Types

| Type | Description |
|------|-------------|
| `prompt_layout_all_en` | Parse all layout information including bounding boxes, categories, and text content, output in JSON format |
| `prompt_layout_only_en` | Detect layout only, output bounding boxes and categories without text content |
| `prompt_ocr` | Extract text content from images |
| `prompt_grounding_ocr` | Extract text content from specified bounding boxes |

#### Image Format Examples

1. **path format**: 
   ```json
   {
     "image": "/path/to/image.jpg",
     "image_format": "path"
   }
   ```

2. **url format**:
   ```json
   {
     "image": "https://example.com/image.jpg",
     "image_format": "url"
   }
   ```

3. **base64 format**:
   ```json
   {
     "image": "iVBORw0KGgoAAAANSUhEUgAAAB...",
     "image_format": "base64"
   }
   ```

---

## Response Format

### Non-streaming Response (stream=false)

**Success Response**:
```json
{
  "model": "dots-ocr",
  "response": "OCR processing result",
  "prompt_type": "prompt_layout_all_en"
}
```

**Response Fields**:
- `model`: Fixed as "dots-ocr"
- `response`: OCR processing result text
- `prompt_type`: Prompt type used

### Streaming Response (stream=true)

**Response Headers**:
```http
Content-Type: application/x-ndjson
```

**Response Format** (one JSON object per line):
```json
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:00.000000", "response": "partial result", "done": false}
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:01.000000", "response": "more results", "done": false}
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:02.000000", "response": "", "done": true}
```

**Streaming Response Fields**:
- `model`: Fixed as "dots-ocr"
- `created_at`: Response creation time (ISO format)
- `response`: Text fragment returned this time
- `done`: Boolean value, true indicates streaming response is complete

---

## Error Responses

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Request parameter error |
| 429 | Server busy |
| 500 | Internal server error |

### Error Response Format

```json
{
  "error": "error description"
}
```

### Common Error Examples

1. **Missing image data**:
   ```json
   {
     "error": "No image data provided"
   }
   ```

2. **Invalid prompt type**:
   ```json
   {
     "error": "Invalid prompt_type. Must be one of: ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_ocr', 'prompt_grounding_ocr']"
   }
   ```

3. **Server busy**:
   ```json
   {
     "error": "Server is busy processing another request"
   }
   ```

4. **Model not loaded**:
   ```json
   {
     "error": "Model not loaded"
   }
   ```

---

## Request Examples

### cURL Examples

#### 1. Health Check
```bash
curl -X GET http://localhost:5000/health
```

#### 2. OCR with File Path
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "prompt_type": "prompt_layout_all_en"
  }'
```

#### 3. OCR with Base64 Image
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAB...",
    "image_format": "base64",
    "prompt_type": "prompt_ocr"
  }'
```

#### 4. Streaming Response
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "stream": true
  }'
```

### Python Example

```python
import requests
import json

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# OCR processing
ocr_data = {
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "prompt_type": "prompt_layout_all_en",
    "temperature": 0.1,
    "max_new_tokens": 12000,
    "stream": False
}

response = requests.post('http://localhost:5000/ocr', json=ocr_data)
print(response.json())
```

---

## Important Notes

1. **Concurrency Limitation**: The server uses a processing lock and can only handle one OCR request at a time
2. **Temporary Files**: For base64 and url format images, the server creates temporary files and automatically cleans them up after processing
3. **GPU Requirements**: Model execution requires CUDA support
4. **Memory Usage**: Processing large images may require substantial memory
5. **Timeout Handling**: Long processing times may cause timeouts; streaming response is recommended for large images

---

## Layout Category Description

When using `prompt_layout_all_en` or `prompt_layout_only_en`, the possible layout categories include:

- `Caption`: Image caption
- `Footnote`: Footnote
- `Formula`: Formula
- `List-item`: List item
- `Page-footer`: Page footer
- `Page-header`: Page header
- `Picture`: Picture
- `Section-header`: Section header
- `Table`: Table
- `Text`: Body text
- `Title`: Title

Output format varies by category:
- `Picture`: Does not contain text content
- `Formula`: LaTeX format
- `Table`: HTML format
- `Others`: Markdown format