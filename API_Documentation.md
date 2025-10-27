# DotsOCR API 服务器接口文档

本文档描述了基于 Flask 的 DotsOCR API 服务器的请求和响应格式。

**由于VLLM的token分割，图片的长和宽的像素数量最好要是28的倍数！！否则可能出现识别坐标略微错位的情况。文档转图片时，推荐使用200dpi。**

**Due to the token segmentation of VLLM, the pixel dimensions of the image's length and width should preferably be multiples of 28! Otherwise, there may be slight misalignment in the recognition coordinates. When converting documents to images, it is recommended to use 200 dpi.**

## 服务器信息

- **基础URL**: `http://localhost:5000`
- **协议**: HTTP/1.1
- **内容类型**: application/json

## 接口端点

### 1. 健康检查

**端点**: `GET /health`

**描述**: 检查服务器状态和模型加载情况

**请求格式**:
```http
GET /health HTTP/1.1
Host: localhost:5000
```

**响应格式**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**响应字段说明**:
- `status`: 服务器状态，固定为 "healthy"
- `model_loaded`: 布尔值，表示OCR模型是否已加载

---

### 2. OCR处理

**端点**: `POST /ocr`

**描述**: 处理图像并返回OCR结果

#### 请求格式

**HTTP头**:
```http
POST /ocr HTTP/1.1
Host: localhost:5000
Content-Type: application/json
```

**请求体**:
```json
{
  "image": "图像数据",
  "image_format": "path|url|base64",
  "prompt_type": "prompt_layout_all_en",
  "temperature": 0.1,
  "top_p": 1.0,
  "max_new_tokens": 12000,
  "stream": false
}
```

#### 请求参数说明

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `image` | string | 是 | - | 图像数据，格式取决于 `image_format` |
| `image_format` | string | 否 | "path" | 图像格式：`path`(文件路径)、`url`(网络地址)、`base64`(Base64编码) |
| `prompt_type` | string | 否 | "prompt_layout_all_en" | 提示类型，决定OCR处理模式 |
| `temperature` | float | 否 | 0.1 | 生成温度参数 |
| `top_p` | float | 否 | 1.0 | nucleus采样参数 |
| `max_new_tokens` | integer | 否 | 12000 | 最大新生成token数 |
| `stream` | boolean | 否 | false | 是否使用流式响应 |

#### 支持的 prompt_type 类型

| 类型 | 描述 |
|------|------|
| `prompt_layout_all_en` | 解析所有布局信息，包括边界框、类别和文本内容，输出JSON格式 |
| `prompt_layout_only_en` | 仅检测布局，输出边界框和类别，不包含文本内容 |
| `prompt_ocr` | 提取图像中的文本内容 |
| `prompt_grounding_ocr` | 从指定边界框中提取文本内容 |

#### 图像格式说明

1. **path格式**: 
   ```json
   {
     "image": "/path/to/image.jpg",
     "image_format": "path"
   }
   ```

2. **url格式**:
   ```json
   {
     "image": "https://example.com/image.jpg",
     "image_format": "url"
   }
   ```

3. **base64格式**:
   ```json
   {
     "image": "iVBORw0KGgoAAAANSUhEUgAAAB...",
     "image_format": "base64"
   }
   ```

---

## 响应格式

### 非流式响应 (stream=false)

**成功响应**:
```json
{
  "model": "dots-ocr",
  "response": "OCR处理结果",
  "prompt_type": "prompt_layout_all_en"
}
```

**响应字段说明**:
- `model`: 固定为 "dots-ocr"
- `response`: OCR处理的结果文本
- `prompt_type`: 使用的提示类型

### 流式响应 (stream=true)

**响应头**:
```http
Content-Type: application/x-ndjson
```

**响应格式** (每行一个JSON对象):
```json
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:00.000000", "response": "部分结果", "done": false}
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:01.000000", "response": "更多结果", "done": false}
{"model": "dots-ocr", "created_at": "2024-01-01T10:00:02.000000", "response": "", "done": true}
```

**流式响应字段说明**:
- `model`: 固定为 "dots-ocr"
- `created_at`: 响应创建时间 (ISO格式)
- `response`: 本次返回的文本片段
- `done`: 布尔值，true表示流式响应结束

---

## 错误响应

### HTTP状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 429 | 服务器繁忙 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
  "error": "错误描述信息"
}
```

### 常见错误示例

1. **缺少图像数据**:
   ```json
   {
     "error": "No image data provided"
   }
   ```

2. **无效的提示类型**:
   ```json
   {
     "error": "Invalid prompt_type. Must be one of: ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_ocr', 'prompt_grounding_ocr']"
   }
   ```

3. **服务器繁忙**:
   ```json
   {
     "error": "Server is busy processing another request"
   }
   ```

4. **模型未加载**:
   ```json
   {
     "error": "Model not loaded"
   }
   ```

---

## 请求示例

### cURL 示例

#### 1. 健康检查
```bash
curl -X GET http://localhost:5000/health
```

#### 2. 使用文件路径进行OCR
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "prompt_type": "prompt_layout_all_en"
  }'
```

#### 3. 使用Base64图像进行OCR
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAB...",
    "image_format": "base64",
    "prompt_type": "prompt_ocr"
  }'
```

#### 4. 流式响应
```bash
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "stream": true
  }'
```

### Python 示例

```python
import requests
import json

# 健康检查
response = requests.get('http://localhost:5000/health')
print(response.json())

# OCR处理
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

## 注意事项

1. **并发限制**: 服务器使用处理锁，同时只能处理一个OCR请求
2. **临时文件**: 对于base64和url格式的图像，服务器会创建临时文件并在处理完成后自动清理
3. **GPU要求**: 模型运行需要CUDA支持
4. **内存使用**: 处理大图像时可能需要大量内存
5. **超时处理**: 长时间处理可能导致超时，建议使用流式响应处理大图像

---

## 布局类别说明

当使用 `prompt_layout_all_en` 或 `prompt_layout_only_en` 时，可能识别的布局类别包括：

- `Caption`: 图像标题
- `Footnote`: 脚注
- `Formula`: 公式
- `List-item`: 列表项
- `Page-footer`: 页脚
- `Page-header`: 页眉
- `Picture`: 图片
- `Section-header`: 章节标题
- `Table`: 表格
- `Text`: 正文
- `Title`: 标题

输出格式根据类别不同：
- `Picture`: 不包含文本内容
- `Formula`: LaTeX格式
- `Table`: HTML格式
- 其他: Markdown格式