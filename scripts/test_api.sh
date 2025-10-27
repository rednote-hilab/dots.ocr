#!/bin/bash
# Test API endpoints

set -e

API_URL=${API_URL:-"http://localhost:5000"}

echo "═══════════════════════════════════════════════════════"
echo "🧪 Testing dots.ocr API at $API_URL"
echo "═══════════════════════════════════════════════════════"
echo ""

# Test 1: Health check
echo "Test 1: Health check..."
response=$(curl -s "$API_URL/health")
echo "Response: $response"

if echo "$response" | grep -q "healthy"; then
    echo "✓ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

echo ""

# Test 2: OCR with demo image
echo "Test 2: OCR with demo image..."
if [ -f "demo/demo_image1.jpg" ]; then
    response=$(curl -s -X POST "$API_URL/ocr" \
      -H "Content-Type: application/json" \
      -d '{
        "image": "demo/demo_image1.jpg",
        "image_format": "path",
        "prompt_type": "prompt_layout_all_en"
      }')
    
    if echo "$response" | grep -q "response"; then
        echo "✓ OCR test passed"
        echo "Response length: $(echo "$response" | wc -c) bytes"
    else
        echo "❌ OCR test failed"
        echo "Response: $response"
        exit 1
    fi
else
    echo "⚠️  Demo image not found, skipping"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ All tests passed!"
echo "═══════════════════════════════════════════════════════"
