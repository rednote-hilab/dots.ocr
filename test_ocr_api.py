import requests
import json
import base64
import time

def test_ocr_api():
    """Test the OCR API server"""
    base_url = "http://localhost:5123"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health status: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return
    
    # Test with a sample image (you'll need to provide an actual image)
    print("\nTesting OCR endpoint...")
    
    # Example with image path (update this path to your actual image)
    sample_image_path = "demo/demo_image2.jpg"  # Update this path

    with open(sample_image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
        
    
    # Test non-streaming request with file path
    print("Testing non-streaming OCR with file path...")
    payload = {
        "image":  f'data:image/jpg;base64,{image_data}',
        "prompt_type": "prompt_layout_all_en",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_new_tokens": 12000,
        "stream": False
    }
    
    # start_time = time.time()
    # try:
    #     response = requests.post(f"{base_url}/ocr", json=payload)
    #     end_time = time.time()
        
    #     if response.status_code == 200:
    #         result = response.json()
    #         print(f"Non-streaming result (took {end_time - start_time:.2f}s):")
    #         print(f"Response: {result['response'][:200]}...")  # Show first 200 chars
    #     else:
    #         print(f"Error: {response.status_code} - {response.text}")
    # except Exception as e:
    #     print(f"Non-streaming test failed: {e}")
    
    # Test streaming request
    print("\nTesting streaming OCR...")
    payload["stream"] = True
    
    start_time = time.time()
    try:
        response = requests.post(f"{base_url}/ocr", json=payload, stream=True)
        
        if response.status_code == 200:
            print("Streaming results:")
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    chunk_count += 1
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'error' in data:
                            print(f"Error in stream: {data['error']}")
                            break
                        print(f"Chunk {chunk_count}: Done: {data['done']}, Response: {data['response']}")
                        
                        if data['done']:
                            end_time = time.time()
                            print(f"Final response: {data['response'][:100]}...")
                            print(f"Streaming completed (took {end_time - start_time:.2f}s)")
                            break
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {line}, error: {e}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Streaming test failed: {e}")

    # # Test with base64 encoded image if file exists
    # try:
    #     with open(sample_image_path, 'rb') as f:
    #         image_data = base64.b64encode(f.read()).decode('utf-8')
        
    #     print("\nTesting with base64 encoded image...")
    #     payload_b64 = {
    #         "image": f'data:image/jpg;base64,{image_data}',
    #         "prompt_type": "prompt_ocr",
    #         "stream": False
    #     }
        
    #     response = requests.post(f"{base_url}/ocr", json=payload_b64)
    #     if response.status_code == 200:
    #         result = response.json()
    #         print(f"Base64 test successful: {result['response'][:100]}...")
    #     else:
    #         print(f"Base64 test failed: {response.status_code} - {response.text}")
            
    # except FileNotFoundError:
    #     print(f"Sample image not found at {sample_image_path}, skipping base64 test")
    # except Exception as e:
    #     print(f"Base64 test failed: {e}")

if __name__ == "__main__":
    test_ocr_api()