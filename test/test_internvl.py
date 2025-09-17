import requests
import base64
import json
import time

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_vllm_image_description():
    """Test function to send image description request to VLLM server"""
    
    # Server URL
    url = "http://localhost:8010/v1/chat/completions"
    
    # Encode the image
    try:
        base64_image = encode_image_to_base64("charts/3.png")
    except FileNotFoundError:
        print("Error: picture not found in current directory")
        return
    
    # Prepare the request payload in OpenAI format
    payload = {
        "model": "/app/models/InternVL3_5-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "extract the information from this image objectively"
                    },
                    {
                        "type": "image_url",
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sagi123"
    }
    
    try:
        # Send the request
        print("Sending request to VLLM server...")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=1000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"Request completed in {elapsed_time:.2f} seconds")
            print(result)
        else:
            print(f"Error: HTTP {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to VLLM server at localhost:8000")
        print("Make sure your VLLM server is running")
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_vllm_image_description()