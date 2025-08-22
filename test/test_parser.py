import pytest
import requests
import time
import logging
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestParserAPI:
    """Test class for DotsOCR Parser API"""
    
    def __init__(self, base_url: str = "http://localhost:6009"):
        self.base_url = base_url
        self.session = requests.Session()
    
    async def test_parse_file_with_status_monitoring(self):
        """Test the /parse/file endpoint and monitor status"""
        
        # Test data
        form_data = {
            "input_s3_path": "s3://monkeyocr/test/input/test_pdf/small.pdf",
            "output_s3_path": "s3://monkeyocr/test/output/test_pdf/small",
            "knowledgebaseId": "1222",
            "workspaceId": "3444"
        }
        
        logger.info(f"Starting parse request with data: {form_data}")
        
        # Step 1: Submit parsing job
        response = self._submit_parse_job(form_data)
        assert response is not None, "Failed to submit parse job"
        
        OCRJobId = response.get("OCRJobId")
        assert OCRJobId is not None, "No OCRJobId returned from parse request"
        logger.info(f"Received OCRJobId: {OCRJobId}")
        
        # Step 2: Monitor job status
        final_status = await self._monitor_job_status(OCRJobId)
        
        # Step 3: Verify final result
        assert final_status["status"] == "completed", f"Job failed with status: {final_status}"
        logging.info("Test completed successfully!")
        
        return final_status
    
    def _submit_parse_job(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Submit a parsing job to the API"""
        try:
            url = f"{self.base_url}/parse/file"
            response = self.session.post(url, data=form_data)
            
            logger.info(f"Parse request status code: {response.status_code}")
            logger.info(f"Parse request response: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Parse request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception during parse request: {e}")
            return None
    
    async def _monitor_job_status(self, OCRJobId: str, max_wait_time: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """Monitor job status until completion or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                status_response = self._check_job_status(OCRJobId)  # 保持原同步调用
                logger.debug(f"Status response: {status_response}")
                
                if status_response is None:
                    logger.warning("Failed to get status response, continuing...")
                    await asyncio.sleep(poll_interval)
                    continue
                
                status = status_response.get("status")
                message = status_response.get("message", "")
                
                # Check if job is completed (success or failure)
                if status in ["completed", "failed", "canceled"]:
                    logger.info(f"Job {OCRJobId} completed with status: {status}")
                    return status_response
                
                # Continue polling for pending/processing/retrying states
                if status in ["pending", "processing", "retrying"]:
                    logger.info(f"Job {OCRJobId} is still {status}, waiting for next check...")
                    await asyncio.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"Unexpected status: {status}")
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Exception during status check: {e}")
                await asyncio.sleep(poll_interval)
        
        # Timeout reached
        logger.error(f"Job {OCRJobId} monitoring timed out after {max_wait_time} seconds")
        return {"status": "timeout", "message": "Status monitoring timed out"}
    
    def _check_job_status(self, OCRJobId: str) -> Dict[str, Any]:
        """Check the status of a specific job"""
        try:
            url = f"{self.base_url}/status"
            data = {"OCRJobId": OCRJobId}
            
            response = self.session.post(url, data=data)
            
            logger.info(f"Status check response code: {response.status_code}")
            logger.info(f"Status check response: {response.text}")
            
            if response.status_code in [200, 202]:
                return response.json()
            elif response.status_code == 500:
                # Job failed
                return response.json()
            else:
                logger.error(f"Status check failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception during status check: {e}")
            return None
    
    def test_health_check(self):
        """Test the health endpoint"""
        try:
            url = f"{self.base_url}/health"
            response = self.session.get(url)
            
            logger.info(f"Health check status code: {response.status_code}")
            logger.info(f"Health check response: {response.text}")
            
            assert response.status_code in [200, 503, 504], f"Unexpected health check status: {response.status_code}"
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None


async def test_parser_api_integration():
    """Integration test function that can be run with pytest"""
    tester = TestParserAPI()
    
    # First check if service is healthy
    health_result = tester.test_health_check()
    if health_result is None:
        logger.warning("Service health check failed, but continuing with test...")
    
    # Run the main parsing test
    result = await tester.test_parse_file_with_status_monitoring()
    assert result["status"] == "completed", f"Parse job failed: {result}"


async def test_multiple_jobs():
    """Test multiple concurrent jobs"""
    tester = TestParserAPI()
    
    # Test data for multiple jobs
    test_cases = [
        {
            "input_s3_path": "s3://monkeyocr/test/input/test_pdf/small.pdf",
            "output_s3_path": "s3://monkeyocr/test/output/test_pdf/small",
            "knowledgebaseId": "1222",
            "workspaceId": "3444"
        },
        {
            "input_s3_path": "s3://monkeyocr/test/input/test_pdf/test.pdf",
            "output_s3_path": "s3://monkeyocr/test/output/test_pdf/test",
            "knowledgebaseId": "1223",
            "workspaceId": "3445"
        },
        {
            "input_s3_path": "s3://monkeyocr/test/input/test_pdf/test2.pdf",
            "output_s3_path": "s3://monkeyocr/test/output/test_pdf/test2",
            "knowledgebaseId": "1223",
            "workspaceId": "3445"
        }
    ]
    
    
    OCRJobIds = []
    for form_data in test_cases:
        response = tester._submit_parse_job(form_data)
        OCRJobIds.append(response["OCRJobId"])

    # 并行监控
    async def monitor(job_id):
        result = await tester._monitor_job_status(job_id)
        assert result["status"] == "completed", f"Job {job_id} failed: {result}"
        return result

    results = await asyncio.gather(*[monitor(job_id) for job_id in OCRJobIds])
    return results


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Running DotsOCR Parser API Tests...")
    
    # Test single job
    print("\n=== Testing Single Job ===")
    asyncio.run(test_parser_api_integration())
    
    # # Test multiple jobs
    # print("\n=== Testing Multiple Jobs ===")
    # asyncio.run(test_multiple_jobs())
    
    print("\n=== All Tests Completed Successfully! ===")
