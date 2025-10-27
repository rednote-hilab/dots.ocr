from loguru import logger
from typing import Union, List, Dict, Any
from paddleocr import LayoutDetection


_layout_detection_model_service = None

class LayoutDetectionService():
    def __init__(
        self,
        model_name="PP-DocLayout_plus-L",
        batch_size=1
    ):
        self._model_name = model_name
        self._model_service = LayoutDetection(model_name=model_name)
        self._batch_size = batch_size

    def _transform_result(
        self,
        result: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Transform result to keep only label and bbox with float values.
        Both single and batch results return a list.
        """
        # TODO(zihao): align the category label format with dotsocr
        def transform_single(item: Dict[str, Any]) -> Dict[str, Any]:
            transformed_boxes = [
                {
                    'category': bbox['label'],
                    'bbox': [float(coord) for coord in bbox['coordinate']]
                }
                for bbox in item.get('boxes', [])
            ]
            img = (item.img)['res'] # PP-DocLayout_plus-L will resize the image but I don't find the method. So we need to recover the original size outside.
            width, height = img.size
            return {
                'page_no': item['page_index'],
                'width': width,
                'height': height,
                'full_layout_info': transformed_boxes
            }
        
        if isinstance(result, list):
            return [transform_single(item) for item in result]
        else:
            return [transform_single(result)]

    def _get_layout_image(
        self,
        image_path: Union[str, List[str]]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get layout detection results.
        
        Args:
            image_path: str or list of str - single image path or list of image paths
        
        Returns:
            Dict or List[Dict]: Layout detection result(s).

            - Single image: {'input_path': str, 'page_index': None, 'boxes': List[Dict]}
            Each box in 'boxes' contains:
                - 'label': str - label name (e.g., 'paragraph_title')
                - 'bbox': List[float] - [x1, y1, x2, y2] bounding box coordinates

            - Multiple images: List of the above structure
        """
        result = self._model_service.predict(image_path, batch_size=self._batch_size, layout_nms=True)
        return self._transform_result(result)

    def _get_layout_pdf(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Get layout detection results for a PDF file.
        Args:
            file_path: str - path to the PDF file
        Returns:
            List of Dict: Layout detection results for each page in the PDF. format same as get_layout_image.
        """
        result = self._model_service.predict(file_path, batch_size=self._batch_size, layout_nms=True)
        return self._transform_result(result)

def get_model_service() -> LayoutDetectionService:
    global _layout_detection_model_service
    if _layout_detection_model_service is None:
        logger.info("Loading layout detection model...")
        _layout_detection_model_service = LayoutDetectionService()
    return _layout_detection_model_service

def get_layout_image(image_path: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    model_service = get_model_service()
    return model_service._get_layout_image(image_path)

def get_layout_pdf(file_path: str) -> Dict[str, Any]:
    model_service = get_model_service()
    return model_service._get_layout_pdf(file_path)
