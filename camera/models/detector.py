import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

class RTMDetector:
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.5,
        providers: List[str] = None
    ):
        """
        Initialize RTMDet detector for person detection
        
        Args:
            model_path: Path to RTMDet ONNX model
            input_size: Input size for RTMDet model (width, height)
            conf_threshold: Confidence threshold for detection
            providers: ONNX Runtime providers
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Load RTMDet model
        print(f"Loading RTMDet model from {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model information
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Person class ID (assuming 0 for person)
        self.person_class_id = 0
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess image for RTMDet inference
        
        Args:
            img: Input image in BGR format (OpenCV)
            
        Returns:
            preprocessed_img: Preprocessed image ready for inference
            ori_size: Original image size (height, width)
            resized_size: Resized image size (height, width)
        """
        # Resize image
        orig_h, orig_w = img.shape[:2]
        input_w, input_h = self.input_size
        
        # Calculate resize ratio
        ratio = min(input_w / orig_w, input_h / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        
        # Resize image
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # Create canvas with padding
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        
        # Paste resized image on canvas
        canvas[:new_h, :new_w, :] = resized_img
        
        # Convert to RGB and normalize
        canvas = canvas[:, :, ::-1]  # BGR to RGB
        canvas = canvas.astype(np.float32) / 255.0
        
        # Transpose to NCHW format (batch, channels, height, width)
        canvas = canvas.transpose(2, 0, 1)[np.newaxis, ...]
        
        return canvas, (orig_h, orig_w), (new_h, new_w)
    
    def postprocess(self, outputs: List[np.ndarray], ori_size: Tuple[int, int], resized_size: Tuple[int, int]) -> np.ndarray:
        """
        Process RTMDet outputs to get bounding boxes
        
        Args:
            outputs: Model output
            ori_size: Original image size (height, width)
            resized_size: Resized image size (height, width)
            
        Returns:
            dets: Detected objects [x1, y1, x2, y2, score, class]
        """
        # Extract output
        predictions = outputs[0]  # assuming output shape is [1, N, 5] or [1, N, 6]
        
        ori_h, ori_w = ori_size
        resized_h, resized_w = resized_size
        
        # Check the shape of predictions to determine the format
        bbox_dims = predictions.shape[2]
        
        # Filter by confidence threshold
        mask = predictions[0, :, 4] > self.conf_threshold
        
        # Filter predictions
        filtered_preds = predictions[0, mask, :]
        
        if len(filtered_preds) == 0:
            return np.array([])
        
        # For models with 5 columns, we don't have class info, so treat all as person
        if bbox_dims == 5:  # [x1, y1, x2, y2, score]
            # Create a new array with class information
            dets = np.zeros((filtered_preds.shape[0], 6), dtype=np.float32)
            dets[:, :5] = filtered_preds  # Copy bbox and score
            dets[:, 5] = self.person_class_id  # Set class to person
        else:  # [x1, y1, x2, y2, score, class, ...]
            # Extract person detections (class 0)
            person_mask = filtered_preds[:, 5] == self.person_class_id
            persons = filtered_preds[person_mask]
            
            if len(persons) == 0:
                return np.array([])
            
            # [x1, y1, x2, y2, score, class]
            dets = persons[:, :6].copy()
        
        # Scale bounding boxes to original image
        scale_w = ori_w / resized_w
        scale_h = ori_h / resized_h
        
        dets[:, 0] *= scale_w  # x1
        dets[:, 1] *= scale_h  # y1
        dets[:, 2] *= scale_w  # x2
        dets[:, 3] *= scale_h  # y2
        
        return dets
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect persons in image
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            detections: Array of [x1, y1, x2, y2, score, class]
        """
        # Preprocess
        preprocessed_img, ori_size, resized_size = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})
        
        # Postprocess
        detections = self.postprocess(outputs, ori_size, resized_size)
        
        return detections