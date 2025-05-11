import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, List

class RTMPoseEstimator:
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (192, 256),  # (width, height)
        providers: List[str] = None
    ):
        """
        Initialize RTMPose estimator for human pose estimation
        
        Args:
            model_path: Path to RTMPose ONNX model
            input_size: Input size for RTMPose model (width, height)
            providers: ONNX Runtime providers
        """
        self.input_size = input_size
        
        # Default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Load RTMPose model
        print(f"Loading RTMPose model from {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model information
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Define keypoint connections for visualization
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], 
            [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], 
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]
        
        # Define keypoint names (COCO format)
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def preprocess(self, img: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Preprocess image for RTMPose inference
        
        Args:
            img: Original image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            preprocessed_img: Cropped and resized person image
            crop_box: Crop box coordinates [x1, y1, x2, y2]
        """
        input_w, input_h = self.input_size
        
        # Get bbox coordinates
        x1, y1, x2, y2 = bbox[:4].astype(int)
        
        # Add some margin to the bbox
        h, w = y2 - y1, x2 - x1
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Make the box square with a better aspect ratio for the pose model
        side_length = max(w, h) * 1.2
        
        # Calculate new box coordinates
        new_x1 = max(0, int(center_x - side_length / 2))
        new_y1 = max(0, int(center_y - side_length / 2))
        new_x2 = min(img.shape[1], int(center_x + side_length / 2))
        new_y2 = min(img.shape[0], int(center_y + side_length / 2))
        
        # Crop image
        cropped_img = img[new_y1:new_y2, new_x1:new_x2]
        
        # Resize to input size
        resized_img = cv2.resize(cropped_img, (input_w, input_h))
        
        # Convert to RGB and normalize
        resized_img = resized_img[:, :, ::-1]  # BGR to RGB
        resized_img = resized_img.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        resized_img = (resized_img - mean) / std
        
        # Transpose to NCHW format
        preprocessed_img = resized_img.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Return preprocessed image and transformation info for postprocessing
        return preprocessed_img, (new_x1, new_y1, new_x2, new_y2)
    
    def postprocess(self, output: List[np.ndarray], crop_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Process RTMPose output to get keypoints
        
        Args:
            output: Model output
            crop_box: Crop box coordinates [x1, y1, x2, y2]
            
        Returns:
            keypoints: Keypoints with coordinates and confidence scores
        """
        # Extract heatmaps from output
        heatmap = output[0]  # Assuming shape [1, 17, H, W]
        
        # Get keypoint locations from heatmaps
        num_keypoints = heatmap.shape[1]
        keypoints = []
        
        # Get crop box coordinates
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_h, crop_w = crop_y2 - crop_y1, crop_x2 - crop_x1
        
        # Scale factors to map back to original image
        scale_x = crop_w / self.input_size[0]
        scale_y = crop_h / self.input_size[1]
        
        # Extract keypoints from heatmaps
        for k in range(num_keypoints):
            heatmap_k = heatmap[0, k]
            
            # Find position with max confidence
            ind = np.unravel_index(np.argmax(heatmap_k), heatmap_k.shape)
            y, x = ind[0], ind[1]
            
            # Get confidence score
            conf = heatmap_k[y, x]
            
            # Scale keypoint coordinates back to original image
            orig_x = x * scale_x + crop_x1
            orig_y = y * scale_y + crop_y1
            
            keypoints.append([orig_x, orig_y, conf])
        
        return np.array(keypoints)
    
    def estimate(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Estimate pose from image and bounding box
        
        Args:
            img: Input image (BGR format)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            keypoints: Array of [x, y, confidence] for each keypoint
        """
        # Preprocess
        preprocessed_img, crop_box = self.preprocess(img, bbox)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed_img})
        
        # Postprocess
        keypoints = self.postprocess(outputs, crop_box)
        
        return keypoints