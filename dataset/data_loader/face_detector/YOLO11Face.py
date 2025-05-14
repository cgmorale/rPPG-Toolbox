"""YOLO11Face detector for face detection in rPPG preprocessing.

Follows the same interface as YOLO5Face for compatibility.
"""
from copy import deepcopy
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


class YOLO11Face(object):
    def __init__(self, backend="Y11F", device=None) -> None:
        """Initialize YOLO11 face detector.
        
        Args:
            backend (str): Backend identifier (Y11F, Y11F-M, Y11F-L)
            device: Device specification for model inference
        """
        self.conf_thres = 0.6
        self.iou_thres = 0.5
        self.img_size = 640
        
        # Device configuration (matching YOLO5Face logic)
        if device is not None:
            if device == "cpu":
                self.device = torch.device("cpu")
            elif torch.cuda.is_available():
                dev_list = [int(d) for d in device.replace("cuda:", "").split(",")]
                self.device = torch.device(dev_list[0])  # currently toolbox only supports 1 GPU
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(0)  # currently toolbox only supports 1 GPU
            else:
                self.device = torch.device("cpu")
        
        # Model selection based on backend
        package_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(package_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        model_map = {
            'Y11F': 'yolov11n.pt',       # Nano
            'Y11F-M': 'yolov11m-face.pt',     # Medium
            'Y11F-L': 'yolov11l.pt',     # Large
        }
        
        if backend not in model_map:
            raise ValueError(f"Unknown YOLO11 backend: {backend}")
        
        model_name = model_map[backend]
        
        # Try face-specific weights first
        face_weights_path = os.path.join(weights_dir, f"{model_name.replace('.pt', '-face.pt')}")
        general_weights_path = os.path.join(weights_dir, model_name)
        
        if os.path.exists(face_weights_path):
            self.model = YOLO(face_weights_path)
            self.is_face_specific = True
            print(f"Loaded face-specific {backend} model from {face_weights_path}")
        elif os.path.exists(general_weights_path):
            self.model = YOLO(general_weights_path)
            self.is_face_specific = False
            print(f"Loaded general {backend} model from {general_weights_path}")
        else:
            # Let YOLO download the model automatically
            self.model = YOLO(model_name)
            self.is_face_specific = False
            print(f"Downloaded and loaded general {backend} model")
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        
        # Adjust thresholds based on model size
        if backend == 'Y11F-M':
            self.conf_thres = 0.5  # Medium model is more accurate
        elif backend == 'Y11F-L':
            self.conf_thres = 0.45  # Large model is most accurate
    
    def detect_face(self, frame):
        img0 = deepcopy(frame)
        h0, w0 = img0.shape[:2]
        
        # Let YOLO handle all preprocessing including letterboxing
        # Don't pre-resize, let YOLO do it
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        # YOLO will internally:
        # 1. Letterbox to square
        # 2. Resize to img_size
        # 3. Normalize and convert to tensor
        results = self.model(
            img_rgb,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=self.device
        )
        
        # The results are already in original image coordinates
        # when using ultralytics YOLO
        if len(results) == 0 or results[0].boxes is None:
            return None
        
        boxes = results[0].boxes
        face_detections = []
        
        for box in boxes:
            conf = box.conf.item()
            if conf >= self.conf_thres:
                # These coordinates are already in original image space
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Just convert to integers
                face_detections.append((x1, y1, x2, y2, conf))
        
        if not face_detections:
            return None
        
        best_face = max(face_detections, key=lambda x: x[4])
        x1, y1, x2, y2, _ = best_face
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
