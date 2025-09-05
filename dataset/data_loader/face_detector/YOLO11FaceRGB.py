"""Enhanced YOLO11Face detector with RGB mean extraction for rPPG preprocessing.

This module extends the existing YOLO11Face implementation to include
RGB signal extraction from detected face regions.
"""
from copy import deepcopy
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from dataset.data_loader import BaseLoader

class YOLO11FaceRGB(object):
    """YOLO11 face detector with RGB extraction capabilities."""
    
    def __init__(self, backend="Y11F", device=None) -> None:
        """Initialize YOLO11 face detector with RGB extraction.
        
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
                self.device = torch.device(dev_list[0])
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(0)
            else:
                self.device = torch.device("cpu")
        
        # Model selection based on backend
        package_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(package_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        model_map = {
            'Y11F': 'yolov11m-face.pt',
            'Y11F-M': 'yolov11m-face.pt',
            'Y11F-L': 'yolov11l.pt',
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
            self.model = YOLO(model_name)
            self.is_face_specific = False
            print(f"Downloaded and loaded general {backend} model")
        
        # Move model to device
        self.model.to(self.device)
        
        # Adjust thresholds based on model size
        if backend == 'Y11F-M':
            self.conf_thres = 0.5
        elif backend == 'Y11F-L':
            self.conf_thres = 0.45
    
    def detect_face(self, frame):
        """Original detect_face method for backward compatibility."""
        img0 = deepcopy(frame)
        h0, w0 = img0.shape[:2]
        
        img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        
        results = self.model(
            img_rgb,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device=self.device
        )
        
        if len(results) == 0 or results[0].boxes is None:
            return None
        
        boxes = results[0].boxes
        face_detections = []
        
        for box in boxes:
            conf = box.conf.item()
            if conf >= self.conf_thres:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                face_detections.append((x1, y1, x2, y2, conf))
        
        if not face_detections:
            return None
        
        best_face = max(face_detections, key=lambda x: x[4])
        x1, y1, x2, y2, _ = best_face
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def extract_face_rgb(self, frame, face_box=None):
        """Extract RGB mean values from detected face region.
        
        Args:
            frame (np.array): Input frame (H, W, 3) in BGR format
            face_box: Optional face box [x1, y1, x2, y2]. If None, detect face first.
            
        Returns:
            dict: RGB means {'R': float, 'G': float, 'B': float} or None
        """
        if face_box is None:
            face_box = self.detect_face(frame)
        
        if face_box is None:
            return None
        
        x1, y1, x2, y2 = face_box
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract face region
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        # Calculate RGB means (OpenCV uses BGR format)
        rgb_means = {
            'R': float(np.mean(face_roi[:, :, 2])),
            'G': float(np.mean(face_roi[:, :, 1])),
            'B': float(np.mean(face_roi[:, :, 0]))
        }
        
        return rgb_means
    
    def process_video_rgb(self, frames, use_median_box=True, detection_freq=30):
        """Process video frames to extract RGB signals from faces.
        
        Args:
            frames (np.array): Video frames (N, H, W, 3) in BGR format
            use_median_box (bool): Use median face box for stability
            detection_freq (int): Frequency of face detection when not using median
            
        Returns:
            tuple: (rgb_signals, face_boxes)
                - rgb_signals: np.array of shape (N, 3) with R, G, B channels
                - face_boxes: List of face boxes for each frame
        """
        n_frames = frames.shape[0]
        rgb_signals = np.zeros((n_frames, 3))
        face_boxes = []
        
        if use_median_box:
            # Detect faces in subset of frames for median calculation
            sample_indices = np.linspace(0, n_frames-1, min(10, n_frames), dtype=int)
            sample_boxes = []
            
            print("Detecting faces for median box calculation...")
            for idx in sample_indices:
                box = self.detect_face(frames[idx])
                if box is not None:
                    sample_boxes.append(box)
            
            if not sample_boxes:
                print("Warning: No faces detected in sample frames")
                return None, None
            
            # Calculate median box
            sample_boxes = np.array(sample_boxes)
            median_box = np.median(sample_boxes, axis=0).astype(int).tolist()
            
            print(f"Using median face box: {median_box}")
            
            # Extract RGB from all frames using median box
            for i, frame in enumerate(frames):
                rgb_means = self.extract_face_rgb(frame, median_box)
                if rgb_means:
                    rgb_signals[i] = [rgb_means['R'], rgb_means['G'], rgb_means['B']]
                else:
                    # Use previous values if extraction fails
                    if i > 0:
                        rgb_signals[i] = rgb_signals[i-1]
                face_boxes.append(median_box)
        else:
            # Dynamic detection mode
            print("Processing video with dynamic face detection...")
            last_valid_box = None
            last_valid_rgb = None
            
            for i, frame in enumerate(frames):
                # Detect face every detection_freq frames or if no valid box exists
                if i % detection_freq == 0 or last_valid_box is None:
                    box = self.detect_face(frame)
                    if box is not None:
                        last_valid_box = box
                else:
                    box = last_valid_box
                
                face_boxes.append(box)
                
                if box is not None:
                    rgb_means = self.extract_face_rgb(frame, box)
                    if rgb_means:
                        rgb_signals[i] = [rgb_means['R'], rgb_means['G'], rgb_means['B']]
                        last_valid_rgb = rgb_signals[i]
                    elif last_valid_rgb is not None:
                        rgb_signals[i] = last_valid_rgb
                elif last_valid_rgb is not None:
                    rgb_signals[i] = last_valid_rgb
        
        return rgb_signals, face_boxes


# Extension functions for MMPDLoader integration
def extract_rgb_signals_from_mmpd(frames, y11f_obj, use_median_box=True):
    """Extract RGB signals from MMPD video frames.
    
    Args:
        frames (np.array): Video frames from MMPD dataset
        y11f_obj: YOLO11FaceRGB object
        use_median_box (bool): Whether to use median box for stability
        
    Returns:
        dict: RGB signals with keys 'R', 'G', 'B' as numpy arrays
    """
    # Ensure frames are in uint8 format
    if frames.dtype != np.uint8:
        if np.max(frames) <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Process video to get RGB signals
    rgb_signals, face_boxes = y11f_obj.process_video_rgb(frames, use_median_box=use_median_box)
    
    if rgb_signals is None:
        print("Failed to extract RGB signals")
        return None
    
    # Convert to dictionary format
    rgb_dict = {
        'R': rgb_signals[:, 0],
        'G': rgb_signals[:, 1],
        'B': rgb_signals[:, 2]
    }
    
    return rgb_dict


def save_rgb_signals(rgb_signals, output_path, subject_info):
    """Save extracted RGB signals to file.
    
    Args:
        rgb_signals (dict): RGB signals dictionary
        output_path (str): Base path for saving
        subject_info (str): Subject and condition information string
    """
    if rgb_signals is None:
        return
    
    # Create output directory if it doesn't exist
    rgb_output_dir = os.path.join(output_path, 'rgb_signals')
    os.makedirs(rgb_output_dir, exist_ok=True)
    
    # Save as numpy array
    rgb_array = np.stack([rgb_signals['R'], rgb_signals['G'], rgb_signals['B']], axis=1)
    output_file = os.path.join(rgb_output_dir, f"{subject_info}_rgb.npy")
    np.save(output_file, rgb_array)
    print(f"Saved RGB signals to {output_file}")


# Modified preprocess_dataset_subprocess for MMPDLoader
def preprocess_dataset_subprocess_with_rgb(self, data_dirs, config_preprocess, i, file_list_dict):
    """Modified preprocessing subprocess that includes RGB extraction.
    
    This function should replace the original preprocess_dataset_subprocess in MMPDLoader.
    """
    frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup \
        = self.read_mat(data_dirs[i]['path'])
    
    saved_filename = 'subject' + str(data_dirs[i]['subject'])
    saved_filename += f'_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}'
    
    frames = (np.round(frames * 255)).astype(np.uint8)
    
    # Extract RGB signals if YOLO11 is available
    rgb_signals = None
    if hasattr(self, 'Y11FObj') and isinstance(self.Y11FObj, YOLO11FaceRGB):
        print(f"Extracting RGB signals for {saved_filename}...")
        rgb_signals = extract_rgb_signals_from_mmpd(frames, self.Y11FObj, use_median_box=True)
        
        # Optionally save RGB signals separately
        if rgb_signals is not None and hasattr(self.config_data, 'SAVE_RGB_SIGNALS') and self.config_data.SAVE_RGB_SIGNALS:
            save_rgb_signals(rgb_signals, self.cached_path, saved_filename)
    
    target_length = frames.shape[0]
    bvps = BaseLoader.resample_ppg(bvps, target_length)
    frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
    
    # If RGB signals were extracted, chunk them as well
    if rgb_signals is not None and config_preprocess.DO_CHUNK:
        rgb_array = np.stack([rgb_signals['R'], rgb_signals['G'], rgb_signals['B']], axis=1)
        chunk_length = config_preprocess.CHUNK_LENGTH
        clip_num = len(frames_clips)
        rgb_clips = [rgb_array[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        
        # Save RGB clips alongside frame and BVP clips
        for j, rgb_clip in enumerate(rgb_clips):
            rgb_path = self.cached_path + os.sep + f"{saved_filename}_rgb{j}.npy"
            np.save(rgb_path, rgb_clip)
    
    input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
    file_list_dict[i] = input_name_list


# Example usage
if __name__ == "__main__":
    # Initialize YOLO11 with RGB extraction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo11_rgb = YOLO11FaceRGB(backend='Y11F-M', device=device)
    
    # Example: Create dummy frames for testing
    test_frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
    
    # Extract RGB signals
    rgb_signals, face_boxes = yolo11_rgb.process_video_rgb(test_frames)
    
    if rgb_signals is not None:
        print(f"RGB signals shape: {rgb_signals.shape}")
        print(f"Mean R: {np.mean(rgb_signals[:, 0]):.2f}")
        print(f"Mean G: {np.mean(rgb_signals[:, 1]):.2f}")
        print(f"Mean B: {np.mean(rgb_signals[:, 2]):.2f}")
    else:
        print("No faces detected in test frames")