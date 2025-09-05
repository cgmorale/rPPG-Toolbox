"""Modified MMPDLoader with YOLO11 face detection and RGB extraction to CSV.

This implementation integrates with your existing config and saves RGB means
as separate CSV files for each video.
"""

import os
import cv2
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from .BaseLoader import BaseLoader


class MMPDLoaderWithRGB(BaseLoader):
    """Extended MMPD dataloader with RGB extraction capabilities."""
    
    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an MMPD dataloader with RGB extraction.
        
        Args:
            name(string): name of the dataloader.
            data_path(str): path of a folder which stores raw video and bvp data.
            config_data(CfgNode): data settings from config file.
            device: Device for YOLO11 model.
        """
        self.info = config_data.INFO
        
        # Initialize YOLO11 for face detection if specified
        if hasattr(config_data.PREPROCESS.CROP_FACE, 'BACKEND') and \
           'Y11F' in config_data.PREPROCESS.CROP_FACE.BACKEND:
            # Import the enhanced YOLO11 with RGB capabilities
            from dataset.data_loader.face_detector.YOLO11FaceRGB import YOLO11FaceRGB
            self.use_yolo11_rgb = True
            print("Initializing YOLO11 with RGB extraction capabilities...")
        else:
            self.use_yolo11_rgb = False
        
        # Check if we should save RGB signals
        self.save_rgb_signals = getattr(config_data, 'SAVE_RGB_SIGNALS', False)
        
        # Create RGB output directory if needed
        if self.save_rgb_signals:
            self.rgb_output_path = os.path.join(
                config_data.CACHED_PATH, 
                'rgb_signals'
            )
            os.makedirs(self.rgb_output_path, exist_ok=True)
            print(f"RGB signals will be saved to: {self.rgb_output_path}")
        
        super().__init__(name, data_path, config_data, device)
    
    def get_raw_data(self, raw_data_path):
        """Returns data directories under the path (For MMPD dataset)."""
        data_dirs = glob.glob(raw_data_path + os.sep + 'subject*')
        if not data_dirs:
            raise ValueError(self.dataset_name + ' data paths empty!')
        dirs = list()
        for data_dir in data_dirs:
            subject = int(os.path.split(data_dir)[-1][7:])
            mat_dirs = os.listdir(data_dir)
            for mat_dir in mat_dirs:
                index = mat_dir.split('_')[-1].split('.')[0]
                dirs.append({
                    'index': index,
                    'path': data_dir + os.sep + mat_dir,
                    'subject': subject
                })
        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:
            return data_dirs
        
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            if subject not in data_info:
                data_info[subject] = list()
            data_info[subject].append(data)
        
        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)
        
        subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))
        print('Used subject ids for split:', [subj_list[i] for i in subj_range])
        
        data_dirs_new = list()
        for i in subj_range:
            subj_num = subj_list[i]
            data_dirs_new += data_info[subj_num]
        
        return data_dirs_new
    
    def extract_and_save_rgb_signals(self, frames, saved_filename):
        """Extract RGB signals from frames and save to CSV.
        
        Args:
            frames (np.array): Video frames (N, H, W, 3)
            saved_filename (str): Base filename for saving
            
        Returns:
            rgb_signals (dict): Dictionary with 'R', 'G', 'B' arrays
        """
        if not self.use_yolo11_rgb or not hasattr(self, 'Y11FObj'):
            return None
        
        print(f"Extracting RGB signals for {saved_filename}...")
        
        # Ensure frames are uint8
        if frames.dtype != np.uint8:
            if np.max(frames) <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
        
        # Extract RGB signals using YOLO11
        rgb_signals, face_boxes = self.Y11FObj.process_video_rgb(
            frames, 
            use_median_box=self.config_data.PREPROCESS.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            detection_freq=self.config_data.PREPROCESS.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY
        )
        
        if rgb_signals is None:
            print(f"Warning: Failed to extract RGB signals for {saved_filename}")
            return None
        
        # Convert to dictionary format
        rgb_dict = {
            'R': rgb_signals[:, 0],
            'G': rgb_signals[:, 1],
            'B': rgb_signals[:, 2]
        }
        
        # Save to CSV if enabled
        if self.save_rgb_signals:
            self.save_rgb_to_csv(rgb_dict, saved_filename, face_boxes)
        
        return rgb_dict
    
    def save_rgb_to_csv(self, rgb_signals, saved_filename, face_boxes=None):
        """Save RGB signals to a CSV file.
        
        Args:
            rgb_signals (dict): Dictionary with 'R', 'G', 'B' arrays
            saved_filename (str): Base filename for the CSV
            face_boxes (list): Optional list of face bounding boxes
        """
        # Create DataFrame
        df_data = {
            'frame': np.arange(len(rgb_signals['R'])),
            'R': rgb_signals['R'],
            'G': rgb_signals['G'],
            'B': rgb_signals['B']
        }
        
        # Add face box coordinates if available
        if face_boxes is not None and len(face_boxes) > 0:
            # Convert face boxes to separate columns
            face_boxes_array = np.array(face_boxes)
            if face_boxes_array.shape[1] == 4:
                df_data['face_x1'] = face_boxes_array[:, 0]
                df_data['face_y1'] = face_boxes_array[:, 1]
                df_data['face_x2'] = face_boxes_array[:, 2]
                df_data['face_y2'] = face_boxes_array[:, 3]
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        csv_filename = os.path.join(self.rgb_output_path, f"{saved_filename}_rgb.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Saved RGB signals to: {csv_filename}")
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocessing subprocess with RGB extraction."""
        # Read mat file
        frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup \
            = self.read_mat(data_dirs[i]['path'])
        
        # Create filename with metadata
        saved_filename = 'subject' + str(data_dirs[i]['subject'])
        saved_filename += f'_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}'
        
        # Convert frames to uint8
        frames = (np.round(frames * 255)).astype(np.uint8)
        
        # Extract and save RGB signals if YOLO11 is available
        rgb_signals = None
        if self.use_yolo11_rgb and self.save_rgb_signals:
            rgb_signals = self.extract_and_save_rgb_signals(frames, saved_filename)
        
        # Continue with standard preprocessing
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        
        # Save preprocessed frames and labels
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        
        # If RGB signals were extracted and chunking is enabled, save chunked RGB as well
        if rgb_signals is not None and config_preprocess.DO_CHUNK:
            self.save_chunked_rgb_signals(rgb_signals, saved_filename, config_preprocess.CHUNK_LENGTH)
        
        file_list_dict[i] = input_name_list
    
    def save_chunked_rgb_signals(self, rgb_signals, saved_filename, chunk_length):
        """Save chunked RGB signals as separate CSV files.
        
        Args:
            rgb_signals (dict): RGB signals dictionary
            saved_filename (str): Base filename
            chunk_length (int): Length of each chunk
        """
        rgb_array = np.stack([rgb_signals['R'], rgb_signals['G'], rgb_signals['B']], axis=1)
        clip_num = len(rgb_array) // chunk_length
        
        for j in range(clip_num):
            start_idx = j * chunk_length
            end_idx = (j + 1) * chunk_length
            chunk_data = {
                'frame': np.arange(chunk_length),
                'R': rgb_array[start_idx:end_idx, 0],
                'G': rgb_array[start_idx:end_idx, 1],
                'B': rgb_array[start_idx:end_idx, 2]
            }
            
            chunk_df = pd.DataFrame(chunk_data)
            chunk_csv_path = os.path.join(
                self.rgb_output_path, 
                f"{saved_filename}_chunk{j}_rgb.csv"
            )
            chunk_df.to_csv(chunk_csv_path, index=False)
    
    def read_mat(self, mat_file):
        """Read mat file and extract data."""
        try:
            mat = sio.loadmat(mat_file)
        except:
            for _ in range(20):
                print(mat_file)
        
        frames = np.array(mat['video'])
        
        if self.config_data.PREPROCESS.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = np.array(mat['GT_ppg']).T.reshape(-1)
        
        light = mat['light']
        motion = mat['motion']
        exercise = mat['exercise']
        skin_color = mat['skin_color']
        gender = mat['gender']
        glasser = mat['glasser']
        hair_cover = mat['hair_cover']
        makeup = mat['makeup']
        information = [light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup]
        
        light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup = self.get_information(information)
        
        return frames, bvps, light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
    
    @staticmethod
    def get_information(information):
        """Parse MMPD metadata information."""
        light = ''
        if information[0] == 'LED-low':
            light = 1
        elif information[0] == 'LED-high':
            light = 2
        elif information[0] == 'Incandescent':
            light = 3
        elif information[0] == 'Nature':
            light = 4
        else:
            raise ValueError(f"Unsupported lighting label: {information[0]}")
        
        motion = ''
        if information[1] == 'Stationary' or information[1] == 'Stationary (after exercise)':
            motion = 1
        elif information[1] == 'Rotation':
            motion = 2
        elif information[1] == 'Talking':
            motion = 3
        elif information[1] == 'Walking' or information[1] == 'Watching Videos':
            motion = 4
        else:
            raise ValueError(f"Unsupported motion label: {information[1]}")
        
        exercise = 1 if information[2] == 'True' else 2
        skin_color = information[3][0][0]
        gender = 1 if information[4] == 'male' else 2
        glasser = 1 if information[5] == 'True' else 2
        hair_cover = 1 if information[6] == 'True' else 2
        makeup = 1 if information[7] == 'True' else 2
        
        return light, motion, exercise, skin_color, gender, glasser, hair_cover, makeup
    
    def load_preprocessed_data(self):
        """Load preprocessed data with filtering based on config."""
        file_list_path = self.file_list_path
        file_list_df = pd.read_csv(file_list_path)
        inputs_temp = file_list_df['input_files'].tolist()
        inputs = []
        
        for each_input in inputs_temp:
            info = each_input.split(os.sep)[-1].split('_')
            light = int(info[1][-1])
            motion = int(info[2][-1])
            exercise = int(info[3][-1])
            skin_color = int(info[4][-1])
            gender = int(info[5][-1])
            glasser = int(info[6][-1])
            hair_cover = int(info[7][-1])
            makeup = int(info[8][-1])
            
            if (light in self.info.LIGHT) and (motion in self.info.MOTION) and \
               (exercise in self.info.EXERCISE) and (skin_color in self.info.SKIN_COLOR) and \
               (gender in self.info.GENDER) and (glasser in self.info.GLASSER) and \
               (hair_cover in self.info.HAIR_COVER) and (makeup in self.info.MAKEUP):
                inputs.append(each_input)
        
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        inputs = sorted(inputs)
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)


# Utility function to read saved RGB CSV files
def read_rgb_csv(csv_path):
    """Read RGB signals from saved CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: RGB signals and face boxes if available
    """
    df = pd.read_csv(csv_path)
    
    result = {
        'R': df['R'].values,
        'G': df['G'].values,
        'B': df['B'].values,
        'frames': df['frame'].values
    }
    
    # Check if face box coordinates are available
    if 'face_x1' in df.columns:
        result['face_boxes'] = df[['face_x1', 'face_y1', 'face_x2', 'face_y2']].values
    
    return result


# Example of how to use the saved RGB data
def analyze_rgb_signals(rgb_csv_path):
    """Analyze RGB signals from a saved CSV file.
    
    Args:
        rgb_csv_path (str): Path to RGB CSV file
        
    Returns:
        dict: Analysis results
    """
    rgb_data = read_rgb_csv(rgb_csv_path)
    
    analysis = {
        'mean_R': np.mean(rgb_data['R']),
        'mean_G': np.mean(rgb_data['G']),
        'mean_B': np.mean(rgb_data['B']),
        'std_R': np.std(rgb_data['R']),
        'std_G': np.std(rgb_data['G']),
        'std_B': np.std(rgb_data['B']),
        'num_frames': len(rgb_data['frames'])
    }
    
    # Calculate signal quality metrics
    analysis['snr_R'] = analysis['mean_R'] / analysis['std_R'] if analysis['std_R'] > 0 else 0
    analysis['snr_G'] = analysis['mean_G'] / analysis['std_G'] if analysis['std_G'] > 0 else 0
    analysis['snr_B'] = analysis['mean_B'] / analysis['std_B'] if analysis['std_B'] > 0 else 0
    
    return analysis