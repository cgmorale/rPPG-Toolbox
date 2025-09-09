"""The dataloader for Field Test dataset.

This loader uses VolleyBallCourtFileReader to access field test data.
The data comes from the VolleyBall Court Field Test recordings.

Usage:
    The data_path parameter is not used since data comes from VolleyBallCourtFileReader.
    The loader automatically discovers all valid participant/condition combinations
    using VolleyBallCourtFileReader.list_all_valid_indices().
    
    Each data entry corresponds to a (participant_id, condition_id) pair.
    Video frames and PPG data are automatically synchronized using interpolation.
"""
import os
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.field_test import VolleyBallCourtFileReader


class FIELDLoader(BaseLoader):
    """The data loader for the Field Test dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a Field Test dataloader.
        Args:
            name(string): name of the dataloader.
            data_path(str): path is not used for field data, data comes from VolleyBallCourtFileReader
            config_data(CfgNode): data settings(ref:config.py).
            device: device for face detection models
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories for Field Test dataset using VolleyBallCourtFileReader."""
        # Use VolleyBallCourtFileReader to get all valid indices
        all_indices = VolleyBallCourtFileReader.list_all_valid_indices()
        
        if not all_indices:
            raise ValueError(self.dataset_name + " data indices empty!")
        
        # Convert to the format expected by BaseLoader
        dirs = []
        for entry in all_indices:
            dirs.append({
                "index": f"participant{entry['participant_id']}_condition{entry['condition_id']}",
                "path": f"field_data",  # dummy path, actual data comes from FileReader
                "participant_id": entry['participant_id'],
                "condition_id": entry['condition_id']
            })
        
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
    
    def multi_process_manager(self, data_dirs, config_preprocess):
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota=4)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        data_entry = data_dirs[i]
        filename = data_entry['index']
        saved_filename = filename
        
        # Use VolleyBallCourtFileReader to get the actual data
        file_reader = VolleyBallCourtFileReader.from_index(
            participant_id=data_entry['participant_id'],
            condition_id=data_entry['condition_id'],
            offset_setup=True  # Use offset setup as default
        )
        
        if file_reader is None:
            print(f"Warning: Could not load data for {filename}")
            file_list_dict[i] = []
            return
        
        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            frames = file_reader.get_frames()
        elif 'Motion' in config_preprocess.DATA_AUG:
            raise ValueError(f'Motion data augmentation not supported for {self.dataset_name} dataset!')
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels (PPG data)
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = file_reader.get_ppg()['value']
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list