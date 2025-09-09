from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import itertools
from typing import Optional
from pathlib import Path
import re
from typing import Optional
import warnings

import cv2
import numpy as np
from numpy.typing import NDArray
import pandas as pd




class FileReader:
    FIELD_TEST_DIRECTORY = (Path('/zfsauton/data/straps/ChironFieldTest')).resolve(strict=True)
    frame_glob_pattern = 'image_*.png'
    timestamp_regex = re.compile(r'image_(\d+)\.png$')

    def __init__(
        self,
        directory: Path | str,
        frames_directory: Optional[Path | str] = None,
        heart_rate_filepath: Optional[Path | str] = None,
        ppg_filepath: Optional[Path | str] = None,
        runlog_filepath: Optional[Optional[Path | str]] = None,
        offset_setup: bool = False,
    ) -> None:
        self.directory = Path(directory).resolve(strict=True)
        self.offset_setup = offset_setup
        
        if frames_directory is None:
            frames_directory = self.directory / 'images'
        self.frames_directory = Path(frames_directory)
        if heart_rate_filepath is None:
            heart_rate_filepath = self.directory / 'heart_rate.csv'
        self.heart_rate_filepath = Path(heart_rate_filepath)
        if ppg_filepath is None:
            if offset_setup:
                # Replace ChironFieldTest with ChironFieldTestOffset in the path
                offset_directory = Path(str(self.directory).replace('ChironFieldTest', 'ChironFieldTestOffset'))
                ppg_filepath = offset_directory / 'offset_ppg_waveform.csv'
            else:
                ppg_filepath = self.directory / 'ppg_waveform.csv'
        self.ppg_filepath = Path(ppg_filepath)
        if runlog_filepath is None:
            runlog_filepath = self.directory.parent / 'runlog.csv'
        self.runlog_filepath = Path(runlog_filepath)

        self._check_existence()
        self._init_cache()
    

    def _check_existence(self) -> None:
        if not self.frames_directory.exists():
            raise FileNotFoundError(f'Frames directory {self.frames_directory} does not exist.')
        if not self.frames_directory.is_dir():
            raise NotADirectoryError(f'{self.frames_directory} is not a directory.')

        # DO NOT MERGE THIS
        # if not self.heart_rate_filepath.exists():
        #     raise FileNotFoundError(f'Heart rate file {self.heart_rate_filepath} does not exist.')
        # if not self.heart_rate_filepath.is_file():
        #     raise ValueError(f'{self.heart_rate_filepath} is not a file.')
        # if not self.heart_rate_filepath.suffix == '.csv':
        #     raise ValueError(f'Heart rate file {self.heart_rate_filepath} is not a CSV file.')
        
        if not self.ppg_filepath.exists():
            raise FileNotFoundError(f'PPG file {self.ppg_filepath} does not exist.')
        if not self.ppg_filepath.is_file():
            raise ValueError(f'{self.ppg_filepath} is not a file.')
        if not self.ppg_filepath.suffix == '.csv':
            raise ValueError(f'PPG file {self.ppg_filepath} is not a CSV file.')
        
        if not self.runlog_filepath.exists():
            raise FileNotFoundError(f'Runlog file {self.runlog_filepath} does not exist.')
        if not self.runlog_filepath.is_file():
            raise ValueError(f'{self.runlog_filepath} is not a file.')
        if not self.runlog_filepath.suffix == '.csv':
            raise ValueError(f'Runlog file {self.runlog_filepath} is not a CSV file.')
    
    def _init_cache(self) -> None:
        self._frame_paths: Optional[list[Path]] = None
        self._frame_timestamps: Optional[NDArray[np.float64]] = None
        self._frame_rate: Optional[float] = None
        self._frame_rate_stdev: Optional[float] = None
    
    def get_frame_paths(self) -> list[Path]:
        if self._frame_paths is None:
            self._frame_paths = sorted(self.frames_directory.glob(self.frame_glob_pattern))
        return self._frame_paths
    
    def get_nframes(self) -> int:
        return len(self.get_frame_paths())
    
    def get_timestamps(self) -> NDArray[np.float64]:
        if self._frame_timestamps is None:
            filepaths = self.get_frame_paths()
            self._frame_timestamps = np.array([
                int(self.timestamp_regex.findall(filepath.name)[0])
                for filepath in filepaths
            ]) / 1e9  # Convert from nanoseconds to seconds
        return self._frame_timestamps
    
    def get_frame(self, index: int) -> NDArray[np.uint8]:
        frame_path = self.get_frame_paths()[index]
        frame = cv2.imread(str(frame_path))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    def get_frames(
        self,
        indices: slice | list[int] | ellipsis = ...,
        max_workers: int = -1,
    ) -> NDArray[np.uint8]:
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(self.get_nframes())))
        elif isinstance(indices, list):
            indices = indices
        elif indices is ...:
            indices = list(range(self.get_nframes()))

        first_frame = cv2.imread(str(self.get_frame_paths()[0]))
        result = np.empty(
            shape=(len(indices), first_frame.shape[0], first_frame.shape[1], 3),
            dtype=np.uint8,
        )
        
        if max_workers == 0:  # No multithreading
            for i, idx in enumerate(indices):
                frame_path = self.get_frame_paths()[idx]
                frame = cv2.imread(str(frame_path))
                result[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        else:
            with ThreadPoolExecutor(max_workers=None if max_workers < 0 else max_workers) as executor:
                futures = {
                    executor.submit(self.get_frame, idx): idx
                    for idx in indices
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    result[idx] = future.result()
        
        return result
    
    def get_frame_rate(self) -> float:
        if self._frame_rate is None:
            timestamps = self.get_timestamps()
            self._frame_rate = float(1.0 / np.mean(np.diff(timestamps)))
        return self._frame_rate
    
    def get_frame_rate_stdev(self) -> float:
        if self._frame_rate_stdev is None:
            timestamps = self.get_timestamps()
            self._frame_rate_stdev = float(np.std(np.diff(timestamps)))
        return self._frame_rate_stdev
    
    def get_heart_rates(self, return_all=False) -> pd.DataFrame:
        df = pd.read_csv(self.heart_rate_filepath)
        if return_all:
            return df
        result = df[['timestamp', 'linear']].copy()
        result = result.rename(columns={'linear': 'value'})  # type: ignore
        return result

    def get_ppg(self, return_all=False) -> pd.DataFrame:
        df = pd.read_csv(self.ppg_filepath)
        if return_all:
            return df
        result = df[['timestamp', 'linear']].copy()
        result = result.rename(columns={'linear': 'value'})  # type: ignore
        return result
    
    def get_runlog(self, return_all=False) -> dict | pd.DataFrame:
        """Returns the runlog.

        Parameters
        ----------
        return_all : bool, default False
            If True, returns the entire runlog as a DataFrame. If False, returns only the entry for the current directory as a dictionary. Default is False.
        
        Returns
        -------
        runlog_entry : dict or pd.DataFrame
            If `return_all` is False, returns a dictionary with the runlog entry for the current directory. If `return_all` is True, returns the entire runlog as a DataFrame.
        """
        df = pd.read_csv(self.runlog_filepath)
        if return_all:
            return df
        
        # Find the row that matches column "directory_name"
        df = df.query(f'directory_name == "{self.directory.name}"')
        if len(df) == 0:
            raise ValueError(f'No runlog entry found for directory {self.directory.name}.')
        if len(df) > 1:
            raise ValueError(f'Multiple runlog entries found for directory {self.directory.name}.')
        return df.iloc[0].to_dict()
    
    def get_readme(self, print_=True) -> str:
        """Return the full README file as a string.

        This includes information about all dataset recorded on the same day,
        not just the current recording.
        """
        result = ''

        # try README.md then README.txt, if both fail, raise FileNotFoundError
        for filename in ['README.md', 'README.txt']:
            filepath = self.directory / '..' / filename
            if filepath.exists() and filepath.is_file():
                with open(filepath, 'r') as f:
                    result = f.read()
                break
        else:
            raise FileNotFoundError(f'No README file found in {self.directory}.')
        
        if print_:
            print(result)
        return result



class PilotStudyFileReader(FileReader):
    _class_initialized = False
    DATES: list[str] = [
        '20250707',
    ]
    FRAME_DIRECTORIES: list[Path]
    RUNLOG: pd.DataFrame

    def __init__(self, participant_id: int, environment: str, experiment_id: int, offset_setup: bool = False, *args, **kwargs) -> None:
        super().__init__(offset_setup=offset_setup, *args, **kwargs)
        self.participant_id = participant_id
        self.environment = environment
        self.experiment_id = experiment_id
    
    @classmethod
    def from_index(cls, participant_id: int, environment: str, experiment_id: int, offset_setup: bool = False, not_found_warning=True) -> PilotStudyFileReader | None:
        if not cls._class_initialized:
            cls.init_class(offset_setup=offset_setup)
        
        query = cls.RUNLOG.query(
            f'human_subject == {participant_id} and '
            f'environment == "{environment}" and '
            f'experiment_number == {experiment_id} and '
            f'experiment_number > 0 and '
            f'"DISCARD" not in remark'
        )
        if len(query) != 1:
            if not_found_warning:
                warnings.warn(
                    f'No unique entry found for participant {participant_id} '
                    f'and experiment {experiment_id}. Returning None.',
                )
            return None
        info = query.iloc[0].to_dict()
        directory = [
            path for path in cls.FRAME_DIRECTORIES
            if path.name == info['directory_name']
        ]
        if len(directory) != 1:
            if not_found_warning:
                warnings.warn(
                    f'No unique directory found for participant {participant_id} '
                    f'and experiment {experiment_id}. Returning None.',
                    UserWarning
                )
            return None
        return cls(
            participant_id=participant_id,
            environment=environment,
            experiment_id=experiment_id,
            offset_setup=offset_setup,
            directory=directory[0],
        )

    @classmethod
    @functools.lru_cache(maxsize=1)
    def init_class(cls, offset_setup: bool = False) -> None:
        if cls._class_initialized:
            return

        cls.FRAME_DIRECTORIES = sorted([
            Path(path)
            for date in cls.DATES
            for path in (cls.FIELD_TEST_DIRECTORY / date).glob('rosbag*')
        ])
        
        # Initialize the runlog DataFrame
        runlog = None
        runlog_files = sorted(set(
            path.parent / 'runlog.csv'
            for path in cls.FRAME_DIRECTORIES
        ))
        for filepath in runlog_files:
            if runlog is None:
                runlog = pd.read_csv(filepath)
            else:
                runlog = pd.concat([runlog, pd.read_csv(filepath)], ignore_index=True)
        if runlog is None:
            raise FileNotFoundError('No runlog files found in the field test directories.')
        cls.RUNLOG = runlog

        cls._class_initialized = True
    
    @classmethod
    @functools.lru_cache(maxsize=1)
    def list_all_valid_indices(cls) -> list[dict[str, int | str]]:
        if not cls._class_initialized:
            cls.init_class()

        result = []
        for participant_id, environment, experiment_id in itertools.product(
            cls.RUNLOG['human_subject'].unique(),
            cls.RUNLOG['environment'].unique(),
            cls.RUNLOG['experiment_number'].unique(),
        ):
            query = cls.RUNLOG.query(
                f'human_subject == {participant_id} and '
                f'environment == "{environment}" and '
                f'experiment_number == {experiment_id} and '
                f'experiment_number > 0 and '
                f'"DISCARD" not in remark'
            )
            if len(query) == 1:
                result.append({
                    'participant_id': participant_id,
                    'environment': environment,
                    'experiment_id': experiment_id,
                })
        # sort by participant_id then experiment_id
        result.sort(key=lambda x: (x['participant_id'], x['environment'], x['experiment_id']))
        return result



class VolleyBallCourtFileReader(FileReader):
    _class_initialized = False
    DATES: list[str] = [
        '20250806',
        '20250807',
        '20250812',
        '20250814',
        '20250815',
    ]
    FRAME_DIRECTORIES: list[Path]
    RUNLOG: pd.DataFrame

    def __init__(self, participant_id: int, condition_id: int, offset_setup: bool = False, *args, **kwargs) -> None:
        super().__init__(offset_setup=offset_setup, *args, **kwargs)
        self.participant_id = participant_id
        self.condition_id = condition_id

    @classmethod
    def from_index(cls, participant_id: int, condition_id: int, offset_setup: bool = False, not_found_warning=True) -> VolleyBallCourtFileReader | None:
        if not cls._class_initialized:
            cls.init_class(offset_setup=offset_setup)

        condition = cls.list_all_conditions().get(condition_id, None)
        if condition is None:
            if not_found_warning:
                warnings.warn(
                    f'Condition ID {condition_id} is not valid. Returning None.',
                    UserWarning
                )
            return None
        query = cls.RUNLOG.query(
            f'participant_id == {participant_id} and '
            f'condition == "{condition}" and '
            f'"DISCARD" not in remark'
        )
        if len(query) != 1:
            if not_found_warning:
                warnings.warn(
                    f'No unique entry found for participant {participant_id} '
                    f'and condition "{condition}". Returning None.',
                )
            return None
        info = query.iloc[0].to_dict()
        directory = [
            path for path in cls.FRAME_DIRECTORIES
            if path.name == info['directory_name']
        ]
        if len(directory) != 1:
            if not_found_warning:
                warnings.warn(
                    f'No unique directory found for participant {participant_id} '
                    f'and condition "{condition}". Returning None.',
                    UserWarning
                )
            return None
        return cls(
            participant_id=participant_id,
            condition_id=condition_id,
            offset_setup=offset_setup,
            directory=directory[0],
        )

    @classmethod
    def init_class(cls, offset_setup: bool = False) -> None:
        if cls._class_initialized:
            return

        cls.FRAME_DIRECTORIES = [
            Path(path)
            for date in cls.DATES
            for path in (cls.FIELD_TEST_DIRECTORY / date).glob('*_*_*')
        ]
    
        # Initialize the runlog DataFrame
        runlog = None
        runlog_files = sorted(set(
            path.parent / 'runlog.csv'
            for path in cls.FRAME_DIRECTORIES
        ))
        for filepath in runlog_files:
            if runlog is None:
                runlog = pd.read_csv(filepath)
            else:
                runlog = pd.concat([runlog, pd.read_csv(filepath)], ignore_index=True)
        if runlog is None:
            raise FileNotFoundError('No runlog files found in the field test directories.')
        runlog['condition'] = runlog['condition'].fillna('TO BE FILLED')
        cls.RUNLOG = runlog
        cls._class_initialized = True
    
    @classmethod
    @functools.lru_cache(maxsize=1)
    def list_all_conditions(cls) -> dict[int, str]:
        if not cls._class_initialized:
            cls.init_class()

        result: dict[int, str] = {
            1: 'Stationary baseline with outdoor background',
            2: 'Stationary baseline with white cardboard as background',
            3: 'Wearing glasses',
            4: 'Wearing face mask',
            5: 'Partial shadow',
            6: 'Direct sunlight',
            7: 'Slow Head Movement - 12s',
            8: 'Medium Head Movement - 8s',
            9: 'Fast Head Movement - 4s',
            10: '2m distance from camera',
            11: '3m distance from camera',
        }
        conditions = cls.RUNLOG['condition'].unique().tolist()

        # update for any new conditions found in the runlog files
        idx = max(result) + 1
        for condition in conditions:
            if condition in result.values():
                continue

            result[idx] = condition
            idx += 1
            warnings.warn(
                f'Unexpected condition found: {condition}. '
                'Please update the list_all_conditions method.',
                UserWarning
            )
        
        return result

    @classmethod
    @functools.lru_cache(maxsize=1)
    def list_all_valid_indices(cls) -> list[dict[str, int]]:
        if not cls._class_initialized:
            cls.init_class()

        result = []
        for participant_id in cls.RUNLOG['participant_id'].unique():
            for condition_id, condition in cls.list_all_conditions().items():
                query = cls.RUNLOG.query(
                    f'participant_id == {participant_id} and '
                    f'condition == "{condition}" and '
                    f'"DISCARD" not in remark'
                )
                if len(query) == 1:
                    result.append({
                        'participant_id': participant_id,
                        'condition_id': condition_id,
                    })

        # sort by participant_id then condition_id
        result.sort(key=lambda x: (x['participant_id'], x['condition_id']))
        return result