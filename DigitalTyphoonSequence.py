import os

import h5py
import numpy as np
from typing import List, Tuple

import pandas as pd

from DigitalTyphoonUtils import parse_image_filename, get_seq_str_from_track_filename, is_image_file


class DigitalTyphoonSequence:

    def __init__(self, sequence: str, year: int, num_frames: int):
        self.sequence_str = sequence
        self.year = year
        self.num_frames = num_frames
        self.track_data = np.array([])
        self.img_root = None

        self.track_path = None

        # Ordered list containing paths to image_arrays
        self.image_filenames: List[str] = list()

        # Ordered list containing sequence_str image_arrays
        self.image_arrays: List[np.ndarray] = list()

        # Track data
        self.track_data = None  # np array containing track data

    def get_sequence_str(self) -> str:
        return self.sequence_str

    def process_seq_img_dir_into_sequence(self, directory_path: str, load_imgs_into_mem=False, spectrum='infrared') -> None:
        self.set_images_root_path(directory_path)
        for root, dirs, files in os.walk(directory_path, topdown=True):
            filepaths = [(file,) + parse_image_filename(file) for file in files if is_image_file(file)]
            filepaths.sort(key=lambda x: x[2])  # sort by datetime
            for filepath, file_sequence, file_date, file_satellite in filepaths:
                self.append_image_path_to_seq(filepath)
                if load_imgs_into_mem:
                    self.append_image_to_sequence(self._get_h5_image_as_numpy(filepath, spectrum))

    def append_image_path_to_seq(self, image_path: str) -> str:
        self.image_filenames.append(image_path)
        return self.image_filenames[-1]

    def append_image_to_sequence(self, image: np.ndarray) -> np.ndarray:
        self.image_arrays.append(image)
        return self.image_arrays[-1]

    def get_start_year(self) -> int:
        return self.year

    def num_image_paths_in_seq(self):
        return len(self.image_filenames)

    def has_images(self) -> bool:
        return len(self.image_arrays) != 0

    def add_track_data(self, filename: str, csv_delimiter=',') -> None:
        df = pd.read_csv(filename)
        self.track_data = df.to_numpy()

    def get_image_at_idx_as_numpy(self, idx: int, spectrum='infrared') -> np.ndarray:
        if idx < 0 or idx >= len(self.image_filenames):
            raise ValueError(f'Requested idx {idx} is outside range of sequence images ({len(self.image_filenames)})')
        return self._get_h5_image_as_numpy(self.image_filenames[idx], spectrum)

    @staticmethod
    def total_idx_to_sequence_idx(total_idx: int, frame_interval: Tuple[int, int]) -> int:
        if not frame_interval[0] <= total_idx <= frame_interval[1]:
            raise ValueError(f'Total idx {total_idx} must be within frame interval for sequence {frame_interval}')
        return total_idx - frame_interval[0]

    def _get_h5_image_as_numpy(self, filename, spectrum='infrared'):
        with h5py.File(self.img_root + filename, 'r') as h5f:
            image = np.array(h5f.get(spectrum))
        return image

    def return_all_images_in_sequence_as_np(self, spectrum='infrared'):
        if len(self.image_arrays) > 0:
            return np.array(self.image_arrays)
        else:
            return np.array([self._get_h5_image_as_numpy(filename, spectrum) for filename in self.image_filenames])

    def num_images_match_num_frames(self):
        return len(self.image_filenames) == self.num_frames

    def get_image_filenames(self) -> List[str]:
        return self.image_filenames

    def set_images_root_path(self, images_root_path: str) -> None:
        if not self.img_root:
            self.img_root = images_root_path

    def set_track_path(self, track_path: str) -> None:
        if not self.track_path:
            self.track_path = track_path

    def get_track_path(self) -> None:
        return self.track_path

    def get_images_root_path(self) -> None:
        return self.img_root

    def get_track_data(self) -> np.ndarray:
        return self.track_data