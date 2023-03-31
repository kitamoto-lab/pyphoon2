import os
import h5py
import numpy as np
from typing import List
import pandas as pd

from DigitalTyphoonUtils import parse_image_filename, is_image_file


class DigitalTyphoonSequence:

    def __init__(self, seq_str: str, start_year: int, num_frames: int):
        """
        Class representing one typhoon sequence from the DigitalTyphoon dataset
        :param seq_str: str, sequence ID as a string
        :param start_year: int, the year in which the typhoon starts in
        :param num_frames: int, number of frames/images in the sequence
        """
        self.sequence_str = seq_str  # sequence ID string
        self.year = start_year
        self.num_frames = num_frames
        self.track_data = np.array([])
        self.img_root = None  # root path to directory containing image files
        self.track_path = None # path to track file data

        # Ordered list containing paths to image_arrays
        self.image_filenames: List[str] = list()

        # Ordered list containing sequence_str image_arrays
        self.image_arrays: List[np.ndarray] = list()

        # Track data
        self.track_data = None  # np array containing track data

    def get_sequence_str(self) -> str:
        """
        Returns the sequence ID as a string
        :return: string sequence ID
        """
        return self.sequence_str

    def process_seq_img_dir_into_sequence(self, directory_path: str, load_imgs_into_mem=False, spectrum='infrared') -> None:
        """
        Given a path to a directory containing images of a typhoon sequence, process the images into the current
        sequence object. If 'load_imgs_into_mem' is set to True, the images will be read as numpy arrays and stored in
        memory. Spectrum refers to what light spectrum the image lies in.

        :param directory_path: Path to the typhoon sequence directory
        :param load_imgs_into_mem: Bool representing if images should be loaded into memory
        :param spectrum: string representing what spectrum the image lies in
        :return: None
        """
        self.set_images_root_path(directory_path)
        for root, dirs, files in os.walk(directory_path, topdown=True):
            filepaths = [(file,) + parse_image_filename(file) for file in files if is_image_file(file)]
            filepaths.sort(key=lambda x: x[2])  # sort by datetime
            for filepath, file_sequence, file_date, file_satellite in filepaths:
                self.append_image_path_to_seq(filepath)
                if load_imgs_into_mem:
                    self.append_image_to_sequence(self._get_h5_image_as_numpy(filepath, spectrum))

    def append_image_path_to_seq(self, image_path: str) -> str:
        """
        Given a path to an image, appends it to the end of the image list in the sequence.
        :param image_path: str, path to image file
        :return: str, path to the image file
        """
        self.image_filenames.append(image_path)
        return self.image_filenames[-1]

    def append_image_to_sequence(self, image: np.ndarray) -> np.ndarray:
        """
        Given an image array, appends it to the end of the image array list in the sequence.
        :param image: np.ndarray, image array
        :return: np.ndarray, the image array
        """
        self.image_arrays.append(image)
        return self.image_arrays[-1]

    def get_start_year(self) -> int:
        """
        Get the start year of the sequence
        :return: int, the start year
        """
        return self.year

    def get_num_frames(self) -> int:
        """
        Get the number of images/frames in the sequence
        :return: int, the number of frames
        """
        return self.num_frames

    def num_image_paths_in_seq(self) -> int:
        """
        Get the number of image filepaths currently stored in the sequence
        :return: int, the number of image filepaths
        """
        return len(self.image_filenames)

    def has_images(self) -> bool:
        """
        Returns true if the sequence currently holds images (or image filepaths). False otherwise.
        :return: bool
        """
        return len(self.image_arrays) != 0

    def add_track_data(self, filename: str, csv_delimiter=',') -> None:
        """
        Reads and adds the track data to the sequence.
        :param filename: str, path to the track data
        :param csv_delimiter: char, delimiter to use to read the csv
        :return: None
        """
        df = pd.read_csv(filename, delimiter=csv_delimiter)
        self.track_data = df.to_numpy()

    def set_track_path(self, track_path: str) -> None:
        """
        Sets the path to the track data file
        :param track_path: str, filepath to the track data
        :return: None
        """
        if not self.track_path:
            self.track_path = track_path

    def get_track_path(self) -> None:
        """
        Gets the path to the track data file
        :return: str, the path to the track data file
        """
        return self.track_path

    def get_track_data(self) -> np.ndarray:
        """
        Returns the track csv data as a numpy array, with each row corresponding to a row in the CSV.
        :return: np.ndarray
        """
        return self.track_data

    def get_image_at_idx_as_numpy(self, idx: int, spectrum='infrared') -> np.ndarray:
        """
        Gets the idx'th image in the sequence as a numpy array. Raises an exception if the idx is outside of the
        sequence's range.
        :param idx: int, idx to access
        :param spectrum: str, spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        if idx < 0 or idx >= len(self.image_filenames):
            raise ValueError(f'Requested idx {idx} is outside range of sequence images ({len(self.image_filenames)})')
        return self._get_h5_image_as_numpy(self.image_filenames[idx], spectrum)

    def _get_h5_image_as_numpy(self, filename: str, spectrum='infrared') -> np.ndarray:
        """
        Given an h5 image filepath, open and return the image as a numpy array
        :param filename: str, the filepath to the h5 image
        :param spectrum: str, the spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        with h5py.File(self.img_root + '/' + filename, 'r') as h5f:
            image = np.array(h5f.get(spectrum))
        return image

    def return_all_images_in_sequence_as_np(self, spectrum='infrared') -> np.ndarray:
        """
        Returns all the images in a sequence as a numpy array of shape (num_images, image_shape[0], image_shape[1])
        :param spectrum: str, spectrum of the image
        :return: np.ndarray of shape (num_image, image_shape[0], image_shape[1])
        """
        if len(self.image_arrays) > 0:
            return np.array(self.image_arrays)
        else:
            return np.array([self._get_h5_image_as_numpy(filename, spectrum) for filename in self.image_filenames])

    def num_images_match_num_frames(self) -> bool:
        """
        Returns True if the number of image filepaths stored matches the number of images stated when initializing
        the sequence object. False otherwise.
        :return: bool
        """
        return len(self.image_filenames) == self.num_frames

    def get_image_filenames(self) -> List[str]:
        """
        Returns a list of the filenames of the images (without the root path)
        :return: List[str], list of the filenames
        """
        return self.image_filenames

    def set_images_root_path(self, images_root_path: str) -> None:
        """
        Sets the root path of the images.
        :param images_root_path: str, the root path
        :return: None
        """
        if not self.img_root:
            self.img_root = images_root_path

    def get_images_root_path(self) -> None:
        """
        Gets the root path to the image directory
        :return: str, the root path
        """
        return self.img_root
