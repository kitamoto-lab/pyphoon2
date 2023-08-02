import os
import warnings
from datetime import datetime

import h5py
from pathlib import Path
import numpy as np
from typing import List, Dict
import pandas as pd
from collections import OrderedDict

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from pyphoon2.DigitalTyphoonUtils import parse_image_filename, is_image_file, TRACK_COLS


class DigitalTyphoonSequence:

    def __init__(self, seq_str: str, start_season: int, num_images: int, transform_func=None,
                 spectrum='Infrared', verbose=False):
        """
        Class representing one typhoon sequence from the DigitalTyphoon dataset

        :param seq_str: str, sequence ID as a string
        :param start_season: int, the season in which the typhoon starts in
        :param num_images: int, number of images in the sequence
        :param transform_func: this function will be called on each image before saving it/returning it.
                               It should take and return a np array
        """
        self.verbose = verbose

        self.sequence_str = seq_str  # sequence ID string
        self.season = start_season
        self.num_track_entries = 0
        self.num_original_images = num_images
        self.track_data = np.array([])
        self.img_root = None  # root path to directory containing image files
        self.track_path = None  # path to track file data
        self.transform_func = transform_func
        self.spectrum = spectrum

        # Ordered list containing image objects with metadata
        self.images: List[DigitalTyphoonImage] = list()

        # Dictionary mapping datetime to Image objects
        self.datetime_to_image: Dict[datetime, DigitalTyphoonImage] = {}

        # TODO: add synthetic image value into metadata and consistency check within dataset loader

    def get_sequence_str(self) -> str:
        """
        Returns the sequence ID as a string

        :return: string sequence ID
        """
        return self.sequence_str

    def process_seq_img_dir_into_sequence(self, directory_path: str,
                                          load_imgs_into_mem=False,
                                          ignore_list=None,
                                          spectrum=None,
                                          filter_func=lambda img: True) -> None:
        """
        Given a path to a directory containing images of a typhoon sequence, process the images into the current
        sequence object. If 'load_imgs_into_mem' is set to True, the images will be read as numpy arrays and stored in
        memory. Spectrum refers to what light spectrum the image lies in.

        :param directory_path: Path to the typhoon sequence directory
        :param load_imgs_into_mem: Bool representing if images should be loaded into memory
        :param ignore_list: list of image filenames to ignore
        :param spectrum: string representing what spectrum the image lies in
        :param filter_func: function that accepts an image and returns True or False if it should be included in the sequence
        :return: None
        """
        if ignore_list is None:
            ignore_list = set([])

        if spectrum is None:
            spectrum = self.spectrum

        self.set_images_root_path(directory_path)
        for root, dirs, files in os.walk(directory_path, topdown=True):
            filepaths = [(file,) + parse_image_filename(file) for file in files if is_image_file(file)]
            filepaths.sort(key=lambda x: x[2])  # sort by datetime
            for filepath, file_sequence, file_date, file_satellite in filepaths:
                if filepath not in ignore_list:
                    self.datetime_to_image[file_date].set_image_data(self.img_root / filepath,
                                                                     load_imgs_into_mem=load_imgs_into_mem,
                                                                     spectrum=spectrum)
                    if filter_func(self.datetime_to_image[file_date]):
                        self.images.append(self.datetime_to_image[file_date])

        if self.verbose:
            if not self.num_images_match_num_expected():
                warnings.warn(f'The number of images ({len(self.images)}) does not match the '
                              f'number of expected images ({self.num_original_images}) from metadata. If this is expected, ignore this warning.')

            if len(self.images) < self.num_track_entries:
                warnings.warn(f'Only {len(self.images)} of {self.num_track_entries} track entries have images.')

    def get_start_season(self) -> int:
        """
        Get the start season of the sequence

        :return: int, the start season
        """
        return self.season

    def get_num_images(self) -> int:
        """
        Gets the number of images in the sequence

        :return: int
        """
        return len(self.images)

    def get_num_original_images(self) -> int:
        """
        Get the number of images in the sequence

        :return: int, the number of images
        """
        return self.num_original_images

    def has_images(self) -> bool:
        """
        Returns true if the sequence currently holds images (or image filepaths). False otherwise.

        :return: bool
        """
        return len(self.images) != 0

    def process_track_data(self, track_filepath: str, csv_delimiter=',') -> None:
        """
        Takes in the track data for the sequence and processes it into the images for the sequence.

        :param track_filepath: str, path to track csv
        :param csv_delimiter: delimiter for the csv file
        :return: None
        """
        df = pd.read_csv(track_filepath, delimiter=csv_delimiter)
        data = df.to_numpy()
        for row in data:
            row_datetime = datetime(int(row[TRACK_COLS.YEAR.value]), int(row[TRACK_COLS.MONTH.value]),
                                    int(row[TRACK_COLS.DAY.value]), int(row[TRACK_COLS.HOUR.value]))
            self.datetime_to_image[row_datetime] = DigitalTyphoonImage(None, row, sequence_id=self.get_sequence_str(),
                                                                       transform_func=self.transform_func,
                                                                       spectrum=self.spectrum)
            self.num_track_entries += 1

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

    def get_track_path(self) -> str:
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

    def get_image_at_idx(self, idx:int, spectrum='Infrared') -> DigitalTyphoonImage:
        """
        Returns the idx'th DigitalTyphoonImage in the sequence. raises an exception if the idx is out of the
        the sequence's range

        :param idx: int, idx to access
        :param spectrum: str, spectrum of the image
        :return: DigitalTyphoonImage, the image object
        """
        if idx < 0 or idx >= len(self.images):
            raise ValueError(f'Requested idx {idx} is outside range of sequence images ({len(self.images)})')
        return self.images[idx]

    def get_image_at_idx_as_numpy(self, idx: int, spectrum=None) -> np.ndarray:
        """
        Gets the idx'th image in the sequence as a numpy array. Raises an exception if the idx is outside of the
        sequence's range.

        :param idx: int, idx to access
        :param spectrum: str, spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        if spectrum is None:
            spectrum = self.spectrum
        return self.get_image_at_idx(idx, spectrum=spectrum).image()

    def get_all_images_in_sequence(self) -> List[DigitalTyphoonImage]:
        """
        Returns all of the image objects (DigitalTyphoonImage) in the sequence in order.

        :return: List[DigitalTyphoonImage]
        """
        return self.images

    def return_all_images_in_sequence_as_np(self, spectrum=None) -> np.ndarray:
        """
        Returns all the images in a sequence as a numpy array of shape (num_images, image_shape[0], image_shape[1])

        :param spectrum: str, spectrum of the image
        :return: np.ndarray of shape (num_image, image_shape[0], image_shape[1])
        """
        if spectrum is None:
            spectrum = self.spectrum
        return np.array([image.image(spectrum=spectrum) for image in self.images])

    def num_images_match_num_expected(self) -> bool:
        """
        Returns True if the number of image filepaths stored matches the number of images stated when initializing
        the sequence object. False otherwise.

        :return: bool
        """
        return len(self.images) == self.num_original_images

    def get_image_filepaths(self) -> List[str]:
        """
        Returns a list of the filenames of the images (without the root path)

        :return: List[str], list of the filenames
        """
        return [image.filepath() for image in self.images]

    def set_images_root_path(self, images_root_path: str) -> None:
        """
        Sets the root path of the images.

        :param images_root_path: str, the root path
        :return: None
        """
        self.img_root = Path(images_root_path)

    def get_images_root_path(self) -> str:
        """
        Gets the root path to the image directory
        
        :return: str, the root path
        """
        return str(self.img_root)
