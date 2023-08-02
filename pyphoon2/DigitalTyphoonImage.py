import os
import h5py
import numpy as np
from typing import List
import pandas as pd
from datetime import datetime

from pyphoon2.DigitalTyphoonUtils import TRACK_COLS


class DigitalTyphoonImage:
    def __init__(self, image_filepath: str, track_entry: np.ndarray, sequence_id=None, load_imgs_into_mem=False,
                 transform_func=None, spectrum='Infrared'):
        """
        Class for one image with metadata for the DigitalTyphoonDataset

        Does NOT check for file existence until accessing the image.

        :param image_filepath: str, path to image file
        :param track_entry: np.ndarray, 1d numpy array for the track csv entry corresponding to the image
        :param load_imgs_into_mem: bool, flag indicating whether images should be loaded into memory
        :param spectrum: str, default spectrum to read the image in
        param transform_func: this function will be called on the image array when the array is accessed (or read into memory).
                               It should take and return a numpy image array

        """
        self.sequence_str = sequence_id

        self.load_imgs_into_mem = load_imgs_into_mem
        self.spectrum = spectrum
        self.transform_func = transform_func


        self.image_filepath = image_filepath
        self.image_array = None
        if image_filepath is not None and self.load_imgs_into_mem:
            self.set_image_data(image_filepath, load_imgs_into_mem=self.load_imgs_into_mem)

        self.track_data = track_entry
        if track_entry is not None:
            self.set_track_data(track_entry)

    def image(self, spectrum=None) -> np.ndarray:
        """
        Returns the image as a numpy array. If load_imgs_into_mem was set to true, it will cache the image

        :param spectrum: spectrum (channel) the image was taken in
        :return: np.ndarray, the image
        """
        open_spectrum = self.spectrum
        if spectrum is not None:
            open_spectrum = spectrum

        if self.image_array is not None:
            return self.image_array

        image = self._get_h5_image_as_numpy(spectrum=open_spectrum)

        if self.transform_func is not None:
            image = self.transform_func(image)

        if self.load_imgs_into_mem:
            self.image_array = image
        return image

    def sequence_id(self) -> str:
        """
        Returns the sequence ID this image belongs to

        :return: str sequence str
        """
        return self.sequence_str

    def track_array(self) -> np.ndarray:
        """
        Returns the csv row for this image

        :return: nparray containing the track data
        """
        return self.track_data

    def value_from_string(self, label):
        """
        Returns the image's value given the label as a string. e.g. value_from_string('month') -> the month

        :return: the element
        """
        label_name = TRACK_COLS.str_to_value(label)
        return self.track_array()[label_name]

    def year(self) -> int:
        """
        Returns the year the image was taken

        :return: int, the year
        """
        return int(self.track_data[TRACK_COLS.YEAR.value])

    def month(self) -> int:
        """
        Returns the month the image was taken

        :return: int, the month (1-12)
        """
        return int(self.track_data[TRACK_COLS.MONTH.value])

    def day(self) -> int:
        """
        Returns the day the image was taken (number 1-31)

        :return: int the day
        """
        return int(self.track_data[TRACK_COLS.DAY.value])

    def hour(self) -> int:
        """
        Returns the hour the image was taken

        :return: int, the hour
        """
        return int(self.track_data[TRACK_COLS.HOUR.value])

    def datetime(self) -> datetime:
        """
        Returns a datetime object of when the image was taken

        :return: datetime
        """
        return datetime(self.year(), self.month(), self.day(), self.hour())

    def grade(self) -> int:
        """
        Returns the grade of the typhoon in the image

        :return: int, the grade
        """
        return int(self.track_data[TRACK_COLS.GRADE.value])

    def lat(self) -> float:
        """
        Returns the latitude of the image

        :return: float
        """
        return float(self.track_data[TRACK_COLS.LAT.value])

    def long(self) -> float:
        """
        Returns the longitude of the image

        :return: float
        """
        return float(self.track_data[TRACK_COLS.LNG.value])

    def pressure(self) -> float:
        """
        Returns the pressure in # TODO: units? probably hg

        :return: float
        """
        return float(self.track_data[TRACK_COLS.PRESSURE.value])

    def wind(self) -> float:
        """
        Returns the wind speed in # TODO: units?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.WIND.value])

    def dir50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.DIR50.value])

    def long50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.LONG50.value])

    def short50(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.SHORT50.value])

    def dir30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.DIR30.value])

    def long30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.LONG30.value])

    def short30(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.SHORT30.value])

    def landfall(self) -> float:
        """
        # TODO: what is this?

        :return: float
        """
        return float(self.track_data[TRACK_COLS.LANDFALL.value])

    def interpolated(self) -> bool:
        """
        Returns whether this entry is interpolated or not

        :return: bool
        """
        return bool(self.track_data[TRACK_COLS.INTERPOLATED.value])

    def filepath(self) -> str:
        """
        Returns the filepath to the image

        :return: str
        """
        return str(self.image_filepath)

    def mask_1(self) -> float:
        """
        Returns number of pixels in the image that are corrupted

        :return: float the number of pixels
        """
        return float(self.track_data[TRACK_COLS.MASK_1.value])

    def mask_1_percent(self) -> float:
        """
        Returns percentage of pixels in the image that are corrupted

        :return: float the percentage of pixels
        """
        return float(self.track_data[TRACK_COLS.MASK_1_PERCENT.value])

    def set_track_data(self, track_entry: np.ndarray) -> None:
        """
        Sets the track entry

        :param track_entry: numpy array representing one entry of the track csv
        :return: None
        """
        # if len(track_entry) != len(TRACK_COLS):
        #     raise ValueError(f'Number of columns in the track entry ({len(track_entry)}) is not equal '
        #                      f'to expected amount ({len(TRACK_COLS)})')
        self.track_data = track_entry

    def set_image_data(self, image_filepath: str, load_imgs_into_mem=False, spectrum=None) -> None:
        """
        Sets the image file data

        :param load_imgs_into_mem: bool, whether to load images into memory
        :param spectrum: str, spectrum to open h5 images with
        :param image_filepath: string to image
        :return: None
        """
        self.load_imgs_into_mem = load_imgs_into_mem
        if spectrum is None:
            spectrum = self.spectrum

        self.image_filepath = image_filepath
        if self.load_imgs_into_mem:
            self.image(spectrum=spectrum)  # Load the image on instantiation if load_imgs_into_mem is set to True

    def _get_h5_image_as_numpy(self, spectrum=None) -> np.ndarray:
        """
        Given an h5 image filepath, open and return the image as a numpy array
        
        :param spectrum: str, the spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        if spectrum is None:
            spectrum = self.spectrum

        with h5py.File(self.image_filepath, 'r') as h5f:
            image = np.array(h5f.get(spectrum))
        return image
