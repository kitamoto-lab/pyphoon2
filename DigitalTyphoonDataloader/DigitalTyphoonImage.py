import os
import h5py
import numpy as np
from typing import List
import pandas as pd
from datetime import datetime

from DigitalTyphoonDataloader.DigitalTyphoonUtils import TRACK_COLS


class DigitalTyphoonImage:
    def __init__(self, image_filepath: str, track_entry: np.ndarray, load_imgs_into_mem=False, spectrum='infrared'):
        """
        Class for one image with metadata for the DigitalTyphoonDataset

        :param image_filepath: str, path to image file
        :param track_entry: np.ndarray, 1d numpy array for the track csv entry corresponding to the image
        :param load_imgs_into_mem: bool, flag indicating whether images should be loaded into memory
        :param spectrum: str, spectrum to read the image in
        """
        self.load_imgs_into_mem = load_imgs_into_mem
        self.spectrum = spectrum

        self.image_filepath = image_filepath
        self.image_array = None

        if self.load_imgs_into_mem:
            self.image()  # Load the image on instantiation if load_imgs_into_mem is set to True

        self.track_data = track_entry

    def image(self, spectrum='infrared') -> np.ndarray:
        """
        Returns the image as a numpy array. If load_imgs_into_mem was set to true, it will cache the image
        :param spectrum: spectrum (channel) the image was taken in
        :return: np.ndarray, the image
        """
        if self.image_array is not None:
            return self.image_array

        image = self._get_h5_image_as_numpy(spectrum=spectrum)
        if self.load_imgs_into_mem:
            self.image_array = image
        return image

    def track_data(self) -> np.ndarray:
        """
        Returns the csv row for this image
        :return: nparray containing the track data
        """
        return self.track_data

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
        :return:
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

    def lat(self) -> float:
        """
        Returns the latitude of the image
        :return: float
        """
        return self.track_data[TRACK_COLS.LAT.value]

    def long(self) -> float:
        """
        Returns the longitude of the image
        :return: float
        """
        return self.track_data[TRACK_COLS.LNG.value]

    def pressure(self) -> float:
        """
        Returns the pressure in # TODO: units? probably hg
        :return: float
        """
        return self.track_data[TRACK_COLS.PRESSURE.value]

    def wind(self) -> float:
        """
        Returns the wind speed in # TODO: units?
        :return: float
        """
        return self.track_data[TRACK_COLS.WIND.value]

    def dir50(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.DIR50.value]

    def long50(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.LONG50.value]

    def short50(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.SHORT50.value]

    def dir30(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.DIR30.value]

    def long30(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.LONG30.value]

    def short30(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.SHORT30.value]

    def landfall(self) -> float:
        """
        # TODO: what is this?
        :return: float
        """
        return self.track_data[TRACK_COLS.LANDFALL.value]

    def interpolated(self) -> bool:
        """
        Returns whether this entry is interpolated or not
        :return: bool
        """
        return self.track_data[TRACK_COLS.INTERPOLATED.value]

    def filepath(self) -> str:
        """
        Returns the filepath to the image
        :return: str
        """
        return str(self.image_filepath)

    def _get_h5_image_as_numpy(self, spectrum='infrared') -> np.ndarray:
        """
        Given an h5 image filepath, open and return the image as a numpy array
        :param spectrum: str, the spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        with h5py.File(self.image_filepath, 'r') as h5f:
            image = np.array(h5f.get(spectrum))
        return image

    def set_track_data(self, track_entry: np.ndarray) -> None:
        """
        Sets the track entry
        :param track_entry: numpy array representing one entry of the track csv
        :return: None
        """
        if len(track_entry) != len(TRACK_COLS):
            raise ValueError(f'Number of columns in the track entry ({len(track_entry)}) is not equal '
                             f'to expected amount ({len(TRACK_COLS)})')
        self.track_data = track_entry
