import os
from datetime import datetime
from enum import Enum
from typing import Tuple


class SPLIT_UNIT(Enum):
    """
    Enum denoting which unit to treat as atomic when splitting the dataset
    """
    SEQUENCE = 'sequence'
    SEASON = 'season'
    IMAGE = 'image'

    @classmethod
    def has_value(cls, value):
        """
        Returns true if value is present in the enum

        :param value: str, the value to check for
        :return: bool
        """
        return value in cls._value2member_map_


class LOAD_DATA(Enum):
    """
    Enum denoting what level of data should be stored in memory
    """
    NO_DATA = False
    ONLY_TRACK = 'track'
    ONLY_IMG = 'images'
    ALL_DATA = 'all_data'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class TRACK_COLS(Enum):
    """
    Enum containing indices in a track csv col to find the respective data
    """
    YEAR = 0
    MONTH = 1
    DAY = 2
    HOUR = 3
    GRADE = 4
    LAT = 5
    LNG = 6
    PRESSURE = 7
    WIND = 8
    DIR50 = 9
    LONG50 = 10
    SHORT50 = 11
    DIR30 = 12
    LONG30 = 13
    SHORT30 = 14
    LANDFALL = 15
    INTERPOLATED = 16
    FILENAME = 17
    MASK_1 = 18
    MASK_1_PERCENT = 19

    @classmethod
    def str_to_value(cls, name):
        name_map = {
            'year': TRACK_COLS.YEAR.value,
            'month': TRACK_COLS.MONTH.value,
            'day': TRACK_COLS.DAY.value,
            'hour': TRACK_COLS.HOUR.value,
            'grade': TRACK_COLS.GRADE.value,
            'lat': TRACK_COLS.LAT.value,
            'lng': TRACK_COLS.LNG.value,
            'pressure': TRACK_COLS.PRESSURE.value,
            'wind': TRACK_COLS.WIND.value,
            'dir50': TRACK_COLS.DIR50.value,
            'long50': TRACK_COLS.LONG50.value,
            'short50': TRACK_COLS.SHORT50.value,
            'dir30': TRACK_COLS.DIR30.value,
            'long30': TRACK_COLS.LONG30.value,
            'short30': TRACK_COLS.SHORT30.value,
            'landfall': TRACK_COLS.LANDFALL.value,
            'interpolated': TRACK_COLS.INTERPOLATED.value,
            'filename': TRACK_COLS.FILENAME.value,
            'mask_1': TRACK_COLS.MASK_1.value,
            'mask_1_percent': TRACK_COLS.MASK_1_PERCENT.value,
        }
        if name in name_map:
            return name_map[name]
        else:
            raise KeyError(f"{name} is not a valid column name.")

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def _verbose_print(string: str, verbose: bool):
    """
    Prints the string if verbose is true

    :param string: str
    :param verbose: bool
    :return: None
    """
    if verbose:
        print(string)


def parse_image_filename(filename: str, separator='-') -> Tuple[str, datetime, str]:
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image

    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    try:
        date, sequence_num, satellite, _ = filename.split(separator)
        season = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_hour = int(date[8:10])
        sequence_datetime = datetime(year=season, month=date_month,
                                     day=date_day, hour=date_hour)
        return sequence_num, sequence_datetime, satellite
    except ValueError:
        raise ValueError(
            f"Filename {filename} does not match the expected format.")


def parse_common_image_filename(filename: str, separator='-') -> Tuple[str, datetime, str]:
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image

    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    try:
        date, sequence_num, satellite = filename.split(separator)
        season = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_hour = int(date[8:10])
        sequence_datetime = datetime(year=season, month=date_month,
                                     day=date_day, hour=date_hour)
        return sequence_num, sequence_datetime, satellite
    except ValueError:
        raise ValueError(
            f"Filename {filename} does not match the expected format.")


def get_seq_str_from_track_filename(filename: str) -> str:
    """
    Given a track filename, returns the sequence ID it belongs to.

    :param filename: str, the filename (e.g., "sequence1.csv")
    :return: str, the sequence ID string (e.g., "sequence1")
    :raises ValueError: If the filename does not end with '.csv'
    """
    # Split the filename into root and extension
    sequence_num, ext = os.path.splitext(filename)

    # Validate the extension
    if ext.lower() != '.csv':
        raise ValueError(
            f"Unexpected file extension: '{ext}'. Expected a '.csv' file.")

    return sequence_num


def is_image_file(filename: str) -> bool:
    """
    Given a DigitalTyphoon file, returns if it is an h5 image.

    :param filename: str, the filename
    :return: bool, True if it is an h5 image, False otherwise
    """
    return filename.endswith(".h5")
