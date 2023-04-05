from datetime import datetime
from enum import Enum


class SPLIT_UNIT(Enum):
    """
    Enum denoting which unit to treat as atomic when splitting the dataset
    """
    SEQUENCE = 'sequence'
    YEAR = 'year'
    FRAME = 'frame'

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


def _verbose_print(string: str, verbose: bool):
    """
    Prints the string if verbose is true
    :param string: str
    :param verbose: bool
    :return: None
    """
    if verbose:
        print(string)


def parse_image_filename(filename: str, separator='-') -> (str, datetime, str):
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image
    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    date, sequence_num, satellite, _ = filename.split(separator)
    date_year = int(date[:4])
    date_month = int(date[4:6])
    date_day = int(date[6:8])
    date_hour = int(date[8:10])
    sequence_datetime = datetime(year=date_year, month=date_month,
                                 day=date_day, hour=date_hour)
    return sequence_num, sequence_datetime, satellite


def get_seq_str_from_track_filename(filename: str) -> str:
    """
    Given a track filename, returns the sequence ID it belongs to
    :param filename: str, the filename
    :return: str, the sequence ID string
    """
    sequence_num = filename.removesuffix(".csv")
    return sequence_num


def is_image_file(filename: str) -> bool:
    """
    Given a DigitalTyphoon file, returns if it is an h5 image.
    :param filename: str, the filename
    :return: bool, True if it is an h5 image, False otherwise
    """
    return filename.endswith(".h5")
