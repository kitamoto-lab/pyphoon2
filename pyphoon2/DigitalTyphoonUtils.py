from datetime import datetime
from enum import Enum


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
        if name == 'year':
            return TRACK_COLS.YEAR.value
        elif name == 'month':
            return TRACK_COLS.MONTH.value
        elif name == 'day':
            return TRACK_COLS.DAY.value
        elif name == 'hour':
            return TRACK_COLS.HOUR.value
        elif name == 'grade':
            return TRACK_COLS.GRADE.value
        elif name == 'lat':
            return TRACK_COLS.LAT.value
        elif name == 'lng':
            return TRACK_COLS.LNG.value
        elif name == 'pressure':
            return TRACK_COLS.PRESSURE.value
        elif name == 'wind':
            return TRACK_COLS.WIND.value
        elif name == 'dir50':
            return TRACK_COLS.DIR50.value
        elif name == 'long50':
            return TRACK_COLS.LONG50.value
        elif name == 'short50':
            return TRACK_COLS.SHORT50.value
        elif name == 'dir30':
            return TRACK_COLS.DIR30.value
        elif name == 'long30':
            return TRACK_COLS.LONG30.value
        elif name == 'short30':
            return TRACK_COLS.SHORT30.value
        elif name == 'landfall':
            return TRACK_COLS.LANDFALL.value
        elif name == 'interpolated':
            return TRACK_COLS.INTERPOLATED.value
        elif name == 'filename':
            return TRACK_COLS.FILENAME.value
        elif name == 'mask_1':
            return TRACK_COLS.MASK_1.value
        elif name == 'mask_1_percent':
            return TRACK_COLS.MASK_1_PERCENT.value
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


def parse_image_filename(filename: str, separator='-') -> (str, datetime, str):
    """
    Takes the filename of a Digital Typhoon image and parses it to return the date it was taken, the sequence ID
    it belongs to, and the satellite that took the image

    :param filename: str, filename of the image
    :param separator: char, separator used in the filename
    :return: (str, datetime, str), Tuple containing the sequence ID, the datetime, and satellite string
    """
    date, sequence_num, satellite, _ = filename.split(separator)
    season = int(date[:4])
    date_month = int(date[4:6])
    date_day = int(date[6:8])
    date_hour = int(date[8:10])
    sequence_datetime = datetime(year=season, month=date_month,
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
