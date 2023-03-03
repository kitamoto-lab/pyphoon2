import os
from datetime import datetime
from enum import Enum
from collections import OrderedDict

from torch.utils.data import Dataset
from DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonImage import DigitalTyphoonImage


class SPLIT_UNIT(Enum):
    SEQUENCE = 'sequence'
    YEAR = 'year'
    FRAME = 'frame'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class LOAD_DATA(Enum):
    NO_DATA = False
    ONLY_TRACK = 'track'
    ALL_DATA = 'all_data'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def _verbose_print(string, verbose):
    if verbose:
        print(string)


class DigitalTyphoonDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 track_dir: str,
                 split_unit='sequence',  # can be [sequence, year, frame]
                 only_sequences_with_images=False,
                 load_data_into_memory=False,
                 verbose=False) -> None:

        if not SPLIT_UNIT.has_value(split_unit):
            raise ValueError(f'Split unit must one of the following\n'
                             f'    {[item.value for item in SPLIT_UNIT]}.\n'
                             f'    Input: {split_unit}')
        self.split_unit = split_unit

        if not LOAD_DATA.has_value(load_data_into_memory):
            raise ValueError(f'Load data option must one of the following\n'
                             f'    {[item.value for item in LOAD_DATA]}.\n'
                             f'    Input: {load_data_into_memory}')
        self.load_data_into_memory = load_data_into_memory

        self.image_dir = image_dir
        self.track_dir = track_dir
        self.only_sequences_with_images = only_sequences_with_images
        self.verbose = verbose

        self.sequences = OrderedDict()
        self.track_sequences = OrderedDict()
        self.track_sequences_without_images = []

        self.number_of_image_seq_start_years = None
        self.number_of_images = None

        self.number_of_start_years_with_track = None
        self.number_of_sequences_with_track = None

        _verbose_print(f'Initializing images from: {image_dir}', self.verbose)
        self._populate_images_into_sequences(self.image_dir)

        _verbose_print(f'Initializing track data from: {track_dir}', self.verbose)
        self._populate_track_data_into_sequences(self.track_dir)

    def __len__(self):
        if self.only_sequences_with_images:
            return self._len_of_sequences_with_images()
        else:
            if SPLIT_UNIT.SEQUENCE == self.split_unit:
                return len(self.track_sequences)
            elif SPLIT_UNIT.YEAR == self.split_unit:
                return self.number_of_start_years_with_track or \
                    len(set([sequence.get_start_year() for sequence in self.track_sequences.values()]))
            elif SPLIT_UNIT.FRAME == self.split_unit:
                print('splitting by frame')
        print('len')

    def __getitem__(self, idx):
        print('getitem')

    def _len_of_sequences_with_images(self):
        if SPLIT_UNIT.SEQUENCE == self.split_unit:
            return len(self.sequences)
        elif SPLIT_UNIT.YEAR == self.split_unit:
            if self.number_of_image_seq_start_years:  # value already cached
                return self.number_of_image_seq_start_years
            return len(set([sequence.get_start_year() for sequence in self.sequences.values()]))
        elif SPLIT_UNIT.FRAME == self.split_unit:
            if self.number_of_images:  # value already cached
                return self.number_of_images
            return sum([sequence.get_number_of_images_in_sequence() for sequence in self.sequences.values()])

    def _populate_images_into_sequences(self, image_dir):
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for file in files:
                if self.is_image_file(file):
                    file_sequence, file_date, file_satellite = self.parse_image_filename(file)
                    typhoon_image = DigitalTyphoonImage(file, file_date, file_sequence, file_satellite)
                    if file_sequence not in self.sequences:
                        self.sequences[file_sequence] = DigitalTyphoonSequence(file_sequence, file_date.year, file_satellite)
                    self.sequences[file_sequence].append_image_to_sequence(typhoon_image)

    def _populate_track_data_into_sequences(self, track_dir: str) -> None:
        for root, dirs, files in os.walk(self.track_dir, topdown=True):
            for file in files:
                file_sequence = self.get_sequence_from_track_filename(file)
                if file_sequence in self.sequences:
                    if self.load_data_into_memory in {LOAD_DATA.ONLY_TRACK, LOAD_DATA.ALL_DATA}:
                        self._read_in_track_file(file_sequence, file)
                    self.sequences[file_sequence].set_track_data_path(file)

    def _read_in_track_file_to_sequence(self, sequence: str, file: str, delimiter='\t') -> DigitalTyphoonSequence:
        with open(file, 'r') as opened_file:
            lines = opened_file.read().splitlines()
            for line in lines:
                # year | month | day | hour | grade | lat | long | pressure | max
                # wind | max gust | storm wind direction | storm radius major
                # | storm radius minor | gale wind direction | gale radius major
                # | gale radius minor | indicator landfall | moving speed
                # | moving direction | interpolated flag
                cols = line.split(delimiter)
                self.sequences[sequence].append_track_frame(cols[0], cols[1], cols[2], cols[3],
                                                            cols[4], cols[5], cols[6], cols[7],
                                                            cols[8], cols[9], cols[10], cols[11],
                                                            cols[12], cols[13], cols[14], cols[15],
                                                            cols[16], cols[17], cols[18], cols[19], )
            return self.sequences[sequence]

    @staticmethod
    def parse_image_filename(filename: str, separator='-') -> (str, datetime, str):
        date, sequence_num, satellite, _ = filename.split(separator)
        date_year = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_hour = int(date[8:10])
        sequence_datetime = datetime(year=date_year, month=date_month,
                                     day=date_day, hour=date_hour)
        return sequence_num, sequence_datetime, satellite

    @staticmethod
    def get_sequence_from_track_filename(filename: str) -> str:
        sequence_num = filename.removesuffix(".tsv")
        return sequence_num

    @staticmethod
    def is_image_file(filename: str) -> bool:
        return filename.endswith(".h5")
