import os
import json
from datetime import datetime
from enum import Enum
from collections import OrderedDict
from typing import List, Tuple
import numpy as np


from torch.utils.data import Dataset
from DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonUtils import parse_image_filename, get_seq_str_from_track_filename, is_image_file


class SPLIT_UNIT(Enum):
    SEQUENCE = 'sequence_str'
    YEAR = 'year'
    FRAME = 'frame'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class LOAD_DATA(Enum):
    NO_DATA = False
    ONLY_TRACK = 'track'
    ONLY_IMG = 'images'
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
                 metadata_filepath: str,
                 split_dataset_by='sequence_str',  # can be [sequence_str, year, frame]
                 get_images_by_sequence=False,
                 only_sequences_with_images=False,
                 load_data_into_memory=False,
                 verbose=False) -> None:

        if not SPLIT_UNIT.has_value(split_dataset_by):
            raise ValueError(f'Split unit must one of the following\n'
                             f'    {[item.value for item in SPLIT_UNIT]}.\n'
                             f'    Input: {split_dataset_by}')
        self.split_dataset_by = split_dataset_by

        if not LOAD_DATA.has_value(load_data_into_memory):
            raise ValueError(f'Load data option must one of the following\n'
                             f'    {[item.value for item in LOAD_DATA]}.\n'
                             f'    Input: {load_data_into_memory}')
        self.load_data_into_memory = load_data_into_memory
        self.get_images_by_sequence = get_images_by_sequence

        self.image_dir = image_dir
        self.track_dir = track_dir
        self.metadata_filepath = metadata_filepath
        self.only_sequences_with_images = only_sequences_with_images
        self.verbose = verbose

        self.sequences: List[Tuple[DigitalTyphoonSequence, Tuple[int, int]]] = list()
        self._sequence_to_idx = {}

        self.number_of_sequences = None
        self.number_of_start_years = None
        self.number_of_frames = 0

        self.years_to_sequence_nums = OrderedDict()

        _verbose_print(f'Processing metadata file at: {metadata_filepath}', self.verbose)
        self.process_metadata_file(metadata_filepath)

        _verbose_print(f'Initializing image_arrays from: {image_dir}', self.verbose)
        self._populate_images_into_sequences(self.image_dir)

        _verbose_print(f'Initializing track data from: {track_dir}', self.verbose)
        self._populate_track_data_into_sequences(self.track_dir)

    def __len__(self) -> int:
        if self.get_images_by_sequence:
            return self.get_number_of_sequences()
        else:
            return self.number_of_frames

    def __getitem__(self, idx) -> np.ndarray:
        if self.get_images_by_sequence:
            seq_str = self._find_sequence_str_from_index(idx)
            seq = self._get_seq_from_seq_str(seq_str)
            return seq.return_all_images_in_sequence_as_np()
        else:
            return self._get_image_from_idx_as_numpy(idx)

    def process_metadata_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.number_of_sequences = len(data)

        prev_interval_end = -1
        for sequence_str, metadata in data.items():
            prev_interval_end = self._read_one_seq_from_metadata(sequence_str, metadata, prev_interval_end)

        return data

    def _read_one_seq_from_metadata(self, sequence_str, metadata_json, prev_interval_end):
        seq_start_date = datetime.strptime(metadata_json['start'], '%Y-%m-%d')
        frame_interval = (prev_interval_end + 1, prev_interval_end + metadata_json['frames'])  # frame ends inclusive
        self.sequences.append((DigitalTyphoonSequence(sequence_str, seq_start_date.year, metadata_json['frames']),
                               frame_interval))
        self._sequence_to_idx[sequence_str] = len(self.sequences) - 1
        if metadata_json['year'] not in self.years_to_sequence_nums:
            self.years_to_sequence_nums[metadata_json['year']] = []
        self.years_to_sequence_nums[metadata_json['year']].append(sequence_str)
        self.number_of_frames += metadata_json['frames']
        return frame_interval[1]

    def _get_seq_from_seq_str(self, sequence) -> DigitalTyphoonSequence:
        return self.sequences[self._sequence_to_idx[sequence]][0]

    def _get_seq_frame_interval_from_seq_str(self, sequence) -> Tuple[int, int]:
        return self.sequences[self._sequence_to_idx[sequence]][1]

    def _find_sequence_str_from_index(self, idx):
        start, end = 0, len(self.sequences)-1
        while start <= end:
            mid = (start + end) // 2
            interval = self.sequences[mid][1]
            if interval[0] <= idx <= interval[1]:
                return self.sequences[mid][0].sequence_str
            elif idx < interval[0]:
                end = mid
            else:
                start = mid + 1
        return self.sequences[mid][0].sequence_str

    def _get_image_from_idx_as_numpy(self, idx) -> np.ndarray:
        sequence_str = self._find_sequence_str_from_index(idx)
        frame_interval = self._get_seq_frame_interval_from_seq_str(sequence_str)
        sequence = self._get_seq_from_seq_str(sequence_str)
        return sequence.get_image_at_idx_as_numpy(sequence.total_idx_to_sequence_idx(idx, frame_interval))

    def _populate_images_into_sequences(self, image_dir):
        load_into_mem = self.load_data_into_memory in {LOAD_DATA.ONLY_IMG, LOAD_DATA.ALL_DATA}
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for dir_name in sorted(dirs):  # Read sequences in chronological order, not necessary but convenient
                sequence_obj = self._get_seq_from_seq_str(dir_name)
                sequence_obj.process_seq_img_dir_into_sequence(root+dir_name, load_into_mem)

        for sequence, interval in self.sequences:
            if not sequence.num_images_match_num_frames():
                raise ValueError(f'Sequence {sequence.sequence_str} has only {sequence.num_image_paths_in_seq()} when '
                                 f'it should have {sequence.num_frames}.')

    def _populate_track_data_into_sequences(self, track_dir: str) -> None:
        for root, dirs, files in os.walk(self.track_dir, topdown=True):
            for file in files:
                file_sequence = get_seq_str_from_track_filename(file)
                if self.sequence_exists(file_sequence):
                    self._get_seq_from_seq_str(file_sequence).set_track_path(root + file)
                    if self.load_data_into_memory in {LOAD_DATA.ONLY_TRACK, LOAD_DATA.ALL_DATA}:
                        self._read_in_track_file_to_sequence(file_sequence, root + file)

    def _read_in_track_file_to_sequence(self, seq_str: str, file: str, csv_delimiter='\t') -> DigitalTyphoonSequence:
        sequence = self._get_seq_from_seq_str(seq_str)
        sequence.add_track_data(file, csv_delimiter)
        return sequence

    def get_number_of_sequences(self):
        return len(self.sequences)

    def _get_list_of_sequence_objs(self) -> List[DigitalTyphoonSequence]:
        return [sequence for sequence, frame_interval in self.sequences]

    def sequence_exists(self, seq_str: str) -> bool:
        return seq_str in self._sequence_to_idx

    def _delete_all_sequences(self):
        self.sequences = []
        self._sequence_to_idx = {}

        self.number_of_sequences = None
        self.number_of_start_years = None
        self.number_of_frames = 0
