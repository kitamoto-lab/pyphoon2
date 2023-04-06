import math
import os
import json
import warnings
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import List, Sequence, Union, Optional, Dict

from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset, random_split

from DigitalTyphoonDataloader import DigitalTyphoonImage
from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonDataloader.DigitalTyphoonUtils import _verbose_print, SPLIT_UNIT, LOAD_DATA, get_seq_str_from_track_filename


class DigitalTyphoonDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 track_dir: str,
                 metadata_filepath: str,
                 split_dataset_by='sequence',  # can be [sequence, year, frame]
                 get_images_by_sequence=False,
                 load_data_into_memory=False,
                 ignore_list=None,
                 verbose=False) -> None:
        """
        Dataloader for the DigitalTyphoon dataset.
        :param image_dir: Path to directory containing directories of typhoon sequences
        :param track_dir: Path to directory containing track data for typhoon sequences
        :param metadata_filepath: Path to the metadata JSON file
        :param split_dataset_by: What unit to treat as an atomic unit when randomly splitting the dataset. Options are
                                "sequence", "year", or "frame" (individual image)
        :param get_images_by_sequence: Boolean representing if an index should refer to an individual image or an entire
                                        sequence. If sequence, returned images are Lists of images.
        :param load_data_into_memory:  String representing if the images and track data should be entirely loaded into
                                        memory. Options are "track" (only track data), "images" (only images), or
                                        "all_data" (both track and images).
        :param ignore_list: a list of filenames (not path) to ignore and NOT add to the dataset
        :param verbose: Print verbose program information
        """

        if not SPLIT_UNIT.has_value(split_dataset_by):
            raise ValueError(f'Split unit must one of the following\n'
                             f'    {[item.value for item in SPLIT_UNIT]}.\n'
                             f'    Input: {split_dataset_by}')
        self.split_dataset_by = split_dataset_by

        if not LOAD_DATA.has_value(load_data_into_memory):
            raise ValueError(f'Load data option must one of the following\n'
                             f'    {[item.value for item in LOAD_DATA]}.\n'
                             f'    Input: {load_data_into_memory}')

        # String determining whether the image data should be fully loaded into memory
        self.load_data_into_memory = load_data_into_memory

        # Bool determining whether an atomic unit should be one image (False) frame or one typhoon (True).
        self.get_images_by_sequence = get_images_by_sequence

        # Directories containing image folders and track data
        self.image_dir = image_dir
        self.track_dir = track_dir

        # Path to the metadata file
        self.metadata_filepath = metadata_filepath
        self.verbose = verbose

        # Set of image filepaths to ignore
        self.ignore_list = set(ignore_list) if ignore_list else set([])

        # Structures holding the data objects
        self.sequences: List[DigitalTyphoonSequence] = list()  # List of seq_str objects
        self._sequence_str_to_seq_idx: Dict[str, int] = {}  # Sequence ID to idx in sequences array
        self._frame_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}  # Image idx to what seq_str it belongs to
        self._seq_str_to_first_total_idx: Dict[str, int] = {}  # Sequence string to the first total idx belonging to
                                                               #  that seq_str

        self.number_of_sequences = None
        self.number_of_original_frames = 0  # Number of images in the original dataset before augmentation and removal
        self.number_of_frames = 0  # number of images in the dataset, after augmentation and removal


        # Year to list of sequences that start in that year
        self.years_to_sequence_nums: OrderedDict[str, List[str]] = OrderedDict()

        # Process the data into the loader
        _verbose_print(f'Processing metadata file at: {metadata_filepath}', self.verbose)
        self.process_metadata_file(metadata_filepath)

        _verbose_print(f'Initializing image_arrays from: {image_dir}', self.verbose)
        self._populate_images_into_sequences(self.image_dir)

        _verbose_print(f'Initializing track data from: {track_dir}', self.verbose)
        self._populate_track_data_into_sequences(self.track_dir)

        _verbose_print(f'Indexing the dataset', verbose=self.verbose)
        self._assign_all_images_a_dataset_idx()

    def __len__(self) -> int:
        """
        Gives the length of the dataset. If "get_images_by_sequence" was set to True on initialization, number of
        sequences is returned. Otherwise, number of images is returned.
        :return: int
        """
        if self.get_images_by_sequence:
            return self.get_number_of_sequences()
        else:
            return self.number_of_frames

    def __getitem__(self, idx):
        """
        Gets an item at a particular dataset index.

        If "get_images_by_sequence" was set to True on initialization,
        the idx'th sequence is returned as a list of DigitalTyphoonImages.

        Otherwise, the DigitalTyphoonImage at total dataset index idx is given.

        :param idx: int, index of image or seq_str within total dataset
        :return: a List of DigitalTyphoonImages, or a single DigitalTyphoonImage
        """
        if self.get_images_by_sequence:
            seq_str = self._find_sequence_str_from_frame_index(idx)
            seq = self._get_seq_from_seq_str(seq_str)
            return seq.get_all_images_in_sequence()
        else:
            return self._get_image_from_idx(idx)

    def random_split(self, lengths: Sequence[Union[int, float]],
                     split_by=None,
                     generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        Given a list of proportions or items, returns a random split of the dataset with proportions as close to
        the requested without causing leakage between requested split_unit. If split is by image, built-in PyTorch
        function is used. If split is by year, all images from typhoons starting in the same year will be placed in
        the same bucket. If split is by seq_str, all images from the same typhoon will be together.

        Returns a list of Subsets of indices according to requested lengths. If split is anything other than frame,
        indices within their split unit are not randomized. (I.e. indices of a seq_str will be kept contiguous, not
        randomized order mixing with other sequences).

        For Subset doc see https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset.

        :param lengths: lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :param split_by: What to treat as an atomic unit (image, seq_str, year)
        :return: List[Subset[idx]]
        """
        if split_by is None:
            split_by = self.split_dataset_by

        _verbose_print(f"Splitting the dataset into proportions {lengths}, by {split_by}.", verbose=self.verbose)

        if not SPLIT_UNIT.has_value(split_by):
            warnings.warn(f'Split unit \'{split_by}\' is not within the list of known split units: '
                          f'\'{[e.value for e in SPLIT_UNIT]}\'. Defaulting to \'{SPLIT_UNIT.SEQUENCE.value}\'')

        # Can use built-in random_split function
        if split_by == SPLIT_UNIT.FRAME.value:
            return random_split(self, lengths, generator=generator)
        elif split_by == SPLIT_UNIT.YEAR.value:
            return self._random_split_by_year(lengths, generator=generator)
        else:  # split_by == SPLIT_UNIT.SEQUENCE.value:
            return self._random_split_by_sequence(lengths, generator=generator)

    def images_from_year(self, year: str) -> List[Subset[int]]:
        """
        Given a start year, return the total dataset image indices of the images from the sequences starting in
        the specified year. Returns it as a List of Subsets, where one inner list represents one sequence.
        :param year: the start year as a string
        :return: the List of Lists of sequences and their image indices
        """
        return_list = []
        sequence_strs = self.get_all_seq_str_from_start_year(year)
        for seq_str in sequence_strs:
            seq_obj = self._get_seq_from_seq_str(seq_str)
            return_list.append(Subset(self, self.seq_indices_to_total_indices(seq_obj)))
        return return_list

    def images_from_sequence(self, sequence_str: str) -> Subset[int]:
        """
        Given a sequence ID, returns a Subset of the dataset of the images in that sequence
        :param sequence_str: str, the sequence ID
        :return: Subset of the total dataset
        """
        seq_object = self._get_seq_from_seq_str(sequence_str)
        indices = self.seq_indices_to_total_indices(seq_object)
        return Subset(self, indices)

    def get_number_of_sequences(self):
        """
        Gets number of sequences (typhoons) in the dataset
        :return: integer number of sequences
        """
        return len(self.sequences)

    def sequence_exists(self, seq_str: str) -> bool:
        """
        Returns if a seq_str with given seq_str number exists in the dataset
        :param seq_str: string of the seq_str ID
        :return: Boolean True if present, False otherwise
        """
        return seq_str in self._sequence_str_to_seq_idx

    def process_metadata_file(self, filepath: str):
        """
        Reads and processes JSON metadata file's information into dataset.
        :param filepath: path to metadata file
        :return: metadata JSON object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.number_of_sequences = len(data)

        for sequence_str, metadata in data.items():
            self._read_one_seq_from_metadata(sequence_str, metadata)

    def get_all_seq_str_from_start_year(self, year: str) -> List[str]:
        """
        Given a start year, give the sequence ID strings of all sequences that start in that year.
        :param year: the start year as a string
        :return: a list of the sequence IDs starting in that year
        """
        if year not in self.years_to_sequence_nums:
            raise ValueError(f'Year \'{year}\' is not within the list of start years.')
        return self.years_to_sequence_nums[year]

    def total_frame_idx_to_sequence_idx(self, total_idx: int) -> int:
        """
        Given a total dataset image index, returns that image's index in its respective sequence. e.g. an image that is
        the 500th in the total dataset may be the 5th image in its sequence.
        :param total_idx: the total dataset image index
        :return: the inner-sequence image index.
        """
        sequence = self._frame_idx_to_sequence[total_idx]
        start_idx = self._seq_str_to_first_total_idx[sequence.get_sequence_str()]
        if total_idx >= self.number_of_original_frames:
            raise ValueError(f'Frame {total_idx} is beyond the number of frames in the dataset.')
        return total_idx - start_idx

    def seq_idx_to_total_frame_idx(self, seq_str: str, seq_idx: int) -> int:
        """
        Given an image with seq_idx position within its sequence, return its total idx within the greater dataset. e.g.
        an image that is the 5th image in the sequence may be the 500th in the total dataset.
        :param seq_str: The sequence ID string to search within
        :param seq_idx: int, the index within the given sequence
        :return: int, the total index within the dataset
        """
        sequence_obj = self._get_seq_from_seq_str(seq_str)
        if seq_idx >= sequence_obj.get_num_images():
            raise ValueError(f'Frame {seq_idx} is beyond the number of frames in the dataset.')
        return self._seq_str_to_first_total_idx[seq_str] + seq_idx

    def seq_indices_to_total_indices(self, seq_obj: DigitalTyphoonSequence) -> List[int]:
        """
        Given a sequence, return a list of the total dataset indices of the sequence's images.
        :param seq_obj: the DigitalTyphoonSequence object to produce the list from
        :return: the List of total dataset indices
        """
        seq_str = seq_obj.get_sequence_str()
        return [i + self._seq_str_to_first_total_idx[seq_str] for i in range(seq_obj.get_num_images())]

    def _get_list_of_sequence_objs(self) -> List[DigitalTyphoonSequence]:
        """
        Gives list of seq_str objects
        :return: List[DigitalTyphoonSequence]
        """
        return self.sequences

    def _populate_images_into_sequences(self, image_dir: str) -> None:
        """
        Traverses the image directory and populates each of the images sequentially into their respective seq_str
        objects.
        :param image_dir: path to directory containing directory of typhoon images.
        :return: None
        """
        load_into_mem = self.load_data_into_memory in {LOAD_DATA.ONLY_IMG, LOAD_DATA.ALL_DATA}
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for dir_name in sorted(dirs):  # Read sequences in chronological order, not necessary but convenient
                sequence_obj = self._get_seq_from_seq_str(dir_name)
                sequence_obj.process_seq_img_dir_into_sequence(root+dir_name, load_into_mem, ignore_list=self.ignore_list)
                self.number_of_frames += sequence_obj.get_num_images()

        for sequence in self.sequences:
            if not sequence.num_images_match_num_frames():
                if self.verbose:
                    warnings.warn(f'Sequence {sequence.sequence_str} has only {sequence.get_num_images()} when '
                                  f'it should have {sequence.num_frames}. If this is intended, ignore this warning.')

    def _populate_track_data_into_sequences(self, track_dir: str) -> None:
        """
        Traverses the track data files and populates each into their respective seq_str objects
        :param track_dir: path to directory containing track data files
        :return: None
        """
        for root, dirs, files in os.walk(track_dir, topdown=True):
            for file in files:
                file_sequence = get_seq_str_from_track_filename(file)
                if self.sequence_exists(file_sequence):
                    self._get_seq_from_seq_str(file_sequence).set_track_path(root + file)
                    # if self.load_data_into_memory in {LOAD_DATA.ONLY_TRACK, LOAD_DATA.ALL_DATA}:
                    self._read_in_track_file_to_sequence(file_sequence, root + file)

    def _read_one_seq_from_metadata(self, sequence_str: str,
                                    metadata_json: Dict):
        """
        Processes one seq_str from the metadata JSON object.
        :param sequence_str: string of the seq_str ID
        :param metadata_json: JSON object from metadata file
        :param prev_interval_end: the final image index of the previous seq_str
        :return: None
        """
        seq_start_date = datetime.strptime(metadata_json['start'], '%Y-%m-%d')

        self.sequences.append(DigitalTyphoonSequence(sequence_str,
                                                     seq_start_date.year,
                                                     metadata_json['frames'],
                                                     verbose=self.verbose))
        self._sequence_str_to_seq_idx[sequence_str] = len(self.sequences) - 1

        if metadata_json['year'] not in self.years_to_sequence_nums:
            self.years_to_sequence_nums[metadata_json['year']] = []
        self.years_to_sequence_nums[metadata_json['year']].append(sequence_str)
        self.number_of_original_frames += metadata_json['frames']


    def _assign_all_images_a_dataset_idx(self):
        """
        Iterates through the sequences and assigns each image (AFTER adding and removing images to the sequences, i.e.
        not the number of original frames stated in the metadata.json) an index within the total dataset.
        :return: None
        """
        dataset_idx_iter = 0
        for sequence in self.sequences:
            self._seq_str_to_first_total_idx[sequence.get_sequence_str()] = dataset_idx_iter
            for idx in range(sequence.get_num_images()):
                self._frame_idx_to_sequence[dataset_idx_iter] = sequence
                dataset_idx_iter += 1

    def _read_in_track_file_to_sequence(self, seq_str: str, file: str, csv_delimiter=',') -> DigitalTyphoonSequence:
        """
        Processes one track file into its seq_str.
        :param seq_str: string of the seq_str ID
        :param file: path to the track file
        :param csv_delimiter: delimiter used in the track csv files
        :return: the DigitalTyphoonSequence object that was just populated
        """
        sequence = self._get_seq_from_seq_str(seq_str)
        sequence.process_track_data(file, csv_delimiter)
        return sequence

    def _calculate_split_lengths(self, lengths: Sequence[Union[int, float]]) -> List[int]:
        """
        Code taken from PyTorch repo. https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

        'If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.'

        :param lengths: Lengths or fractions of splits to be produced
        :return: A list of integers representing the size of the buckets of each split
        """

        dataset_length = self.__len__()
        #  Lengths code taken from:
        #    https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    math.floor(dataset_length * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = dataset_length - sum(subset_lengths)  # type: ignore[arg-type]
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            lengths = subset_lengths
            for i, length in enumerate(lengths):
                if length == 0:
                    warnings.warn(f"Length of split at index {i} is 0. "
                                  f"This might result in an empty dataset.")

            # Cannot verify that dataset is Sized
        if sum(lengths) != dataset_length:  # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        return lengths

    def _random_split_by_year(self, lengths: Sequence[Union[int, float]],
                              generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly splits the dataset s.t. each bucket has close to the requested number of indices in each split.
        Images (indices) from typhoons starting in the same year are not split across different buckets. Indices within
        the same year are given contiguously in the list of indices.

        As a year is treated as an atomic unit, achieving the exact split requested may not be possible. An
        approximation where each bucket is guaranteed to have at least one item is used. Randomization is otherwise
        preserved.

        :param lengths: Lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :return: List of Subset objects
        """
        lengths = self._calculate_split_lengths(lengths)
        return_indices_sorted = [[length, i, []] for i, length in enumerate(lengths)]
        return_indices_sorted.sort(key=lambda x: x[0])
        randomized_year_list = [list(self.years_to_sequence_nums.keys())[i]
                                for i in randperm(len(self.years_to_sequence_nums), generator=generator)]
        year_iter = 0
        for i in range(len(return_indices_sorted)):
            while year_iter < len(self.years_to_sequence_nums) and len(return_indices_sorted[i][2]) < return_indices_sorted[i][0]:
                for seq in self.years_to_sequence_nums[randomized_year_list[year_iter]]:
                    return_indices_sorted[i][2] \
                        .extend(self.seq_indices_to_total_indices(self._get_seq_from_seq_str(seq)))
                year_iter += 1

        return_indices_sorted.sort(key=lambda x: x[1])
        return [Subset(self, bucket_indices) for _, _, bucket_indices in return_indices_sorted]

    def _random_split_by_sequence(self, lengths: Sequence[Union[int, float]],
                                  generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly splits the dataset s.t. each bucket has close to the requested number of indices in each split.
        Images (indices) from a given typhoon are not split across different buckets. Indices within a seq_str
        are given contiguously in the returned Subset.

        As a seq_str is treated as an atomic unit, achieving the exact split requested may not be possible. An
        approximation where each bucket is guaranteed to have at least one item is used. Randomization is otherwise
        preserved.

        :param lengths: Lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :return: List of Subset objects
        """
        lengths = self._calculate_split_lengths(lengths)

        return_indices_sorted = [[length, i, []] for i, length in enumerate(lengths)]
        return_indices_sorted.sort(key=lambda x: x[0])

        randomized_seq_indices = randperm(len(self.sequences), generator=generator)
        seq_iter = 0

        for i in range(len(return_indices_sorted)):
            while seq_iter < len(self.sequences) and len(return_indices_sorted[i][2]) < return_indices_sorted[i][0]:
                sequence_obj = self.sequences[randomized_seq_indices[seq_iter]]
                sequence_idx_to_total = self.seq_indices_to_total_indices(sequence_obj)
                return_indices_sorted[i][2].extend(sequence_idx_to_total)
                seq_iter += 1

        return_indices_sorted.sort(key=lambda x: x[1])

        return_list = [Subset(self, bucket_indices) for _, _, bucket_indices in return_indices_sorted]
        return return_list

    def _get_seq_from_seq_str(self, seq_str: str) -> DigitalTyphoonSequence:
        """
        Gets a sequence object from the sequence ID string
        :param seq_str: sequence ID string
        :return: DigitalTyphoonSequence object corresponding to the Sequence string
        """
        return self.sequences[self._sequence_str_to_seq_idx[seq_str]]

    def _find_sequence_str_from_frame_index(self, idx: int) -> str:
        """
        Given an image index from the whole dataset, returns the sequence ID it belongs to
        :param idx: int, the total dataset image idx
        :return: the sequence string ID it belongs to
        """
        return self._frame_idx_to_sequence[idx].get_sequence_str()

    def _get_image_from_idx(self, idx) -> DigitalTyphoonImage:
        """
        Given a dataset image idx, returns the image object from that index.
        :param idx: int, the total dataset image idx
        :return: DigitalTyphoonImage object for that image
        """
        sequence_str = self._find_sequence_str_from_frame_index(idx)
        sequence = self._get_seq_from_seq_str(sequence_str)
        return sequence.get_image_at_idx(self.total_frame_idx_to_sequence_idx(idx))

    def _get_image_from_idx_as_numpy(self, idx) -> np.ndarray:
        """
        Given a dataset image idx, return the image as a numpy array.
        :param idx: int, the total dataset image idx
        :return: numpy array of the image, with shape of the image's dimensions
        """
        sequence_str = self._find_sequence_str_from_frame_index(idx)
        sequence = self._get_seq_from_seq_str(sequence_str)
        return sequence.get_image_at_idx_as_numpy(self.total_frame_idx_to_sequence_idx(idx))

    def _delete_all_sequences(self):
        """
        Clears all the sequences and other datastructures containing data.
        :return: None
        """
        self.sequences: List[DigitalTyphoonSequence] = list()  # List of seq_str objects
        self._sequence_str_to_seq_idx: Dict[str, int] = {}  # Sequence ID to idx in sequences array
        self._frame_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}  # Image idx to what seq_str it belongs to
        self._seq_str_to_first_total_idx: Dict[str, int] = {}  # Sequence string to the first total idx belonging to
                                                               #  that seq_str
        self.years_to_sequence_nums: OrderedDict[str, List[str]] = OrderedDict()

        self.number_of_sequences = None
        self.number_of_original_frames = 0
