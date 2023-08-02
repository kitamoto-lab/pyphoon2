import math
import os
import json
import warnings
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import List, Sequence, Union, Optional, Dict

import torch
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset, random_split

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
from pyphoon2.DigitalTyphoonUtils import _verbose_print, SPLIT_UNIT, LOAD_DATA, TRACK_COLS, get_seq_str_from_track_filename


class DigitalTyphoonDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 metadata_dir: str,
                 metadata_json: str,
                 labels,
                 split_dataset_by='image',  # can be [sequence, season, image]
                 spectrum='Infrared',
                 get_images_by_sequence=False,
                 load_data_into_memory=False,
                 ignore_list=None,
                 filter_func=None,
                 transform_func=None,
                 transform=None,
                 verbose=False) -> None:
        """
        Dataloader for the DigitalTyphoon dataset.

        :param image_dir: Path to directory containing directories of typhoon sequences
        :param metadata_dir: Path to directory containing track data for typhoon sequences
        :param metadata_json: Path to the metadata JSON file
        :param split_dataset_by: What unit to treat as an atomic unit when randomly splitting the dataset. Options are
                                "sequence", "season", or "image" (individual image)
        :param spectrum: Spectrum to access h5 image files with
        :param get_images_by_sequence: Boolean representing if an index should refer to an individual image or an entire
                                        sequence. If sequence, returned images are Lists of images.
        :param load_data_into_memory:  String representing if the images and track data should be entirely loaded into
                                        memory. Options are "track" (only track data), "images" (only images), or
                                        "all_data" (both track and images).
        :param ignore_list: a list of filenames (not path) to ignore and NOT add to the dataset
        :param filter_func: a function used to filter out images from the dataset. Should accept an DigitalTyphoonImage object
                       and return a bool True or False if it should be included in the dataset
        :param transform_func: this function will be called on the image array for each image when reading in the dataset.
                               It should take and return a numpy image array
        :param transform: Pytorch transform func. Will be called on the tuple of (image/sequence, label array). It should
                         take in said tuple, and return a tuple of (transformed image/sequence, transformed label)
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

        # Bool determining whether an atomic unit should be one image (False) image or one typhoon (True).
        self.get_images_by_sequence = get_images_by_sequence

        # Directories containing image folders and track data
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir

        # Path to the metadata file
        self.metadata_json = metadata_json

        # labels to retrieve when accessing the dataset
        self.labels = None
        self.set_label(labels)

        self.verbose = verbose

        # Spectrum to open images with
        self.spectrum = spectrum

        # Set of image filepaths to ignore
        self.ignore_list = set(ignore_list) if ignore_list else set([])
        if filter_func:
            self.filter = filter_func
        else:
            self.filter = lambda img: True

        self.transform_func = transform_func
        self.transform = transform

        # Structures holding the data objects
        self.sequences: List[DigitalTyphoonSequence] = list()  # List of seq_str objects
                                                                # contains sequences in order they are present in metadata.json
        self._sequence_str_to_seq_idx: Dict[str, int] = {}  # Sequence ID to idx in sequences array
        self._image_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}  # Image idx to what seq_str it belongs to
        self._seq_str_to_first_total_idx: Dict[str, int] = {}  # Sequence string to the first total idx belonging to
                                                               #  that seq_str

        self.label = ('grade', 'lat')

        self.number_of_sequences = 0
        self.number_of_nonempty_sequences = 0
        self.number_of_original_images = 0  # Number of images in the original dataset before augmentation and removal
        self.number_of_images = 0  # number of images in the dataset, after augmentation and removal
        self.number_of_nonempty_seasons = None

        # Season to list of sequences that start in that season
        self.season_to_sequence_nums: OrderedDict[int, List[str]] = OrderedDict()

        # Process the data into the loader
        # It must happen in this order!
        _verbose_print(f'Processing metadata file at: {metadata_json}', self.verbose)
        self.process_metadata_file(metadata_json)

        _verbose_print(f'Initializing track data from: {metadata_dir}', self.verbose)
        self._populate_track_data_into_sequences(self.metadata_dir)

        _verbose_print(f'Initializing image_arrays from: {image_dir}', self.verbose)
        self._populate_images_into_sequences(self.image_dir)

        _verbose_print(f'Indexing the dataset', verbose=self.verbose)
        self._assign_all_images_a_dataset_idx()

    def __len__(self) -> int:
        """
        Gives the length of the dataset. If "get_images_by_sequence" was set to True on initialization, number of
        sequences is returned. Otherwise, number of images is returned.
        
        :return: int
        """
        if self.get_images_by_sequence:
            return self.get_number_of_nonempty_sequences()
        else:
            return self.number_of_images

    def __getitem__(self, idx):
        """
        Gets an image and its label at a particular dataset index.

        If "get_images_by_sequence" was set to True on initialization,
        the idx'th sequence is returned as a np array of the image arrays.

        Otherwise, the single image np array is given.

        Returns a tuple with the image array in the first position, and the label in the second.

        The label will take on the shape of desired labels specified in the class attribute.
        e.g. if the dataset was instantiated with labels='grade', dataset[0] will return image, grade
             If the dataset was instantiated with labels=('lat', 'lng') dataset[0] will return image, [lat, lng]

        :param idx: int, index of image or seq within total dataset
        :return: a List of image arrays and labels, or single image and labels
        """
        if self.get_images_by_sequence:
            seq = self.get_ith_sequence(idx)
            images = seq.get_all_images_in_sequence()
            image_arrays = np.array([image.image() for image in images])
            labels = np.array([self._labels_from_label_strs(image, self.labels) for image in images])
            if self.transform:
                return self.transform((image_arrays, labels))
            return image_arrays, labels
        else:
            image = self.get_image_from_idx(idx)
            labels = self._labels_from_label_strs(image, self.labels)
            ret_img = image.image()
            if self.transform:
                return self.transform((ret_img, labels))
            return ret_img, labels

    def set_label(self, label_strs) -> None:
        """
        Sets what label to retrieve when accessing the data set via dataset[idx] or dataset.__getitem__(idx)
        Options are:
        season, month, day, hour, grade, lat, lng, pressure, wind, dir50, long50, short50, dir30, long30, short30, landfall, interpolated
      
        :param label_strs: a single string (e.g. 'grade') or a list/tuple of strings (e.g. ['lat', 'lng']) of labels.
        :return: None
        """
        if (type(label_strs) is list) or (type(label_strs) is tuple):
            for label in label_strs:
                TRACK_COLS.str_to_value(label)  # For error checking
        else:
            TRACK_COLS.str_to_value(label_strs)  # For error checking
        self.labels = label_strs

    def random_split(self, lengths: Sequence[Union[int, float]],
                     split_by=None,
                     generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        Given a list of proportions or items, returns a random split of the dataset with proportions as close to
        the requested without causing leakage between requested split_unit. If split is by image, built-in PyTorch
        function is used. If split is by season, all images from typhoons starting in the same season will be placed in
        the same bucket. If split is by seq_str, all images from the same typhoon will be together.

        Returns a list of Subsets of indices according to requested lengths. If split is anything other than image,
        indices within their split unit are not randomized. (I.e. indices of a seq_str will be kept contiguous, not
        randomized order mixing with other sequences).

        If "get_images_by_sequence" is set to True on initialization, split_by image and sequence are functionally
        identical, and will split the number of sequences into the requested bucket sizes.
        If split_by='season', then sequences with the same season will be placed in the same bucket.

        Only non-empty sequences are returned in the split.

        For Subset doc see https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset.

        :param lengths: lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :param split_by: What to treat as an atomic unit (image, seq_str, season). Options are
                         "image", "sequence" or "season" respectively
        :return: List[Subset[idx]]
        """
        if split_by is None:
            split_by = self.split_dataset_by

        _verbose_print(f"Splitting the dataset into proportions {lengths}, by {split_by}.", verbose=self.verbose)

        if not SPLIT_UNIT.has_value(split_by):
            warnings.warn(f'Split unit \'{split_by}\' is not within the list of known split units: '
                          f'\'{[e.value for e in SPLIT_UNIT]}\'. Defaulting to \'{SPLIT_UNIT.SEQUENCE.value}\'')

        # Can use built-in random_split function
        if split_by == SPLIT_UNIT.IMAGE.value:
            return random_split(self, lengths, generator=generator)
        elif split_by == SPLIT_UNIT.SEASON.value:
            return self._random_split_by_season(lengths, generator=generator)
        else:  # split_by == SPLIT_UNIT.SEQUENCE.value:
            return self._random_split_by_sequence(lengths, generator=generator)

    def images_from_season(self, season: int) -> Subset:
        """
        Given a start season, return a Subset (Dataset) object containing all the images from that season, in order
        
        :param season: the start season as a string
        :return: Subset
        """
        return_indices = []
        sequence_strs = self.get_seq_ids_from_season(season)
        for seq_str in sequence_strs:
            seq_obj = self._get_seq_from_seq_str(seq_str)
            return_indices.extend(self.seq_indices_to_total_indices(seq_obj))
        return Subset(self, return_indices)

    def image_objects_from_season(self, season: int) -> List:
        """
        Given a start season, return a list of DigitalTyphoonImage objects for images from that season
        
        :param season: the start season as a string
        :return: List[DigitalTyphoonImage]
        """
        return_images = []
        sequence_strs = self.get_seq_ids_from_season(season)
        for seq_str in sequence_strs:
            seq_obj = self._get_seq_from_seq_str(seq_str)
            return_images.extend(seq_obj.get_all_images_in_sequence())
        return return_images

    def images_from_seasons(self, seasons: List[int]):
        """
        Given a list of seasons, returns a dataset Subset containing all images from those seasons, in order

        :param seasons: List of season integers
        :return: Subset
        """
        return_indices = []
        for season in seasons:
            sequence_strs = self.get_seq_ids_from_season(season)
            for seq_str in sequence_strs:
                seq_obj = self._get_seq_from_seq_str(seq_str)
                return_indices.extend(self.seq_indices_to_total_indices(seq_obj))
        return Subset(self, return_indices)

    def images_from_sequence(self, sequence_str: str) -> Subset:
        """
        Given a sequence ID, returns a Subset of the dataset of the images in that sequence

        :param sequence_str: str, the sequence ID
        :return: Subset of the total dataset
        """
        seq_object = self._get_seq_from_seq_str(sequence_str)
        indices = self.seq_indices_to_total_indices(seq_object)
        return Subset(self, indices)

    def image_objects_from_sequence(self, sequence_str: str) -> List:
        """
        Given a sequence ID, returns a list of the DigitalTyphoonImage objects in the sequence in chronological order.

        :param sequence_str:
        :return: List[DigitalTyphoonImage]
        """
        seq_object = self._get_seq_from_seq_str(sequence_str)
        return seq_object.get_all_images_in_sequence()

    def images_from_sequences(self, sequence_strs: List[str]) -> Subset:
        """
        Given a list of sequence IDs, returns a dataset Subset containing all the images within the
        sequences, in order

        :param sequence_strs: List[str], the sequence IDs
        :return: Subset of the total dataset
        """
        return_indices = []
        for sequence_str in sequence_strs:
            seq_object = self._get_seq_from_seq_str(sequence_str)
            return_indices.extend(self.seq_indices_to_total_indices(seq_object))
        return Subset(self, return_indices)

    def images_as_tensor(self, indices: List[int]) -> torch.Tensor:
        """
        Given a list of dataset indices, returns the images as a Torch Tensor

        :param indices: List[int]
        :return: torch Tensor
        """
        images = np.array([self.get_image_from_idx(idx).image() for idx in indices])
        return torch.Tensor(images)

    def labels_as_tensor(self, indices: List[int], label: str) -> torch.Tensor:
        """
        Given a list of dataset indices, returns the specified labels as a Torch Tensor

        :param indices: List[int]
        :param label: str, denoting which label to retrieve
        :return: torch Tensor
        """
        images = [self.get_image_from_idx(idx).value_from_string(label) for idx in indices]
        return torch.Tensor(images)

    def get_number_of_sequences(self):
        """
        Gets number of sequences (typhoons) in the dataset

        :return: integer number of sequences
        """
        return len(self.sequences)

    def get_number_of_nonempty_sequences(self):
        """
        Gets number of sequences (typhoons) in the dataset that have at least 1 image

        :return: integer number of sequences
        """
        return self.number_of_nonempty_sequences

    def get_sequence_ids(self) -> List[str]:
        """
        Returns a list of the sequence ID's in the dataset, as strings

        :return: List[str]
        """
        return list(self._sequence_str_to_seq_idx.keys())

    def get_seasons(self) -> List[int]:
        """
        Returns a list of the seasons that typhoons have started in chronological order

        :return: List[int]
        """
        return sorted([int(season) for season in self.season_to_sequence_nums.keys()])

    def get_nonempty_seasons(self) -> List[int]:
        """
        Returns a list of the seasons that typhoons have started in, that have at least one image, in chronological order
        
        :return: List[int]
        """
        if self.number_of_nonempty_seasons is None:
            self.number_of_nonempty_seasons = 0
            for key, seq_list in self.season_to_sequence_nums.items():
                empty = True
                seq_iter = 0
                while empty and seq_iter < len(seq_list):
                    seq_str = seq_list[seq_iter]
                    seq_obj = self._get_seq_from_seq_str(seq_str)
                    if seq_obj.get_num_images() > 0:
                        self.number_of_nonempty_seasons += 1
                        empty = False
                    seq_iter += 1

        return self.number_of_nonempty_seasons

    def sequence_exists(self, seq_str: str) -> bool:
        """
        Returns if a seq_str with given seq_str number exists in the dataset

        :param seq_str: string of the seq_str ID
        :return: Boolean True if present, False otherwise
        """
        return seq_str in self._sequence_str_to_seq_idx

    def get_ith_sequence(self, idx: int) -> DigitalTyphoonSequence:
        """
        Given an index idx, returns the idx'th sequence in the dataset

        :param idx: int index
        :return: DigitalTyphoonSequence
        """
        if idx >= len(self.sequences):
            raise ValueError(f'Index {idx} is outside the range of sequences.')
        return self.sequences[idx]

    def process_metadata_file(self, filepath: str):
        """
        Reads and processes JSON metadata file's information into dataset.

        :param filepath: path to metadata file
        :return: metadata JSON object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.number_of_sequences = len(data)

        for sequence_str, metadata in sorted(data.items()):
            self._read_one_seq_from_metadata(sequence_str, metadata)

    def get_seq_ids_from_season(self, season: int) -> List[str]:
        """
        Given a start season, give the sequence ID strings of all sequences that start in that season.

        :param season: the start season as a string
        :return: a list of the sequence IDs starting in that season
        """
        if season not in self.season_to_sequence_nums:
            raise ValueError(f'Season \'{season}\' is not within the list of start seasons.')
        return self.season_to_sequence_nums[season]

    def total_image_idx_to_sequence_idx(self, total_idx: int) -> int:
        """
        Given a total dataset image index, returns that image's index in its respective sequence. e.g. an image that is
        the 500th in the total dataset may be the 5th image in its sequence.

        :param total_idx: the total dataset image index
        :return: the inner-sequence image index.
        """
        sequence = self._image_idx_to_sequence[total_idx]
        start_idx = self._seq_str_to_first_total_idx[sequence.get_sequence_str()]
        if total_idx >= self.number_of_images:
            raise ValueError(f'Image {total_idx} is beyond the number of images in the dataset.')
        return total_idx - start_idx

    def seq_idx_to_total_image_idx(self, seq_str: str, seq_idx: int) -> int:
        """
        Given an image with seq_idx position within its sequence, return its total idx within the greater dataset. e.g.
        an image that is the 5th image in the sequence may be the 500th in the total dataset.

        :param seq_str: The sequence ID string to search within
        :param seq_idx: int, the index within the given sequence
        :return: int, the total index within the dataset
        """
        sequence_obj = self._get_seq_from_seq_str(seq_str)
        if seq_idx >= sequence_obj.get_num_images():
            raise ValueError(f'Image {seq_idx} is beyond the number of images in the dataset.')
        return self._seq_str_to_first_total_idx[seq_str] + seq_idx

    def seq_indices_to_total_indices(self, seq_obj: DigitalTyphoonSequence) -> List[int]:
        """
        Given a sequence, return a list of the total dataset indices of the sequence's images.

        :param seq_obj: the DigitalTyphoonSequence object to produce the list from
        :return: the List of total dataset indices
        """
        seq_str = seq_obj.get_sequence_str()
        return [i + self._seq_str_to_first_total_idx[seq_str] for i in range(seq_obj.get_num_images())]

    def get_image_from_idx(self, idx) -> DigitalTyphoonImage:
        """
        Given a dataset image idx, returns the image object from that index.

        :param idx: int, the total dataset image idx
        :return: DigitalTyphoonImage object for that image
        """
        sequence_str = self._find_sequence_str_from_image_index(idx)
        sequence = self._get_seq_from_seq_str(sequence_str)
        return sequence.get_image_at_idx(self.total_image_idx_to_sequence_idx(idx))

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
                sequence_obj.process_seq_img_dir_into_sequence(root+dir_name, load_into_mem,
                                                               ignore_list=self.ignore_list,
                                                               filter_func=self.filter,
                                                               spectrum=self.spectrum)
                self.number_of_images += sequence_obj.get_num_images()

        for sequence in self.sequences:
            if sequence.get_num_images() > 0:
                self.number_of_nonempty_sequences += 1

            if not sequence.num_images_match_num_expected():
                if self.verbose:
                    warnings.warn(f'Sequence {sequence.sequence_str} has only {sequence.get_num_images()} when '
                                  f'it should have {sequence.num_original_images}. If this is intended, ignore this warning.')

    def _populate_track_data_into_sequences(self, metadata_dir: str) -> None:
        """
        Traverses the track data files and populates each into their respective seq_str objects

        :param metadata_dir: path to directory containing track data files
        :return: None
        """
        for root, dirs, files in os.walk(metadata_dir, topdown=True):
            for file in sorted(files):
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
                                                     metadata_json['images'],
                                                     transform_func=self.transform_func,
                                                     spectrum=self.spectrum,
                                                     verbose=self.verbose))
        self._sequence_str_to_seq_idx[sequence_str] = len(self.sequences) - 1

        if metadata_json['season'] not in self.season_to_sequence_nums:
            self.season_to_sequence_nums[metadata_json['season']] = []
        self.season_to_sequence_nums[metadata_json['season']].append(sequence_str)
        self.number_of_original_images += metadata_json['images']

    def _assign_all_images_a_dataset_idx(self):
        """
        Iterates through the sequences and assigns each image (AFTER adding and removing images to the sequences, i.e.
        not the number of original images stated in the metadata.json) an index within the total dataset.
        :return: None
        """
        dataset_idx_iter = 0
        for sequence in self.sequences:
            self._seq_str_to_first_total_idx[sequence.get_sequence_str()] = dataset_idx_iter
            for idx in range(sequence.get_num_images()):
                self._image_idx_to_sequence[dataset_idx_iter] = sequence
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

    def _random_split_by_season(self, lengths: Sequence[Union[int, float]],
                                generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly splits the dataset s.t. each bucket has close to the requested number of indices in each split.
        Images (indices) from typhoons starting in the same season are not split across different buckets. Indices within
        the same season are given contiguously in the list of indices.

        As a season is treated as an atomic unit, achieving the exact split requested may not be possible. An
        approximation where each bucket is guaranteed to have at least one item is used. Randomization is otherwise
        preserved.

        If "get_images_by_sequence" was set to True, then each returned index refers to a sequence. No season will be
        split between two buckets.

        Only non-empty sequences are returned in the split.

        :param lengths: Lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :return: List of Subset objects
        """
        lengths = self._calculate_split_lengths(lengths)
        return_indices_sorted = [[length, i, []] for i, length in enumerate(lengths)]
        return_indices_sorted.sort(key=lambda x: x[0])

        non_empty_season_indices = []
        for idx, item in enumerate(self.season_to_sequence_nums.items()):
            key, val = item
            nonempty = False
            for seq_id in val:
                if self._get_seq_from_seq_str(seq_id).get_num_images() > 0:
                    nonempty = True
                    break
            if nonempty:
                non_empty_season_indices.append(idx)
        non_empty_season_indices = [non_empty_season_indices[idx] for idx in randperm(len(non_empty_season_indices), generator=generator)]
        randomized_season_list = [list(self.season_to_sequence_nums.keys())[i] for i in non_empty_season_indices]

        num_buckets = len(return_indices_sorted)
        bucket_counter = 0
        season_iter = 0
        while season_iter < len(randomized_season_list):
            if len(return_indices_sorted[bucket_counter][2]) < return_indices_sorted[bucket_counter][0]:
                for seq in self.season_to_sequence_nums[randomized_season_list[season_iter]]:
                    sequence_obj = self._get_seq_from_seq_str(seq)
                    if self.get_images_by_sequence:
                        if sequence_obj.get_num_images() > 0:  # Only append if the sequence has images
                            return_indices_sorted[bucket_counter][2].append(self._sequence_str_to_seq_idx[seq])
                    else:
                        return_indices_sorted[bucket_counter][2] \
                            .extend(self.seq_indices_to_total_indices(self._get_seq_from_seq_str(seq)))
                season_iter += 1
            bucket_counter += 1
            if bucket_counter == num_buckets:
                bucket_counter = 0

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

        If "get_images_by_sequence" was set to true, then random_split returns buckets containing indices referring to
        entire sequences. As an atomic unit is a sequence, this function adds no extra functionality over
        the default random_split function.

        Only non-empty sequences are returned in the split.

        :param lengths: Lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :return: List of Subset objects
        """
        lengths = self._calculate_split_lengths(lengths)
        return_indices_sorted = [[length, i, []] for i, length in enumerate(lengths)]
        return_indices_sorted.sort(key=lambda x: x[0])
        num_buckets = len(return_indices_sorted)

        non_empty_sequence_indices = [idx for idx in range(len(self.sequences)) if self.sequences[idx].get_num_images() > 0]
        randomized_seq_indices = [non_empty_sequence_indices[idx] for idx in randperm(len(non_empty_sequence_indices), generator=generator)]

        bucket_counter = 0
        seq_iter = 0
        while seq_iter < len(randomized_seq_indices):
            if len(return_indices_sorted[bucket_counter][2]) < return_indices_sorted[bucket_counter][0]:
                sequence_obj = self.sequences[randomized_seq_indices[seq_iter]]
                if self.get_images_by_sequence:
                    return_indices_sorted[bucket_counter][2].append(randomized_seq_indices[seq_iter])
                else:
                    sequence_idx_to_total = self.seq_indices_to_total_indices(sequence_obj)
                    return_indices_sorted[bucket_counter][2].extend(sequence_idx_to_total)
                seq_iter += 1
            bucket_counter += 1
            if bucket_counter == num_buckets:
                bucket_counter = 0
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

    def _find_sequence_str_from_image_index(self, idx: int) -> str:
        """
        Given an image index from the whole dataset, returns the sequence ID it belongs to

        :param idx: int, the total dataset image idx
        :return: the sequence string ID it belongs to
        """
        return self._image_idx_to_sequence[idx].get_sequence_str()

    def _get_image_from_idx_as_numpy(self, idx) -> np.ndarray:
        """
        Given a dataset image idx, return the image as a numpy array.

        :param idx: int, the total dataset image idx
        :return: numpy array of the image, with shape of the image's dimensions
        """
        sequence_str = self._find_sequence_str_from_image_index(idx)
        sequence = self._get_seq_from_seq_str(sequence_str)
        return sequence.get_image_at_idx_as_numpy(self.total_image_idx_to_sequence_idx(idx))

    def _labels_from_label_strs(self, image: DigitalTyphoonImage, label_strs):
        """
        Given an image and the label/labels to retrieve from the image, returns a single label or
        a list of labels

        :param image: image to access labels for
        :param label_strs: either a List of label strings or a single label string
        :return: a List of label strings or a single label string
        """
        if (type(label_strs) is list) or (type(label_strs) is tuple):
            label_ray = np.array([image.value_from_string(label) for label in label_strs])
            return label_ray
        else:
            label = image.value_from_string(label_strs)
            return label

    def _delete_all_sequences(self):
        """
        Clears all the sequences and other datastructures containing data.
        :return: None
        """
        self.sequences: List[DigitalTyphoonSequence] = list()  # List of seq_str objects
        self._sequence_str_to_seq_idx: Dict[str, int] = {}  # Sequence ID to idx in sequences array
        self._image_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}  # Image idx to what seq_str it belongs to
        self._seq_str_to_first_total_idx: Dict[str, int] = {}  # Sequence string to the first total idx belonging to
                                                               #  that seq_str
        self.season_to_sequence_nums: OrderedDict[str, List[str]] = OrderedDict()

        self.number_of_sequences = 0
        self.number_of_original_images = 0
        self.number_of_images = 0