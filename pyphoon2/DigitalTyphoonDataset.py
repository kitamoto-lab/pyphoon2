import math
import os
import json
import warnings
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import List, Sequence, Union, Optional, Dict
import re
import glob
import random

import torch
from torch import default_generator, randperm, Generator
from torch.utils.data import Dataset, Subset, random_split

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
from pyphoon2.DigitalTyphoonUtils import _verbose_print, SPLIT_UNIT, LOAD_DATA, TRACK_COLS, get_seq_str_from_track_filename, parse_image_filename, is_image_file, parse_common_image_filename


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
                 verbose=False,
                 image_dirs: List[str] = None,
                 metadata_dirs: List[str] = None,
                 metadata_jsons: List[str] = None
                 ) -> None:
        """
        Implementation of pytorch dataset for the Digital Typhoon Dataset, allows both random iteration and
        deterministic splitting based on image, sequence or season.

        :param image_dir: str, path to directory containing h5 image files
        :param metadata_dir: str, path to directory containing track data in csv files
        :param metadata_json: str, path to metadata json containing some metadata about sequences and images
        :param labels: str or tuple of str, which columns from the track data to return as labels
        :param split_dataset_by: str in [sequence, season, image], determines which objects are kept fully within splits
        :param spectrum: str, spectrum of images to load
        :param get_images_by_sequence: bool, returns entire sequences at a time, if True, __getitem__(i) will return all images in one sequence
        :param load_data_into_memory: bool, determines if images should be loaded into memory when instantiated
        :param ignore_list: List of filenames to ignore (just the filenames, not paths)
        :param filter_func: function taking a DigitalTyphoonImage object and returning a bool; if True returned, image is kept
        :param transform_func: function applied after filter func, taking and returning a np array
        :param transform: function to apply to both image and label tensors, takes and returns a tensor
        :param verbose: bool, flag for additional logging output
        :param image_dirs: List[str], list of paths to directories containing h5 image files (for multi-source)
        :param metadata_dirs: List[str], list of paths to directories containing track data (for multi-source)
        :param metadata_jsons: List[str], list of paths to metadata json files (for multi-source)
        """
        # Core parameters
        self.verbose = verbose
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.metadata_json = metadata_json
        self.spectrum = spectrum
        self.load_data_into_memory = load_data_into_memory
        self.split_dataset_by = split_dataset_by
        self.get_images_by_sequence = get_images_by_sequence
        
        # Initialize multi-directory attributes first
        self.image_dirs = image_dirs or []
        self.metadata_dirs = metadata_dirs or []
        self.metadata_jsons = metadata_jsons or []
        
        # NOW check if loading from multiple directories (after attributes are initialized)
        self.load_from_multi_dirs = self.is_valid_input_multi_dirs(
            image_dirs, metadata_dirs, metadata_jsons)
        
        # Store ignore_list as a set for faster lookups
        self.ignore_list = set(ignore_list) if ignore_list else []
        
        # Functions for processing
        self.filter_func = filter_func
        self.transform_func = transform_func
        self.transform = transform

        # Input validation for split unit
        if not SPLIT_UNIT.has_value(split_dataset_by):
            raise ValueError(f'Split unit must one of the following\n'
                             f'    {[item.value for item in SPLIT_UNIT]}.\n'
                             f'    Input: {split_dataset_by}')
        self.split_dataset_by = split_dataset_by

        # Input validation for load data option
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

        # Directories containing images folders and track datas
        self.image_dirs = image_dirs or []
        self.metadata_dirs = metadata_dirs or []

        # List of metadata files
        self.metadata_jsons = metadata_jsons or []

        # labels to retrieve when accessing the dataset
        self.labels = None
        self.set_label(labels)

        # Spectrum to open images with
        self.spectrum = spectrum

        # List of DigitalTyphoonSequence objects, each a typhoon
        self.sequences: List[DigitalTyphoonSequence] = []

        # Dictionary mapping sequence strings to index in the sequences list
        self._sequence_str_to_seq_idx: Dict[str, int] = {}

        # Dictionary mapping sequence strings to the first total index that sequence's images start at
        self._seq_str_to_first_total_idx: Dict[str, int] = {}

        # Keep track of the total number of sequences, number of images, and the original number of images
        # The original num of images is the number listed in the metadata.json file
        # The actual number of images is the number we actually read in and have in our dataset.
        # These can be different due to filtering or missing images.
        self.number_of_sequences = 0
        self.number_of_nonempty_sequences = 0
        self.number_of_images = 0
        self.number_of_original_images = 0
        
        # Dictionary mapping season to list of sequences (strings)
        self.season_to_sequence_nums: Dict[int, List[str]] = {}
        
        # Dictionary mapping the total image idx to a sequence object
        self._image_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}
        
        # Additional attribute for caching nonempty seasons
        self.number_of_nonempty_seasons = None

        # Load data
        if self.is_valid_input_multi_dirs(image_dirs, metadata_dirs, metadata_jsons):
            self.load_data_from_multi_dirs(image_dirs, metadata_dirs, metadata_jsons)
        else:
            self.load_data_from_single_dir(metadata_json, metadata_dir, image_dir)
            
        if verbose:
            print("Done initializing Digital Typhoon Dataset.")

        # Compute mapping from total image index to sequence object
        self._assign_all_images_a_dataset_idx()

        _verbose_print(f'***DigitalTyphoonDataset statistics***', self.verbose)
        _verbose_print(
            f'Loaded {len(self.sequences)} sequences with {self.number_of_images} total images.', self.verbose)

    def __len__(self) -> int:
        """
        Gives the length of the dataset. If "get_images_by_sequence" was set to True on initialization, number of
        sequences is returned. Otherwise, number of images is returned.

        :return: int
        """
        if self.get_images_by_sequence:
            # When get_images_by_sequence is True, return total number of sequences
            # regardless of whether they have images after filtering
            return self.get_number_of_sequences()
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
            # breakpoint()
            labels = np.array([self._labels_from_label_strs(
                image, self.labels) for image in images])
            
            # Debug prints
            print(f"Returning sequence {seq.get_sequence_str()} with {len(images)} images, shape: {image_arrays.shape}, labels shape: {labels.shape}")
            
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

    def is_valid_input_multi_dirs(self, image_dirs: List[str] = None, metadata_dirs: List[str] = None, metadata_jsons: List[str] = None) -> bool:
        """
        Check if there are valid inputs to load data from multiple directories.
        
        :param image_dirs: list of paths to image directories
        :param metadata_dirs: list of paths to metadata directories
        :param metadata_jsons: list of paths to metadata json files
        :return: boolean indicating if these inputs are valid for multi-directory loading
        """
        # Use parameters directly instead of instance attributes
        img_dirs = image_dirs or []
        meta_dirs = metadata_dirs or []
        meta_jsons = metadata_jsons or []
        
        # Check if we have valid inputs for multi-directory loading
        has_image_dirs = len(img_dirs) > 0
        has_metadata_dirs = len(meta_dirs) > 0
        has_metadata_jsons = len(meta_jsons) > 0
        
        if not (has_image_dirs and has_metadata_dirs and has_metadata_jsons):
            return False
            
        # Check consistency
        if len(img_dirs) != len(meta_dirs) or len(meta_dirs) != len(meta_jsons):
            if self.verbose:
                print("Warning: Inconsistent number of directories for multi-directory loading")
                print(f"Image dirs: {len(img_dirs)}, Metadata dirs: {len(meta_dirs)}, Metadata jsons: {len(meta_jsons)}")
            return False
            
        # Check existence
        for directory in img_dirs:
            if not os.path.exists(directory):
                if self.verbose:
                    print(f"Warning: Image directory {directory} does not exist")
                return False
                
        for directory in meta_dirs:
            if not os.path.exists(directory):
                if self.verbose:
                    print(f"Warning: Metadata directory {directory} does not exist")
                return False
                
        for json_file in meta_jsons:
            if not os.path.exists(json_file):
                if self.verbose:
                    print(f"Warning: Metadata JSON file {json_file} does not exist")
                return False
                
        return True

    def load_data_from_multi_dirs(self, image_dirs: List[str] = None, metadata_dirs: List[str] = None, metadata_jsons: List[str] = None):
        """
        Load data from multiple directories for multi-channel processing.
        
        :param image_dirs: List of image directory paths
        :param metadata_dirs: List of metadata directory paths
        :param metadata_jsons: List of metadata JSON file paths
        :return: None
        """
        # Use instance variables if not provided as arguments
        image_dirs = image_dirs or self.image_dirs
        metadata_dirs = metadata_dirs or self.metadata_dirs
        metadata_jsons = metadata_jsons or self.metadata_jsons
        
        print("\nLoading data from multiple directories:")
        print(f"  Image dirs: {image_dirs}")
        print(f"  Metadata dirs: {metadata_dirs}")
        print(f"  Metadata JSONs: {metadata_jsons}")
        
        if not self.is_valid_input_multi_dirs(image_dirs, metadata_dirs, metadata_jsons):
            print("Invalid multi-directory input, cannot load data.")
            return
        
        try:
            # Find common sequences across all data sources
            common_sequences = self.get_common_sequences_from_files(
                metadata_jsons, metadata_dirs, image_dirs)
            
            if not common_sequences:
                print("No common sequences found across data sources.")
                return
            
            print(f"Found {len(common_sequences)} common sequences")
            
            # Process metadata JSONs
            for metadata_json in metadata_jsons:
                if os.path.exists(metadata_json):
                    self.preprocess_metadata_json_with_common_sequences(
                        metadata_json, common_sequences)
                else:
                    print(f"Warning: Metadata JSON file not found: {metadata_json}")
            
            # Process track data
            for metadata_dir in metadata_dirs:
                if os.path.exists(metadata_dir):
                    self._populate_track_data_into_sequences(
                        metadata_dir, common_sequences)
                else:
                    print(f"Warning: Metadata directory not found: {metadata_dir}")
            
            # Process images
            self._populate_images_into_sequences_from_multi_dirs(
                image_dirs, common_sequences)
            
        except Exception as e:
            print(f"Error loading data from multiple directories: {str(e)}")
            import traceback
            traceback.print_exc()

    def _populate_images_into_sequences_from_multi_dirs(self, root_image_dirs: List[str] = None, common_sequences: List[str] = None):
        """
        Populates sequence objects with images from multiple directories (for multi-channel input).
        Each directory corresponds to a different channel.

        :param root_image_dirs: List of paths to top-level directories containing sequence folders
        :param common_sequences: List of sequence IDs common to all directories
        :return: None
        """
        # Ensure we have inputs
        if not root_image_dirs or not common_sequences:
            print("Missing required inputs for populating sequences from multiple directories")
            return
        
        print("\nPopulating images from multiple directories...")
        print(f"Root directories: {root_image_dirs}")
        print(f"Processing {len(common_sequences)} common sequences")
        
        # Handle special case - single directory
        if len(root_image_dirs) == 1:
            print("Only one directory provided - using single-directory processing")
            for common_sequence in common_sequences:
                sequence_obj = self._get_seq_from_seq_str(common_sequence)
                sequence_path = os.path.join(root_image_dirs[0], common_sequence)
                
                if not os.path.exists(sequence_path):
                    print(f"Warning: Sequence directory {sequence_path} does not exist")
                    continue
                    
                print(f"Processing sequence: {common_sequence} from {sequence_path}")
                sequence_obj.process_seq_img_dir_into_sequence(
                    sequence_path,
                    load_imgs_into_mem=self.load_data_into_memory in {LOAD_DATA.ONLY_IMG, LOAD_DATA.ALL_DATA},
                    ignore_list=self.ignore_list,
                    filter_func=self.filter_func,
                    spectrum=self.spectrum
                )
                
                self.number_of_images += sequence_obj.get_num_images()
                if sequence_obj.get_num_images() > 0:
                    self.number_of_nonempty_sequences += 1
                    
                print(f"Sequence {common_sequence} now has {sequence_obj.get_num_images()} images")
            return
        
        # Setup for multi-directory processing
        load_into_mem = self.load_data_into_memory in {LOAD_DATA.ONLY_IMG, LOAD_DATA.ALL_DATA}
        
        # Reset counters
        self.number_of_images = 0
        self.number_of_nonempty_sequences = 0
        
        # Process each sequence
        sequence_idx = 0
        for common_sequence in common_sequences:
            # Get the sequence object
            sequence_obj = self._get_seq_from_seq_str(common_sequence)
            print(f"\nProcessing sequence {sequence_idx+1}/{len(common_sequences)}: {common_sequence}")
            sequence_idx += 1
            
            # Get the sequence directories
            sequence_dirs = []
            for root_dir in root_image_dirs:
                seq_dir = os.path.join(root_dir, common_sequence)
                if os.path.isdir(seq_dir):
                    sequence_dirs.append(seq_dir)
                else:
                    print(f"Warning: Directory not found: {seq_dir}")
            
            if not sequence_dirs:
                print(f"No valid directories found for sequence {common_sequence}, skipping")
                continue
            
            # Find common images across all sequence directories - directly search for image files
            print(f"Finding common images across {len(sequence_dirs)} directories...")
            images_by_dir = {}
            
            # Collect image files from each directory

            for seq_dir in sequence_dirs:
                try:
                    # Get all files in this directory
                    all_files = os.listdir(seq_dir)
                    
                    # Filter for image files
                    image_files = []
                    for filename in all_files:
                        if is_image_file(filename):
                            # Get base name without channel index
                            base_name = self.get_name_image_remove_channel_idx(filename)
                            if base_name not in self.ignore_list:
                                image_files.append(base_name)
                    
                    # Store unique base names
                    images_by_dir[seq_dir] = list(set(image_files))
                    print(f"  Found {len(images_by_dir[seq_dir])} unique image names in {seq_dir}")
                    
                    # Show a few example files for debugging
                    if len(all_files) > 0:
                        print(f"  Sample files: {all_files[:3]}")
                    if len(image_files) > 0:
                        print(f"  Sample image names: {image_files[:3]}")
                except Exception as e:
                    print(f"  Error processing directory {seq_dir}: {str(e)}")
                    images_by_dir[seq_dir] = []
            
            # Find image names common to all directories
            if not images_by_dir or all(len(imgs) == 0 for imgs in images_by_dir.values()):
                print(f"  No images found in any directory for sequence {common_sequence}")
                continue
            
            # Start with names from first directory
            common_images = set(next(iter(images_by_dir.values())))
            
            # Find intersection with all other directories
            for images in images_by_dir.values():
                common_images.intersection_update(set(images))
            
            print(f"  Found {len(common_images)} common images across all directories")
            
            # Skip if no common images
            if not common_images:
                print(f"  No common images found for sequence {common_sequence}")
                continue
            
            # Process this sequence with the common images
            try:
                sequence_obj.process_seq_img_dirs_into_sequence(
                    directory_paths=sequence_dirs,
                    common_image_names=list(common_images),
                    load_imgs_into_mem=load_into_mem,
                    ignore_list=self.ignore_list,
                    filter_func=self.filter_func,
                    spectrum=self.spectrum
                )
                
                # Update sequence counts
                num_images = sequence_obj.get_num_images()
                self.number_of_images += num_images
                
                if num_images > 0:
                    self.number_of_nonempty_sequences += 1
                    
                print(f"  Sequence {common_sequence} now has {num_images} images")
                
            except Exception as e:
                print(f"  Error processing sequence {common_sequence}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Log summary
        print(f"\nFinished populating images: {self.number_of_images} total images across {self.number_of_nonempty_sequences} non-empty sequences")
        
        # Sanity check
        if self.number_of_images == 0:
            print("WARNING: No images were loaded! Check your directories and file naming consistency.")
        if self.number_of_nonempty_sequences == 0:
            print("WARNING: No sequences contain images! Check your filtering and image loading logic.")

    def get_names_from_images_removed_channel_idx(self, image_dir: str) -> List[str]:
        """
        Gets all image names in an image directory with their channel index removed.
        
        Example:
        A directory with:
        - 000_Infrared-0.h5
        - 000_Infrared-1.h5
        - 001_Infrared-0.h5
        - 001_Infrared-1.h5
        
        Returns: ['000_Infrared', '001_Infrared']
        
        :param image_dir: path to the directory containing image files
        :return: list of image names with channel index removed
        """
        print(f"Scanning directory: {image_dir}")
        
        # Check if directory exists
        if not os.path.exists(image_dir):
            print(f"Warning: Directory {image_dir} does not exist")
            return []
        
        # Get all files in the directory
        try:
            files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and is_image_file(f)]
            print(f"Found {len(files)} image files in {image_dir}")
        except Exception as e:
            print(f"Error listing files in {image_dir}: {str(e)}")
            return []
        
        # Extract base names (removing channel index)
        names = set()
        for file in files:
            try:
                name = self.get_name_image_remove_channel_idx(file)
                names.add(name)
            except Exception as e:
                print(f"Error processing filename {file}: {str(e)}")
                continue
        
        # Convert set to list and return
        result = list(names)
        print(f"Extracted {len(result)} unique image names after removing channel indices")
        if len(result) > 0:
            print(f"Sample names: {result[:3]}")
        
        return result

    def get_name_image_remove_channel_idx(self, image_name: str) -> str:
        match = re.match(r"^(.*)-\d+\.h5", image_name)
        if match:
            name = match.group(1)
            return name
        else:
            return image_name

    def preprocess_metadata_json_with_common_sequences(self, metadata_json: str, common_sequences: List[str]):
        with open(metadata_json, 'r') as f:
            data = json.load(f)
        new_data = {}
        for sequence_str, metadata in data.items():
            if sequence_str in common_sequences:
                new_data[sequence_str] = metadata
        self.number_of_sequences += len(new_data)
        for sequence_str, metadata in sorted(new_data.items()):
            self._read_one_seq_from_metadata(sequence_str, metadata)

    def load_data_from_single_dir(self, metadata_json: str, metadata_dir: str, image_dir: str):
        """
        Load data from a single directory.
        :param metadata_json: Path to the metadata JSON file.
        :param metadata_dir: Path to the metadata directory.
        :param image_dir: Path to the image directory.
        :return: None.
        """
        # Process the metadata JSON file
        self.process_metadata_file(metadata_json)
        
        # IMPORTANT: Load track data first, so it's available when loading images
        if self.verbose:
            print("Loading track data first to ensure it's available for images...")
        self._populate_track_data_into_sequences(metadata_dir)
        
        # Now populate image data - images will be able to access track data
        if self.verbose:
            print("Now loading images, which can use the track data...")
        self._populate_images_into_sequences(image_dir)
        
        # Assign indices
        self._assign_all_images_a_dataset_idx()

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
                     generator: Optional[Generator] = default_generator,
                     split_by='default') -> List[Subset]:
        """
        Splits the dataset randomly according to the proportions in `lengths`.

        :param lengths: Sequence of proportions (should sum to 1.0)
        :param generator: Random number generator for reproducibility
        :param split_by: How to split the dataset ('image', 'sequence', 'season', or 'default')
                         If 'default', uses the value from self.split_dataset_by
        :return: List of Subsets containing the split data
        """
        # Use default split method if not specified
        if split_by == 'default':
            split_by = self.split_dataset_by
            
        if self.verbose:
            print(f"Random split by: {split_by}")

        # Dispatch to appropriate split method
        if split_by == 'sequence':
            # When splitting by sequence and get_images_by_sequence=True,
            # only include sequences that have images after filtering
            return self._random_split_by_sequence(lengths, generator)
        elif split_by == 'season':
            # Splitting by season
            return self._random_split_by_season(lengths, generator)
        else:
            # Default to splitting by individual image
            if self.get_images_by_sequence and self.verbose:
                print("Warning: Splitting by image when get_images_by_sequence=True may produce unexpected results")
            
            # Calculate split lengths
            split_lengths = self._calculate_split_lengths(lengths)
            
            # Get all images
            all_indices = list(range(len(self)))
            
            # Shuffle indices correctly using torch.randperm with the generator
            if generator is not None:
                # Use torch's randperm with the generator instead of shuffle
                shuffled_indices = torch.randperm(len(all_indices), generator=generator).tolist()
                all_indices = [all_indices[i] for i in shuffled_indices]
            else:
                random.shuffle(all_indices)
            
            # Split indices according to proportions
            result = []
            offset = 0
            for length in split_lengths:
                result.append(Subset(self, all_indices[offset:offset + length]))
                offset += length
                
            return result

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
                return_indices.extend(
                    self.seq_indices_to_total_indices(seq_obj))
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
            return_indices.extend(
                self.seq_indices_to_total_indices(seq_object))
        return Subset(self, return_indices)

    def images_as_tensor(self, indices: List[int]) -> torch.Tensor:
        """
        Given a list of dataset indices, returns the images as a Torch Tensor

        :param indices: List[int]
        :return: torch Tensor
        """
        images = np.array([self.get_image_from_idx(idx).image()
                          for idx in indices])
        return torch.Tensor(images)

    def labels_as_tensor(self, indices: List[int], label: str) -> torch.Tensor:
        """
        Given a list of dataset indices, returns the specified labels as a Torch Tensor

        :param indices: List[int]
        :param label: str, denoting which label to retrieve
        :return: torch Tensor
        """
        images = [self.get_image_from_idx(
            idx).value_from_string(label) for idx in indices]
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
        # For consistency with __len__, if get_images_by_sequence is True,
        # we should return the total number of sequences regardless of images
        if self.get_images_by_sequence:
            return self.get_number_of_sequences()
        else:
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
        Process a single metadata JSON file.
        
        :param filepath: Path to the metadata JSON file.
        :return: None
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Process each sequence in the metadata file
            for sequence_str, metadata in sorted(data.items()):
                self._read_one_seq_from_metadata(sequence_str, metadata)
                
        except Exception as e:
            print(f"Error processing metadata file {filepath}: {e}")
            raise

    def process_metadata_files(self, filepaths: List[str]):
        """
        Reads and processes JSON metadata file's information into dataset.

        :param filepath: path to metadata file
        :return: metadata JSON object
        """
        length_filepaths = len(filepaths)
        for i in range(length_filepaths):
            filepath = filepaths[i]
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.number_of_sequences += len(data)
            for sequence_str, metadata in sorted(data.items()):
                sequence_str_unique = f'{i}_{sequence_str}'
                self._read_one_seq_from_metadata(sequence_str_unique, metadata)
            print("self.sequences", len(self.sequences))

    def get_common_sequences_from_files(self, metadata_jsons: List[str], metadata_dirs: List[str], image_dirs: List[str]):
        """
        Find common sequences across metadata and image directories.

        :param metadata_jsons: List of paths to metadata JSON files
        :param metadata_dirs: List of paths to metadata directories
        :param image_dirs: List of paths to image directories
        :return: Set of sequences common to all provided data sources
        """
        print("\n" + "="*50)
        print("Finding common sequences across data sources")
        print(f"Checking {len(metadata_jsons)} metadata JSONs, {len(metadata_dirs)} metadata dirs, {len(image_dirs)} image dirs")
        
        # Print the actual paths for easier debugging
        if metadata_jsons:
            print(f"Metadata JSONs: {metadata_jsons}")
        if metadata_dirs:
            print(f"Metadata directories: {metadata_dirs}")
        if image_dirs:
            print(f"Image directories: {image_dirs}")
        
        # Check consistency of data indices
        try:
            self.assert_consistency_data_in_same_index(
                metadata_jsons=metadata_jsons, metadata_dirs=metadata_dirs, image_dirs=image_dirs)
            print("Data consistency check passed")
        except AssertionError as e:
            print(f"WARNING: Data consistency check failed: {str(e)}")
            print("Still trying to find common sequences...")
        
        # Get sequences from metadata JSON files
        common_sequences_from_metadata_jsons = self.get_common_sequences_from_metadata_files(
            metadata_jsons)
        print(f"Found {len(common_sequences_from_metadata_jsons)} sequences from metadata JSON files")
        if common_sequences_from_metadata_jsons:
            print(f"Sample sequences from JSON: {list(common_sequences_from_metadata_jsons)[:5]}")
        
        # Get sequences from metadata directories
        common_sequences_from_metadata_dirs = self.get_common_sequences_from_metadata_dirs(
            metadata_dirs)
        print(f"Found {len(common_sequences_from_metadata_dirs)} sequences from metadata directories")
        if common_sequences_from_metadata_dirs:
            print(f"Sample sequences from metadata dirs: {list(common_sequences_from_metadata_dirs)[:5]}")
        
        # Get sequences from image directories
        common_sequences_from_image_dirs = self.get_common_sequences_from_image_dirs(
            image_dirs)
        print(f"Found {len(common_sequences_from_image_dirs)} sequences from image directories")
        if common_sequences_from_image_dirs:
            print(f"Sample sequences from image dirs: {list(common_sequences_from_image_dirs)[:5]}")
        
        # Find sequences common to all data sources, with graceful fallbacks
        common_sequences = set()
        
        # Try all combinations of data sources
        if common_sequences_from_metadata_jsons and common_sequences_from_metadata_dirs and common_sequences_from_image_dirs:
            common_sequences = common_sequences_from_metadata_jsons.intersection(
                common_sequences_from_metadata_dirs).intersection(common_sequences_from_image_dirs)
            print(f"Using sequences common to all three sources: {len(common_sequences)}")
        elif common_sequences_from_metadata_jsons and common_sequences_from_metadata_dirs:
            common_sequences = common_sequences_from_metadata_jsons.intersection(
                common_sequences_from_metadata_dirs)
            print(f"Using sequences common to metadata JSONs and dirs: {len(common_sequences)}")
        elif common_sequences_from_metadata_jsons and common_sequences_from_image_dirs:
            common_sequences = common_sequences_from_metadata_jsons.intersection(
                common_sequences_from_image_dirs)
            print(f"Using sequences common to metadata JSONs and image dirs: {len(common_sequences)}")
        elif common_sequences_from_metadata_dirs and common_sequences_from_image_dirs:
            common_sequences = common_sequences_from_metadata_dirs.intersection(
                common_sequences_from_image_dirs)
            print(f"Using sequences common to metadata dirs and image dirs: {len(common_sequences)}")
        # If any single source has sequences, use that
        elif common_sequences_from_metadata_jsons:
            common_sequences = common_sequences_from_metadata_jsons
            print(f"Falling back to sequences from metadata JSONs only")
        elif common_sequences_from_metadata_dirs:
            common_sequences = common_sequences_from_metadata_dirs
            print(f"Falling back to sequences from metadata dirs only")
        elif common_sequences_from_image_dirs:
            common_sequences = common_sequences_from_image_dirs
            print(f"Falling back to sequences from image dirs only")
        
        if not common_sequences:
            print("WARNING: No common sequences found across data sources!")
        else:
            print(f"Final set: {len(common_sequences)} common sequences")
            print(f"Sample sequences: {list(common_sequences)[:10]}")
        
        print("="*50 + "\n")
        return common_sequences

    def assert_consistency_data_in_same_index(self, metadata_jsons: List[str], metadata_dirs: List[str], image_dirs: List[str]):
        length_metadata_jsons = len(metadata_jsons)
        for i in range(length_metadata_jsons):
            # Create a new metadata_jsons_checking array, it contains only the data of this index
            metadata_jsons_checking = [metadata_jsons[i]]
            metadata_dirs_checking = [metadata_dirs[i]]
            image_dirs_checking = [image_dirs[i]]
            common_sequences_from_metadata_jsons = self.get_common_sequences_from_metadata_files(
                metadata_jsons_checking)
            common_sequences_from_metadata_dirs = self.get_common_sequences_from_metadata_dirs(
                metadata_dirs_checking)
            common_sequences_from_image_dirs = self.get_common_sequences_from_image_dirs(
                image_dirs_checking)
            # Make sure common_sequences_from_image_dirs, common_sequences_from_metadata_dirs, common_sequences_from_metadata_jsons are the same
            assert common_sequences_from_metadata_jsons == common_sequences_from_metadata_dirs
            assert common_sequences_from_metadata_dirs == common_sequences_from_image_dirs
            assert common_sequences_from_image_dirs == common_sequences_from_metadata_jsons

        pass

    def get_common_sequences_from_metadata_files(self, metadata_jsons: List[str]):
        """
        Reads and processes JSON metadata file's information into dataset.

        :param filepath: path to metadata file
        :return: metadata JSON object
        """
        datas_sequences_information = {}
        for filepath in metadata_jsons:
            datas_sequences_information[filepath] = []
            with open(filepath, 'r') as f:
                data = json.load(f)
            for sequence_str, metadata in sorted(data.items()):
                datas_sequences_information[filepath].append(sequence_str)
        # Get the common sequences for all keys in datas_sequences_information
        common_sequences = set(datas_sequences_information[metadata_jsons[0]])
        for key in datas_sequences_information.keys():
            common_sequences = common_sequences.intersection(
                datas_sequences_information[key])
        return common_sequences

    def get_common_sequences_from_metadata_dirs(self, metadata_dirs: List[str]):
        data_sequences_information = {}
        for metadata_dir in metadata_dirs:
            data_sequences_information[metadata_dir] = []
            for root, dirs, files in os.walk(metadata_dir, topdown=True):
                for file in sorted(files):
                    file_sequence = get_seq_str_from_track_filename(file)
                    data_sequences_information[metadata_dir].append(
                        file_sequence)
        common_sequences = set(data_sequences_information[metadata_dirs[0]])
        for key in data_sequences_information.keys():
            common_sequences = common_sequences.intersection(
                data_sequences_information[key])
        return common_sequences

    def get_common_sequences_from_image_dirs(self, image_dirs: List[str]):
        """
        Gets the sequences common to all dirs in image_dirs.

        :param image_dirs: List of image directory paths
        :return: Set of sequences common to all dirs
        """
        if not image_dirs:
            print("No image directories provided, returning empty set")
            return set()

        # Initialize with directories from first image dir
        print(f"Scanning image directories for sequences...")
        common_sequences = set()
        first_dir_scanned = False
        
        # Process each image directory
        for idx, image_dir in enumerate(image_dirs):
            if not os.path.isdir(image_dir):
                print(f"WARNING: {image_dir} is not a valid directory, skipping")
                continue
            
            try:
                # Get all subdirectories (which should be sequence folders)
                sequence_dirs = set([d for d in os.listdir(image_dir) 
                                   if os.path.isdir(os.path.join(image_dir, d))])
                
                # Check if we found any sequences
                if not sequence_dirs:
                    print(f"WARNING: No sequence directories found in {image_dir}")
                    continue
                
                print(f"  Found {len(sequence_dirs)} sequence directories in {image_dir}")
                
                # First directory sets the initial sequence set
                if not first_dir_scanned:
                    common_sequences = sequence_dirs
                    first_dir_scanned = True
                    print(f"  First directory: Initial sequences = {len(common_sequences)}")
                else:
                    # Intersect with sequences from this directory
                    prev_count = len(common_sequences)
                    common_sequences.intersection_update(sequence_dirs)
                    print(f"  After intersection: {prev_count} → {len(common_sequences)} sequences")
            
            except Exception as e:
                print(f"Error processing image directory {image_dir}: {str(e)}")
        
        # Show some sample sequences
        if common_sequences:
            sample = list(common_sequences)[:5]
            print(f"Found {len(common_sequences)} common sequences across image directories")
            print(f"Sample sequences: {sample}")
            
            # Check if sample sequences have image files
            for seq in sample[:2]:  # Check just a couple to avoid too much output
                for image_dir in image_dirs:
                    seq_dir = os.path.join(image_dir, seq)
                    if os.path.isdir(seq_dir):
                        try:
                            files = [f for f in os.listdir(seq_dir) if is_image_file(f)]
                            print(f"  Sequence {seq} in {image_dir} has {len(files)} image files")
                        except Exception as e:
                            print(f"  Error listing files in {seq_dir}: {str(e)}")
        else:
            print("No common sequences found across image directories!")
        
        return common_sequences

    def get_seq_ids_from_season(self, season: int) -> List[str]:
        """
        Given a start season, give the sequence ID strings of all sequences that start in that season.

        :param season: the start season as a string
        :return: a list of the sequence IDs starting in that season
        """
        if season not in self.season_to_sequence_nums:
            raise ValueError(
                f'Season \'{season}\' is not within the list of start seasons.')
        return self.season_to_sequence_nums[season]

    def total_image_idx_to_sequence_idx(self, total_idx: int) -> int:
        """
        Given a total dataset image index, returns that image's index in its respective sequence. e.g. an image that is
        the 500th in the total dataset may be the 5th image in its sequence.

        :param total_idx: the total dataset image index
        :return: the inner-sequence image index.
        """
        sequence = self._image_idx_to_sequence[total_idx]
        start_idx = self._seq_str_to_first_total_idx[sequence.get_sequence_str(
        )]
        if total_idx >= self.number_of_images:
            raise ValueError(
                f'Image {total_idx} is beyond the number of images in the dataset.')
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
            raise ValueError(
                f'Image {seq_idx} is beyond the number of images in the dataset.')
        return self._seq_str_to_first_total_idx[seq_str] + seq_idx

    def seq_indices_to_total_indices(self, seq_obj: DigitalTyphoonSequence) -> List[int]:
        """
        Given a sequence, return a list of the total dataset indices of the sequence's images.

        :param seq_obj: the DigitalTyphoonSequence object to produce the list from
        :return: the List of total dataset indices
        """
        seq_str = seq_obj.get_sequence_str()
        num_images = seq_obj.get_num_images()
        
        # Skip excessive debugging output
        if self.verbose and num_images > 0:
            print(f"Sequence {seq_str} has {num_images} images")
        
        # Return empty list if no images or sequence not in mapping
        if num_images == 0:
            return []
            
        if seq_str not in self._seq_str_to_first_total_idx:
            if self.verbose:
                print(f"WARNING: Sequence {seq_str} not found in mapping")
            return []
        
        # Return list of indices for this sequence
        return [i + self._seq_str_to_first_total_idx[seq_str] for i in range(num_images)]

    def get_image_from_idx(self, idx) -> DigitalTyphoonImage:
        """
        Given a dataset image idx, returns the image object from that index.

        :param idx: int, the total dataset image idx
        :return: DigitalTyphoonImage object for that image
        """
        try:
            # For valid indices within range, use direct access (original behavior)
            if 0 <= idx < self.number_of_images and idx in self._image_idx_to_sequence:
                sequence_str = self._find_sequence_str_from_image_index(idx)
                if sequence_str:
                    sequence = self._get_seq_from_seq_str(sequence_str)
                    seq_idx = self.total_image_idx_to_sequence_idx(idx)
                    if 0 <= seq_idx < sequence.get_num_images():
                        return sequence.get_image_at_idx(seq_idx)
                
            # Fall through to error handling only if something is wrong
            if self.verbose:
                print(f"Warning: Could not find valid image at index {idx}")
            
            # Return a placeholder image only if needed
            # Important: Pass empty array as track_data, not empty string
            return DigitalTyphoonImage("", np.array([]), 
                                      sequence_id=None, 
                                      spectrum=self.spectrum, 
                                      verbose=self.verbose,
                                      image_filepaths=self.image_filepaths)
        except (IndexError, ValueError, KeyError, AttributeError) as e:
            if self.verbose:
                print(f"Error accessing image at index {idx}: {str(e)}")
            # Important: Pass empty array as track_data, not empty string
            return DigitalTyphoonImage("", np.array([]), 
                                      sequence_id=None,
                                      spectrum=self.spectrum,
                                      verbose=self.verbose,
                                      image_filepaths=self.image_filepaths)

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
                sequence_obj.process_seq_img_dir_into_sequence(root+dir_name, load_into_mem,ignore_list=self.ignore_list,filter_func=self.filter_func,spectrum=self.spectrum)
                if self.image_dir =='test_data_files/image/':
                    print("sequence_obj.get_num_images()", sequence_obj.get_num_images())
                    print("self.number_of_images", self.number_of_images)
                self.number_of_images += sequence_obj.get_num_images()

        for sequence in self.sequences:
            if sequence.get_num_images() > 0:
                self.number_of_nonempty_sequences += 1

            if not sequence.num_images_match_num_expected():
                if self.verbose:
                    warnings.warn(f'Sequence {sequence.sequence_str} has only {sequence.get_num_images()} when '
                                  f'it should have {sequence.num_original_images}. If this is intended, ignore this warning.')
            

    def _populate_track_data_into_sequences(self, metadata_dir: str, common_sequences: List[str] = None) -> None:
        """
        Traverses the track data files and populates each into their respective seq_str objects

        :param metadata_dir: path to directory containing track data files
        :param common_sequences: optional list of sequences to filter by
        :return: None
        """
        NEED_FILTER_COMMON_SEQUENCES = common_sequences is not None
        
        if self.verbose:
            print(f"\nPopulating track data from {metadata_dir}")
            if NEED_FILTER_COMMON_SEQUENCES:
                print(f"Filtering to common sequences: {common_sequences[:5]}... (total {len(common_sequences)})")
        
        files_processed = 0
        sequences_with_track_data = 0
        
        for root, dirs, files in os.walk(metadata_dir, topdown=True):
            if self.verbose:
                print(f"Found {len(files)} files in {root}")
                
            for file in sorted(files):
                file_valid = True
                file_sequence = get_seq_str_from_track_filename(file)
                
                if NEED_FILTER_COMMON_SEQUENCES:
                    file_valid = file_sequence in common_sequences
                
                if self.sequence_exists(file_sequence) and file_valid:
                    if self.verbose and files_processed < 5:
                        print(f"Processing track file: {file} for sequence {file_sequence}")
                        
                    full_path = os.path.join(root, file)
                    
                    # Set the track path
                    sequence = self._get_seq_from_seq_str(file_sequence)
                    sequence.set_track_path(full_path)
                    
                    # Read the track data into the sequence
                    try:
                        self._read_in_track_file_to_sequence(file_sequence, full_path)
                        sequences_with_track_data += 1
                        
                        # Verify after reading
                        if self.verbose and files_processed < 3:
                            image_count = len(sequence.images)
                            images_with_track = sum(1 for img in sequence.images if img.track_data is not None and len(img.track_data) > 0)
                            print(f"Sequence {file_sequence}: {images_with_track}/{image_count} images have track data")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Error reading track file {full_path}: {str(e)}")
                            
                    files_processed += 1
                elif self.verbose and files_processed < 10:
                    if not self.sequence_exists(file_sequence):
                        print(f"Skipping track file {file}: sequence {file_sequence} does not exist in dataset")
                    elif not file_valid:
                        print(f"Skipping track file {file}: sequence {file_sequence} not in common sequences")
        
        if self.verbose:
            print(f"Processed {files_processed} track files for {sequences_with_track_data} sequences")
            
            # Verify track data assignment
            total_images = 0
            images_with_track = 0
            for sequence in self.sequences:
                seq_images = len(sequence.images)
                seq_with_track = sum(1 for img in sequence.images if img.track_data is not None and len(img.track_data) > 0)
                total_images += seq_images
                images_with_track += seq_with_track
                
                if seq_images > 0 and seq_with_track == 0 and self.verbose:
                    print(f"WARNING: Sequence {sequence.sequence_str} has {seq_images} images but none have track data")
            
            if total_images > 0:
                print(f"Overall: {images_with_track}/{total_images} images ({images_with_track/total_images*100:.1f}%) have track data")

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
        num_images = metadata_json['images'] if 'images' in metadata_json.keys(
        ) else metadata_json['frames']
        metadata_json['images'] = num_images
        self.sequences.append(DigitalTyphoonSequence(sequence_str,
                                                     seq_start_date.year,
                                                     num_images,
                                                     transform_func=self.transform_func,
                                                     spectrum=self.spectrum,
                                                     verbose=self.verbose))
        self._sequence_str_to_seq_idx[sequence_str] = len(self.sequences) - 1

        does_metadata_has_season_key = 'season' not in metadata_json.keys()
        if does_metadata_has_season_key:
            metadata_json.__setitem__('season', metadata_json['year'])

        if metadata_json['season'] not in self.season_to_sequence_nums:
            self.season_to_sequence_nums[metadata_json['season']] = []
        self.season_to_sequence_nums[metadata_json['season']].append(
            sequence_str)
        self.number_of_original_images += metadata_json['images']

    def _assign_all_images_a_dataset_idx(self):
        """
        Assigns every relevant image (i.e. images that passed our filter function and are included in our subset,
        not the number of original images stated in the metadata.json) an index within the total dataset.
        :return: None
        """
        dataset_idx_iter = 0
        empty_sequences = 0
        sequences_with_images = 0
        total_images_found = 0
        
        if self.verbose:
            print("Assigning indices to sequence images...")
        
        for seq_idx, sequence in enumerate(self.sequences):
            seq_str = sequence.get_sequence_str()
            num_images = sequence.get_num_images()
            
            if num_images == 0:
                empty_sequences += 1
                if self.verbose:
                    print(f"Warning: Sequence {seq_str} has 0 images")
                # Still assign a starting index, even if empty
                self._seq_str_to_first_total_idx[seq_str] = dataset_idx_iter
            else:
                sequences_with_images += 1
                total_images_found += num_images
                self._seq_str_to_first_total_idx[seq_str] = dataset_idx_iter
                
                # Assign each image in the sequence to the sequence object
                for _ in range(num_images):
                    self._image_idx_to_sequence[dataset_idx_iter] = sequence
                    dataset_idx_iter += 1
        
        # Summary statistics - keep this but make it conditional on verbose
        if self.verbose:
            print(f"Index assignment complete: {sequences_with_images} sequences with images, {empty_sequences} empty sequences")
            print(f"Total images found: {total_images_found}")
        
        self.number_of_images = total_images_found

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
                    raise ValueError(
                        f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    math.floor(dataset_length * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = dataset_length - \
                sum(subset_lengths)  # type: ignore[arg-type]
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
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!")

        return lengths

    def _random_split_by_season(self, lengths: Sequence[Union[int, float]],
                                generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Randomly splits the dataset s.t. each bucket has close to the requested number of indices in each split.
        Images (indices) from a given season are not split across different buckets. Indices within a season
        are given contiguously in the returned Subset.

        As a season is treated as an atomic unit, achieving the exact split requested may not be possible. An
        approximation where each bucket is guaranteed to have at least one item is used. Randomization is otherwise
        preserved.

        If "get_images_by_sequence" was set to true, then random_split returns buckets containing indices referring to
        entire sequences. As an atomic unit is a sequence, this function adds no extra functionality over
        the default random_split function.

        Only non-empty seasons are returned in the split.

        :param lengths: Lengths or fractions of splits to be produced
        :param generator: Generator used for the random permutation.
        :return: List of Subset objects
        """
        lengths = self._calculate_split_lengths(lengths)
        return_indices_sorted = [[length, i, []]
                                 for i, length in enumerate(lengths)]
        return_indices_sorted.sort(key=lambda x: x[0])

        # make a list of all non-empty seasons
        non_empty_season_indices = []
        for idx, (season, val) in enumerate(self.season_to_sequence_nums.items()):
            nonempty = False
            for seq_id in val:
                if self._get_seq_from_seq_str(seq_id).get_num_images() > 0:
                    nonempty = True
                    break
            if nonempty:
                non_empty_season_indices.append(idx)
        non_empty_season_indices = [non_empty_season_indices[idx] for idx in randperm(
            len(non_empty_season_indices), generator=generator)]
        randomized_season_list = [list(self.season_to_sequence_nums.keys())[
            i] for i in non_empty_season_indices]

        num_buckets = len(return_indices_sorted)
        bucket_counter = 0
        season_iter = 0
        while season_iter < len(randomized_season_list):
            print("season_iter", season_iter)
            if len(return_indices_sorted[bucket_counter][2]) < return_indices_sorted[bucket_counter][0]:
                for seq in self.season_to_sequence_nums[randomized_season_list[season_iter]]:
                    sequence_obj = self._get_seq_from_seq_str(seq)
                    if self.get_images_by_sequence:
                        if sequence_obj.get_num_images() > 0:  # Only append if the sequence has images
                            return_indices_sorted[bucket_counter][2].append(
                                self._sequence_str_to_seq_idx[seq])
                    else:
                        return_indices_sorted[bucket_counter][2] \
                            .extend(self.seq_indices_to_total_indices(self._get_seq_from_seq_str(seq)))
                
                # Increment season_iter after processing the current season
                season_iter += 1
            else:
                # If current bucket is full, move to next bucket
                bucket_counter += 1
                if bucket_counter == num_buckets:
                    bucket_counter = 0

        return_indices_sorted.sort(key=lambda x: x[1])
        return [Subset(self, bucket_indices) for _, _, bucket_indices in return_indices_sorted]

    def _random_split_by_sequence(self, lengths: Sequence[Union[int, float]],
                                  generator: Optional[Generator] = default_generator) -> List[Subset]:
        """
        Splits the dataset by sequence according to the specified proportions.
        
        When get_images_by_sequence=True, only non-empty sequences are included in the split.
        This ensures that sequences without any valid images after filtering are excluded.
        
        :param lengths: Sequence of proportions that should sum to 1.0
        :param generator: Random number generator for reproducibility (not used in simplified version)
        :return: List of Subsets containing the split data
        """
        # Calculate actual split lengths
        split_lengths = self._calculate_split_lengths(lengths)
        
        # Create empty buckets for distribution
        buckets = [(i, l, []) for i, l in enumerate(split_lengths)]
        
        # Get all sequence indices
        if self.get_images_by_sequence:
            # When get_images_by_sequence=True, use only sequences that have images after filtering
            seq_indices = []
            for i, seq in enumerate(self.sequences):
                # Only include sequences that have at least one image after filtering
                if seq.has_images():
                    seq_indices.append(i)
        else:
            # Otherwise use all sequences
            seq_indices = list(range(len(self.sequences)))
        
        if not seq_indices:
            # No sequences available, return empty subsets
            if self.verbose:
                print("Warning: No sequences available for split")
            return [Subset(self, []) for _ in split_lengths]
        
        # Deterministically distribute sequences to buckets
        # No need for complex shuffling with generators
        for i, seq_idx in enumerate(seq_indices):
            # Simple round-robin distribution
            bucket_idx = i % len(buckets)
            _, current_length, current_bucket = buckets[bucket_idx]
            
            # Add sequence to the bucket
            if self.get_images_by_sequence:
                # When get_images_by_sequence=True, add the sequence index directly
                current_bucket.append(seq_idx)
            else:
                # Otherwise, add all image indices from this sequence
                sequence_obj = self.sequences[seq_idx]
                current_bucket.extend(self.seq_indices_to_total_indices(sequence_obj))
            
            # Update the bucket
            buckets[bucket_idx] = (bucket_idx, current_length, current_bucket)
        
        # Sort buckets by original index
        buckets.sort(key=lambda x: x[0])
        
        # Create and return subsets
        return [Subset(self, indices) for _, _, indices in buckets]

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
        if idx in self._image_idx_to_sequence:
            return self._image_idx_to_sequence[idx].get_sequence_str()
        
        # Only if the image index is not found, show a warning
        if self.verbose:
            print(f"Warning: No sequence mapping found for image index {idx}")
        return ""

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
            label_ray = np.array([image.value_from_string(label)
                                 for label in label_strs])
            return label_ray
        else:
            label = image.value_from_string(label_strs)
            return label

    def _delete_all_sequences(self):
        """
        Clears all the sequences and other datastructures containing data.
        :return: None
        """
        self.sequences: List[DigitalTyphoonSequence] = list(
        )  # List of seq_str objects
        # Sequence ID to idx in sequences array
        self._sequence_str_to_seq_idx: Dict[str, int] = {}
        # Image idx to what seq_str it belongs to
        self._image_idx_to_sequence: Dict[int, DigitalTyphoonSequence] = {}
        # Sequence string to the first total idx belonging to
        self._seq_str_to_first_total_idx: Dict[str, int] = {}
        #  that seq_str
        self.season_to_sequence_nums: OrderedDict[int, List[str]] = OrderedDict(
        )

        self.number_of_sequences = 0
        self.number_of_original_images = 0
        self.number_of_images = 0

    def is_load_data_from_multi_dirs(self) -> bool:
        """
        Check if the data is loaded from multiple directories.
        :return: True if the data is loaded from multiple directories, False otherwise.
        """
        return self.image_dirs is not None and len(self.image_dirs) > 0
