import os
import warnings
from datetime import datetime
from typing import Callable, Union

from pathlib import Path
import numpy as np
from typing import List, Dict
import pandas as pd

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from pyphoon2.DigitalTyphoonUtils import parse_image_filename, is_image_file, TRACK_COLS, parse_common_image_filename


class DigitalTyphoonSequence:

    def __init__(self, seq_str: str, start_season: int, num_images: int, transform_func=None,
                 spectrum='Infrared', verbose=False, load_imgs_into_mem=False):
        """
        Class representing one typhoon sequence from the DigitalTyphoon dataset

        :param seq_str: str, sequence ID as a string
        :param start_season: int, the season in which the typhoon starts in
        :param num_images: int, number of images in the sequence
        :param transform_func: this function will be called on each image before saving it/returning it.
                            It should take and return a np array
        :param spectrum: str, specifies the spectrum of the images (e.g., 'Infrared')
        :param verbose: bool, if True, additional information and warnings will be printed during processing
        :param load_imgs_into_mem: bool, if True, images will be loaded into memory when creating image objects
        """
        self.verbose = verbose
        self.load_imgs_into_mem = load_imgs_into_mem

        self.sequence_str = seq_str  # sequence ID string
        self.season = start_season
        self.num_track_entries = 0
        self.num_original_images = num_images
        self.track_data = np.array([])
        self.img_root = None  # root path to directory containing image files
        self.track_path = None  # path to track file data
        self.transform_func = transform_func
        self.spectrum = spectrum

        # Ordered list containing image objects with metadata
        self.images: List[DigitalTyphoonImage] = list()

        # Dictionary mapping datetime to Image objects
        self.datetime_to_image: Dict[datetime, DigitalTyphoonImage] = {}

        # TODO: add synthetic image value into metadata and consistency check within dataset loader

    def get_sequence_str(self) -> str:
        """
        Returns the sequence ID as a string

        :return: string sequence ID
        """
        return self.sequence_str

    def process_seq_img_dir_into_sequence(self, directory_path: str,
                                          load_imgs_into_mem=False,
                                          ignore_list=None,
                                          spectrum=None,
                                          filter_func: Callable[[DigitalTyphoonImage], bool] = None) -> None:
        """
        Processes the image directory and stores the images found within the sequence.

        :param directory_path: Path of directory containing images of this sequence
        :param load_imgs_into_mem: Boolean for whether the image data should be stored in mem
        :param ignore_list: Set of image filenames to ignore
        :param spectrum: string, name of which spectrum the image is from, IR is default
        :param filter_func: function that takes in a DigitalTyphoonImage and returns bool, whether that img should be kept
        :return: None
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(
                f"{directory_path} is not a valid directory.")

        if ignore_list is None:
            ignore_list = set([])

        if spectrum is None:
            spectrum = self.spectrum
            
        # Ensure filter_func is callable
        if filter_func is None or not callable(filter_func):
            if self.verbose:
                print(f"Warning: filter_func is not callable for sequence {self.get_sequence_str()}, using default filter (accept all images)")
            # Default filter function that accepts all images
            filter_func = lambda img: True

        # Ensure we have an absolute directory path
        abs_directory_path = os.path.abspath(directory_path)
        self.set_images_root_path(abs_directory_path)

        # Debug print
        if self.verbose:
            print(f"Processing images from directory: {abs_directory_path}")
            print(f"For sequence: {self.get_sequence_str()}")
            print(f"Using filter_func: {filter_func}")

        # Clear existing images to prevent duplicates
        # Important: We need to clear these before reprocessing
        old_image_count = len(self.images)
        if old_image_count > 0 and self.verbose:
            print(f"Clearing {old_image_count} existing images before reprocessing")
        self.images = []
        
        # Save track data from datetime_to_image before clearing
        track_data_mapping = {}
        for dt, img in self.datetime_to_image.items():
            if hasattr(img, 'track_data') and img.track_data is not None and len(img.track_data) > 0:
                track_data_mapping[dt] = img.track_data
        
        # Keep the original datetime_to_image for track data reference
        # but clear image associations so we can rebuild
        # DO NOT completely reset datetime_to_image as it may contain track data
        # Only remove actual image objects but keep the track data
        for dt in list(self.datetime_to_image.keys()):
            img = self.datetime_to_image[dt]
            if img is not None and hasattr(img, 'track_data') and img.track_data is not None and len(img.track_data) > 0:
                # Keep track data by creating a placeholder with just track data
                track_data = img.track_data
                # Replace image with track-data-only placeholder
                self.datetime_to_image[dt] = DigitalTyphoonImage(
                    None, track_data, self.get_sequence_str(), 
                    transform_func=None, spectrum=self.spectrum,
                    verbose=False)
            else:
                # Remove entries with no track data
                del self.datetime_to_image[dt]
        
        files_to_parse = []
        for root, dirs, files in os.walk(abs_directory_path, topdown=True):
            for file in files:
                if is_image_file(file):
                    path_of_file = os.path.join(root, file)
                    files_to_parse.append(path_of_file)

        if self.verbose:
            print(f"Found {len(files_to_parse)} files to parse in {abs_directory_path}")
            print(f"Have track data for {len(track_data_mapping)} datetimes")

        # Using sorted makes it deterministic
        for file_path in sorted(files_to_parse):
            try:
                # Extract metadata from file, important is sequence ID and datetime of image
                file_basename = os.path.basename(file_path)
                
                # Skip if in ignore list
                if file_basename in ignore_list:
                    if self.verbose:
                        print(f"Skipping ignored file: {file_basename}")
                    continue
                    
                # Parse filename for metadata    
                metadata = parse_image_filename(file_basename)
                file_sequence, file_datetime, _ = metadata

                # Check if this sequence and ID matches
                if file_sequence == self.get_sequence_str():
                    # Ensure we have an absolute file path
                    abs_file_path = os.path.abspath(file_path)
                    
                    # Check if we have track data for this datetime
                    track_data = None
                    if file_datetime in self.datetime_to_image:
                        existing_img = self.datetime_to_image[file_datetime]
                        if existing_img and hasattr(existing_img, 'track_data') and existing_img.track_data is not None and len(existing_img.track_data) > 0:
                            track_data = existing_img.track_data
                            if self.verbose:
                                print(f"Found track data for {file_basename} at datetime {file_datetime} with year {int(track_data[TRACK_COLS.YEAR.value]) if TRACK_COLS.YEAR.value < len(track_data) else 'unknown'}")
                    
                    # Create the image object with track data if available
                    try:
                        # Check if file exists first to avoid constructor error
                        if not os.path.exists(abs_file_path):
                            if self.verbose:
                                print(f"Warning: Image file does not exist: {abs_file_path}")
                        image = DigitalTyphoonImage(
                            abs_file_path, 
                            track_data=track_data,  # Pass track data if available
                            sequence_id=self.get_sequence_str(),
                            transform_func=self.transform_func, 
                            spectrum=self.spectrum,
                            load_imgs_into_mem=load_imgs_into_mem,
                            verbose=self.verbose,
                        )
                        
                        # Apply filter
                        try:
                            # Only add if the filter passes
                            if filter_func(image):
                                if self.verbose:
                                    print(f"  Image passed filter: {abs_file_path}")
                                    if track_data is not None:
                                        print(f"    Has track data with year={image.year()}")
                                    else:
                                        print(f"    No track data available")
                                self.images.append(image)
                                self.datetime_to_image[file_datetime] = image
                            else:
                                if self.verbose:
                                    print(f"  Image filtered out: {abs_file_path}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Error applying filter to {abs_file_path}: {e}")
                                import traceback
                                traceback.print_exc()
                    except Exception as e:
                        if self.verbose:
                            print(f"Error creating image object for {abs_file_path}: {e}")
                            import traceback
                            traceback.print_exc()

            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Error processing file {file_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # Make sure image dataset is ordered w.r.t. time
        try:
            self.images.sort(key=lambda x: x.get_datetime())
        except Exception as e:
            if self.verbose:
                print(f"Error sorting images: {str(e)}")

        # Print warning if there is an inconsistency between number of found images and expected number
        if self.verbose:
            print(f"Final image count for sequence {self.get_sequence_str()}: {len(self.images)}")
            
            # Check track data counts
            images_with_track = sum(1 for img in self.images if len(img.track_data) > 0)
            print(f"Images with track data: {images_with_track} of {len(self.images)}")
            
            if not self.num_images_match_num_expected():
                warnings.warn(f'The number of images ({len(self.images)}) does not match the '
                              f'number of expected images ({self.num_original_images}) from metadata. If this is expected, ignore this warning.')

            if self.num_track_entries > 0 and len(self.images) < self.num_track_entries:
                warnings.warn(
                    f'Only {len(self.images)} of {self.num_track_entries} track entries have images.')

    def process_seq_img_dirs_into_sequence(self, directory_paths: List[str],
                                           common_image_names: List[str],
                                           load_imgs_into_mem=False,
                                           ignore_list=None,
                                           filter_func: Callable[[DigitalTyphoonImage], bool] = lambda img: True,
                                           spectrum=None) -> None:
        """
        Process images from multiple directories into a sequence, combining them as multiple channels.
        
        :param directory_paths: List of paths to the directories containing images
        :param common_image_names: List of common image names across directories
        :param load_imgs_into_mem: Bool representing if images should be loaded into memory
        :param ignore_list: list of image filenames to ignore
        :param filter_func: function that accepts an image and returns True/False if it should be included
        :param spectrum: string representing what spectrum the image lies in
        :return: None
        """
        print(f"\nProcessing sequence: {self.get_sequence_str()}")
        print(f"  - Directory paths: {directory_paths}")
        print(f"  - Common image names: {len(common_image_names)} images")
        if common_image_names:
            print(f"  - First few image names: {common_image_names[:3]}")
        if filter_func is None:
            filter_func = lambda img: True
        # Input validation
        valid_dirs = []
        for directory_path in directory_paths:
            if os.path.isdir(directory_path):
                valid_dirs.append(directory_path)
            else:
                print(f"  - Warning: {directory_path} is not a valid directory, skipping.")
        
        if not valid_dirs:
            print(f"  - Error: No valid directories found for sequence {self.get_sequence_str()}")
            return
        
        directory_paths = valid_dirs
        
        if not common_image_names:
            print(f"  - Error: No common image names provided for sequence {self.get_sequence_str()}")
            return
        
        if ignore_list is None:
            ignore_list = set([])

        if spectrum is None:
            spectrum = self.spectrum
        
        # Set the root path to the first directory for reference
        self.set_images_root_path(directory_paths[0])
        
        # Dictionary to store filepath lists for each common image name
        filepath_for_common_image_names = {}
        
        # Collect all filepaths for each common image name across all directories
        print("  - Looking for image files in directories...")
        files_found = 0
        
        for directory_path in directory_paths:
            dir_files_found = 0
            
            # Get all files in this directory
            try:
                all_files_in_dir = os.listdir(directory_path)
                for file in all_files_in_dir:
                    file_path = os.path.join(directory_path, file)
                    if not os.path.isfile(file_path):
                        continue
                        
                    # Check if this file matches any common image name
                    for common_image_name in common_image_names:
                        if common_image_name in file and is_image_file(file):
                            if common_image_name not in filepath_for_common_image_names:
                                filepath_for_common_image_names[common_image_name] = []
                            filepath_for_common_image_names[common_image_name].append(file_path)
                            dir_files_found += 1
                
                files_found += dir_files_found
            except Exception as e:
                print(f"  - Error reading directory {directory_path}: {str(e)}")
        
        # Count how many common images have matching files
        images_with_files = sum(1 for img_name in filepath_for_common_image_names if filepath_for_common_image_names[img_name])
        
        # Detailed debugging for no images found
        if images_with_files == 0:
            print("  - ERROR: No image files found for any common image names!")
            print("  - Sample directory contents:")
            for directory_path in directory_paths[:2]:  # Show only first two to avoid excessive output
                try:
                    all_files = os.listdir(directory_path)
                    image_files = [f for f in all_files if is_image_file(f)]
                except Exception as e:
                    print(f"    Error listing {directory_path}: {str(e)}")
            return
        
        # Parse metadata from common image names and sort chronologically
        print("  - Parsing image metadata...")
        common_name_name_with_metadata = []
        for common_image_name in common_image_names:
            try:
                metadata = parse_common_image_filename(common_image_name)
                common_name_name_with_metadata.append((common_image_name,) + metadata)
            except ValueError as e:
                print(f"  - Warning: Skipping {common_image_name}: {e}")
                continue
            
        common_name_name_with_metadata.sort(key=lambda x: x[2])  # Sort by datetime
        
        # Process each common image
        print("  - Processing individual images...")
        images_processed = 0
        for common_image_name, file_sequence, common_image_date, _ in common_name_name_with_metadata:
            if common_image_name not in ignore_list:
                filepaths = filepath_for_common_image_names.get(common_image_name, [])
                
                # Only process if we have filepaths for this image
                if filepaths:
                    try:
                        if common_image_date not in self.datetime_to_image:
                            # Create a new image entry if it doesn't exist
                            self.datetime_to_image[common_image_date] = DigitalTyphoonImage(
                                None, None, sequence_id=self.get_sequence_str(),
                                transform_func=self.transform_func,
                                spectrum=self.spectrum)
                        
                        # Set image data with all filepaths (multiple channels)
                        self.datetime_to_image[common_image_date].set_image_datas(
                            image_filepaths=filepaths, 
                            load_imgs_into_mem=load_imgs_into_mem, 
                            spectrum=spectrum)
                        
                        # Apply filter if filter_func
                        if filter_func(self.datetime_to_image[common_image_date]):
                            if self.datetime_to_image[common_image_date] not in self.images:
                                self.images.append(self.datetime_to_image[common_image_date])
                                images_processed += 1
                    except Exception as e:
                        print(f"  - Error processing image {common_image_name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        # Make sure images are sorted by time
        try:
            self.images.sort(key=lambda x: x.get_datetime())
        except Exception as e:
            pass
        
        
        if self.verbose:
            if not self.num_images_match_num_expected():
                warnings.warn(f'The number of images ({len(self.images)}) does not match the '
                             f'number of expected images ({self.num_original_images}) from metadata. If this is expected, ignore this warning.')

            if len(self.images) < self.num_track_entries:
                warnings.warn(
                    f'Only {len(self.images)} of {self.num_track_entries} track entries have images.')

    def get_start_season(self) -> int:
        """
        Get the start season of the sequence

        :return: int, the start season
        """
        return self.season

    def get_num_images(self) -> int:
        """
        Gets the number of images in the sequence

        :return: int
        """
        return len(self.images)

    def get_num_original_images(self) -> int:
        """
        Get the number of images in the sequence

        :return: int, the number of images
        """
        return self.num_original_images

    def has_images(self) -> bool:
        """
        Returns true if the sequence currently holds images (or image filepaths). False otherwise.

        :return: bool
        """
        return len(self.images) != 0

    def process_track_data(self, track_filepath: str, csv_delimiter=',') -> None:
        """
        Takes in the track data for the sequence and processes it into the images for the sequence.

        :param track_filepath: str, path to track csv
        :param csv_delimiter: delimiter for the csv file
        :return: None
        """
        if not os.path.exists(track_filepath):
            raise FileNotFoundError(
                f"The track file {track_filepath} does not exist.")

        if self.verbose:
            print(f"Processing track data from: {track_filepath}")
            
        try:
            # Load the CSV file
            df = pd.read_csv(track_filepath, delimiter=csv_delimiter)
            
            if self.verbose:
                print(f"Track data loaded: {len(df)} rows, {len(df.columns)} columns")
                print(f"Column names: {df.columns.tolist()}")
                
            # Convert to numpy array
            data = df.to_numpy()
            
            # Verify data shape
            if self.verbose:
                print(f"Track data shape: {data.shape}")
                print(f"First few rows of track data:")
                for i in range(min(3, len(data))):
                    print(f"  Row {i}: {data[i]}")
                
            # Process each row
            processed_rows = 0
            matched_rows = 0
            for row in data:
                try:
                    # Check if year is present and valid
                    if TRACK_COLS.YEAR.value < len(row):
                        year_val = row[TRACK_COLS.YEAR.value]
                        if self.verbose and processed_rows < 3:
                            print(f"Year value in row {processed_rows}: {year_val}")
                    else:
                        if self.verbose:
                            print(f"Warning: Year index {TRACK_COLS.YEAR.value} out of bounds for row length {len(row)}")
                            
                    row_datetime = datetime(int(row[TRACK_COLS.YEAR.value]), 
                                          int(row[TRACK_COLS.MONTH.value]),
                                          int(row[TRACK_COLS.DAY.value]), 
                                          int(row[TRACK_COLS.HOUR.value]))
                    
                    # Check if we already have an image for this datetime
                    existing_image = self.datetime_to_image.get(row_datetime)
                    if existing_image is not None:
                        # Update the track data for the existing image
                        if self.verbose and matched_rows < 3:
                            print(f"Found existing image for datetime {row_datetime}, updating track data")
                        existing_image.set_track_data(row)
                        matched_rows += 1
                    else:
                        # Create a new image object with this track data
                        self.datetime_to_image[row_datetime] = DigitalTyphoonImage(
                            None, row, sequence_id=self.get_sequence_str(),
                            transform_func=self.transform_func,
                            spectrum=self.spectrum,
                            verbose=self.verbose,
                            load_imgs_into_mem=self.load_imgs_into_mem
                        )
                    
                    # Debug the first row if verbose
                    if self.verbose and processed_rows == 0:
                        print(f"First row datetime: {row_datetime}")
                        print(f"First row data: {row}")
                        self.datetime_to_image[row_datetime].debug_track_data()
                        
                    processed_rows += 1
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing track row: {str(e)}")
                        import traceback
                        traceback.print_exc()
            
            # Update the counter
            self.num_track_entries = processed_rows
            
            if self.verbose:
                print(f"Successfully processed {processed_rows} track entries, matched with {matched_rows} existing images")
                # Check if we have images that don't have track data
                missing_track_data = 0
                for image in self.images:
                    if image.track_data is None or len(image.track_data) == 0:
                        missing_track_data += 1
                        if missing_track_data <= 5:  # Only show the first 5
                            print(f"  Image {image.filepath()} has no track data")
                
                if missing_track_data > 0:
                    print(f"Found {missing_track_data} images without track data")
                
        except Exception as e:
            if self.verbose:
                print(f"Error processing track file {track_filepath}: {str(e)}")
                import traceback
                traceback.print_exc()
            # Re-raise to ensure calling code knows there was an error
            raise

    def add_track_data(self, filename: str, csv_delimiter=',') -> None:
        """
        Reads and adds the track data to the sequence.

        :param filename: str, path to the track data
        :param csv_delimiter: char, delimiter to use to read the csv
        :return: None
        """
        df = pd.read_csv(filename, delimiter=csv_delimiter)
        self.track_data = df.to_numpy()

    def set_track_path(self, track_path: str) -> None:
        """
        Sets the path to the track data file

        :param track_path: str, filepath to the track data
        :return: None
        """
        if not self.track_path:
            self.track_path = track_path

    def get_track_path(self) -> str:
        """
        Gets the path to the track data file

        :return: str, the path to the track data file
        """
        return self.track_path

    def get_track_data(self) -> np.ndarray:
        """
        Returns the track csv data as a numpy array, with each row corresponding to a row in the CSV.

        :return: np.ndarray
        """
        return self.track_data

    def get_image_at_idx(self, idx, spectrum=None) -> DigitalTyphoonImage:
        """
        Gets the image at the specified index.
        
        :param idx: The index of the image to retrieve
        :param spectrum: Optional spectrum to use (defaults to sequence's spectrum)
        :return: DigitalTyphoonImage object
        """
        try:
            if idx < 0 or idx >= len(self.images):
                if self.verbose:
                    print(f"Warning: Image index {idx} out of range [0, {len(self.images) - 1}]")
                # Return a minimal DigitalTyphoonImage object
                return DigitalTyphoonImage("", np.array([]), self.get_sequence_str(), 
                                          spectrum=spectrum or self.spectrum, 
                                          verbose=self.verbose, transform_func=self.transform_func,
                                          load_imgs_into_mem=self.load_imgs_into_mem)
            
            # Get the existing image
            image = self.images[idx]
            
            # Ensure the image has a full path
            if image.image_filepath and not os.path.isabs(image.image_filepath) and os.path.exists(image.image_filepath):
                image.image_filepath = os.path.abspath(image.image_filepath)
                
            # Also fix multi-channel paths if present
            if image.image_filepaths:
                abs_image_filepaths = []
                for path in image.image_filepaths:
                    if path and not os.path.isabs(path) and os.path.exists(path):
                        abs_image_filepaths.append(os.path.abspath(path))
                    else:
                        abs_image_filepaths.append(path)
                image.image_filepaths = abs_image_filepaths
                
            return image
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving image at index {idx}: {str(e)}")
            # Return a minimal DigitalTyphoonImage object
            return DigitalTyphoonImage("", np.array([]), self.get_sequence_str(), 
                                      spectrum=spectrum or self.spectrum,
                                      verbose=self.verbose, transform_func=self.transform_func,
                                      load_imgs_into_mem=self.load_imgs_into_mem)

    def get_image_at_idx_as_numpy(self, idx: int, spectrum=None) -> np.ndarray:
        """
        Gets the idx'th image in the sequence as a numpy array. Raises an exception if the idx is outside of the
        sequence's range.

        :param idx: int, idx to access
        :param spectrum: str, spectrum of the image
        :return: np.ndarray, image as a numpy array with shape of the image dimensions
        """
        if spectrum is None:
            spectrum = self.spectrum
        return self.get_image_at_idx(idx, spectrum=spectrum).image()

    def get_all_images_in_sequence(self) -> List[DigitalTyphoonImage]:
        """
        Returns all of the image objects (DigitalTyphoonImage) in the sequence in order.

        :return: List[DigitalTyphoonImage]
        """
        return self.images

    def return_all_images_in_sequence_as_np(self, spectrum=None) -> np.ndarray:
        """
        Returns all the images in a sequence as a numpy array of shape (num_images, image_shape[0], image_shape[1])

        :param spectrum: str, spectrum of the image
        :return: np.ndarray of shape (num_image, image_shape[0], image_shape[1])
        """
        if spectrum is None:
            spectrum = self.spectrum
        return np.array([image.image(spectrum=spectrum) for image in self.images])

    def num_images_match_num_expected(self) -> bool:
        """
        Returns True if the number of image filepaths stored matches the number of images stated when initializing
        the sequence object. False otherwise.

        :return: bool
        """
        return len(self.images) == self.num_original_images

    def get_image_filepaths(self) -> List[str]:
        """
        Returns a list of the filenames of the images (without the root path)

        :return: List[str], list of the filenames
        """
        return [image.filepath() for image in self.images]

    def set_images_root_path(self, images_root_path: str) -> None:
        """
        Sets the root path of the images.

        :param images_root_path: str, the root path
        :return: None
        """
        self.img_root = Path(images_root_path)

    def get_images_root_path(self) -> str:
        """
        Gets the root path to the image directory

        :return: str, the root path
        """
        return str(self.img_root)

    def add_image_path(self, path: str, verbose=False, ignore_list=None) -> bool:
        """
        Adds an image path to the sequence. Assigns corresponding track data if available.

        :param path: Path to the image file
        :param verbose: Bool for verbose output
        :param ignore_list: Set of image filenames to ignore
        :return: True if added, False otherwise
        """
        # Skip if path doesn't exist
        if not os.path.exists(path):
            if verbose:
                print(f"Image file does not exist: {path}")
            return False

        # Skip if in ignore list
        if ignore_list and os.path.basename(path) in ignore_list:
            if verbose:
                print(f"Image file in ignore list: {path}")
            return False

        # Parse filename for metadata
        try:
            filename = os.path.basename(path)
            metadata = parse_image_filename(filename)
            file_sequence, file_datetime, _ = metadata
            
            # Check if this file belongs to this sequence
            if file_sequence != self.get_sequence_str():
                if verbose:
                    print(f"Image {filename} belongs to sequence {file_sequence}, not {self.get_sequence_str()}")
                return False
                
            if verbose:
                print(f"Processing image {filename} with datetime {file_datetime}")

            # Check if we already have track data for this datetime
            track_data = None
            if file_datetime in self.datetime_to_image:
                existing_img = self.datetime_to_image[file_datetime]
                if existing_img and hasattr(existing_img, 'track_data') and existing_img.track_data is not None:
                    if len(existing_img.track_data) > 0:
                        track_data = existing_img.track_data
                        if verbose:
                            print(f"Found existing track data for datetime {file_datetime}")

            # Create the image object with track data if available
            image = DigitalTyphoonImage(
                path, 
                track_data=track_data,
                sequence_id=self.get_sequence_str(),
                transform_func=self.transform_func, 
                spectrum=self.spectrum,
                load_imgs_into_mem=self.load_imgs_into_mem,
                verbose=verbose
            )
            
            # Store the image
            self.images.append(image)
            self.datetime_to_image[file_datetime] = image
            
            if verbose:
                print(f"Added image: {filename}")
                if track_data is not None:
                    print(f"  With track data - year: {image.year()}, wind: {image.wind()}")
                else:
                    print(f"  Warning: No track data available")
                    
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error processing image {path}: {str(e)}")
                import traceback
                traceback.print_exc()
            return False
