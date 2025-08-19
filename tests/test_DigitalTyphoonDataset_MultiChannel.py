# autopep8: off
import os
import sys
import os.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from unittest import TestCase
from pyphoon2.DigitalTyphoonUtils import parse_image_filename
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from config_test import *
# autopep8: on


class TestDigitalTyphoonDatasetMultiChannel(TestCase):

    def create_test_dataset(
        self,
        image_dir=IMAGE_DIR,
        metadata_dir=METADATA_DIR,
        metadata_json=METADATA_JSON,
        label_column='grade',
        split_dataset_by='image',  # Use the correct parameter name
        verbose=False,
        image_dirs=IMAGE_DIRS,
        metadata_dirs=METADATA_DIRS,
        metadata_jsons=METADATA_JSONS,
        **kwargs
    ):
        """
        Helper function to initialize a DigitalTyphoonDataset.

        Args:
            image_dir (str): Path to the image directory.
            metadata_dir (str): Path to the metadata directory.
            metadata_json (str): Path to the metadata JSON file.
            label_column (str): Label column name (default is 'grade').
            split_dataset_by (str): Method to split the dataset (e.g., 'image', 'sequence').
            verbose (bool): Verbosity flag (default is False).
            **kwargs: Additional keyword arguments for customization.

        Returns:
            DigitalTyphoonDataset: Initialized dataset object.
        """
        return DigitalTyphoonDataset(
            image_dir,
            metadata_dir,
            metadata_json,
            label_column,
            split_dataset_by=split_dataset_by,  # Correct parameter passed
            verbose=verbose,
            image_dirs=image_dirs,
            metadata_dirs=metadata_dirs,
            metadata_jsons=metadata_jsons,
            **kwargs
        )
        # return DigitalTyphoonDataset(
        #     image_dir,
        #     metadata_dir,
        #     metadata_json,
        #     label_column,
        #     split_dataset_by=split_dataset_by,  # Correct parameter passed
        #     verbose=verbose,
        #     **kwargs
        # )

    def test__initialize_and_populate_images_into_sequences(self):
        test_dataset = self.create_test_dataset()

    # def test_populate_images_seq_images_are_read_in_chronological_order(self):
    #     test_dataset = self.create_test_dataset()
    #     sequences_list = test_dataset._get_list_of_sequence_objs()
    #     for sequence in sequences_list:
    #         image_paths = sequence.get_image_filepaths()
    #         datelist = [parse_image_filename(os.path.basename(image_path))[
    #             1] for image_path in image_paths]
    #         sorted_datelist = sorted(datelist)
    #         for i in range(0, len(datelist)):
    #             if datelist[i] != sorted_datelist[i]:
    #                 self.fail(
    #                     f'Sequence \'{sequence.get_sequence_str()}\' was not read in chronological order.')
