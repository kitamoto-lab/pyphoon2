# autopep8: off
import os
import sys
import h5py
# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from config_test import *

from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from unittest import TestCase
import numpy as np
from datetime import datetime
# autopep8: on

IMAGE_FILE_PATH = f'{IMAGE_DIR}200801/2008041304-200801-MTS1-1.h5'
TRACK_ENTRY = np.array(
    [2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., ...])
FIRST_VALUES = [295.52186672208387, 295.41941557506,
                295.41941557506, 295.41941557506, 295.41941557506]
LAST_VALUES = [288.0905295158757, 283.8272408836852,
               285.0799629800007, 286.7644375372904, 283.8272408836852]


class TestDigitalTyphoonImage(TestCase):
    def setUp(self):
        self.image_file_path = IMAGE_FILE_PATH
        self.track_entry = TRACK_ENTRY
        self.test_image = DigitalTyphoonImage(
            self.image_file_path,
            self.track_entry,
            load_imgs_into_mem=True,
            spectrum='Infrared'
        )

    def test_initialization_should_succeed(self):
        """Test successful initialization with valid data."""
        self.assertIsInstance(self.test_image, DigitalTyphoonImage)

    def test_initialization_load_image_into_memory_should_fail(self):
        """Test that initializing with a nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            DigitalTyphoonImage('nonexistent/file',self.track_entry, load_imgs_into_mem=True)

    def test_image_loading(self):
        """Test that image is loaded correctly and values match expectations."""
        read_in_image = self.test_image.image()
        np.testing.assert_array_equal(
            read_in_image[0, :len(FIRST_VALUES)], FIRST_VALUES)
        np.testing.assert_array_equal(
            read_in_image[-1, -len(LAST_VALUES):], LAST_VALUES)

    def test_transform_func(self):
        """Test that transform_func correctly transforms the image."""
        test_image = DigitalTyphoonImage(
            self.image_file_path,
            self.track_entry,
            load_imgs_into_mem=True,
            transform_func=lambda img: np.ones(img.shape),
            spectrum='Infrared'
        )
        transformed_image = test_image.image()
        self.assertTrue(np.array_equal(transformed_image,
                        np.ones(transformed_image.shape)))

    def test_track_getters(self):
        """Test that track data getters return correct values."""
        self.assertEqual(self.test_image.year(), 2008)
        self.assertEqual(self.test_image.month(), 5)
        self.assertEqual(self.test_image.day(), 7)
        self.assertEqual(self.test_image.grade(), 2)
        self.assertEqual(self.test_image.lat(), 7.80)
        self.assertEqual(self.test_image.long(), 133.30)
        self.assertEqual(self.test_image.pressure(), 1004.0)
        self.assertEqual(self.test_image.wind(), 0.0)

    def test_set_track_data(self):
        """Test that setting track data later works correctly."""
        test_image = DigitalTyphoonImage(
            self.image_file_path, None, load_imgs_into_mem=False)
        test_image.set_track_data(self.track_entry)
        np.testing.assert_array_equal(
            test_image.track_array(), self.track_entry)

    def _create_modified_h5_file(self):
        """Create a modified h5 file for testing."""
        with h5py.File(self.image_file_path, 'r') as f:
            data = f['Infrared'][:]
        image_file_path_modified = f'{IMAGE_DIR}200801/2008041304-200801-MTS1-1_modified.h5'
        with h5py.File(image_file_path_modified, 'w') as f:
            f.create_dataset('Infrared_modified', data=data)
        return image_file_path_modified

    def test_initialization_with_no_spectrum(self):
        """Test successful initialization with valid data."""
        # image_file_path_modified = self._create_modified_h5_file()
        image_file_path_modified = '/app/197830/1978120103-197830-GMS1-1.h5'
        test_image_no_specific_spectrum = DigitalTyphoonImage(
            image_file_path_modified,
            self.track_entry,
            load_imgs_into_mem=True,
        )
        self.assertIsInstance(
            test_image_no_specific_spectrum, DigitalTyphoonImage)
        # remove image_file_path_modified
        # os.remove(image_file_path_modified)
