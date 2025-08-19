from config_test import *
import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
import numpy as np
from unittest import TestCase
import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestDigitalTyphoonSequence(TestCase):

    def test_get_sequence_str_should_return_right_str(self):
        sequence_str = '123456'
        test_sequence = DigitalTyphoonSequence(sequence_str, 0, 0)
        if test_sequence.get_sequence_str() != sequence_str:
            self.fail(
                f'Sequence string should be {sequence_str}. Program gave {test_sequence.get_sequence_str()}')

    def test_load_images_into_memory_on_startup(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        if len(test_sequence.images) != 157:
            self.fail(
                f'Sequence should have 157 images loaded in. Program gave {len(test_sequence.image_filenames)}')

    def test_process_seq_img_dir_into_seq_no_image_loading_should_process_correctly(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/')
        cwd = os.getcwd()
        should_be = [
            f'{cwd}/test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041301-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041302-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041304-200801-MTS1-1.h5'
        ]
        # Taking first 5 file
        filenames = test_sequence.get_image_filepaths()[:5]
        if should_be != filenames:
            self.fail(
                f'Processed filenames is incorrect. Program gave: \n {filenames} \n Should be: \n {should_be}')

    def test_process_seq_img_dir_into_seq_with_image_loading_should_load_correct_number(self):
        should_have = 157
        test_sequence = DigitalTyphoonSequence('200801', 2008, should_have)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        if len(test_sequence.images) != should_have:
            self.fail(
                f'Sequence should have {should_have}. Program gave {len(test_sequence.images)}')

    def test_return_all_images_in_sequence_as_np(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=False)
        length_test_sequences_images = len(test_sequence.images)
        length_test_sequences_image_should = 157
        if length_test_sequences_images != length_test_sequences_image_should:
            self.fail(
                f'Sequence should have {length_test_sequences_image_should} images. Program gave {length_test_sequences_images}')
        
        sequence_imgs = test_sequence.return_all_images_in_sequence_as_np()
        if sequence_imgs.shape[0] != length_test_sequences_image_should:
            self.fail(
                f'Returned sequence np array should have {length_test_sequences_image_should} frames in it. Shape of array is {sequence_imgs.shape}')

    def test_get_start_year_should_return_correct_year(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(2008, test_sequence.get_start_season())

    def test_get_num_images_should_return_correct_amounts(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(0, test_sequence.get_num_images())
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(157, test_sequence.get_num_images())

    def test_get_num_original_frames_should_return_5(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(5, test_sequence.get_num_original_images())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(5, test_sequence.get_num_original_images())

    def test_has_images_should_return_false_then_true(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertFalse(test_sequence.has_images())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertTrue(test_sequence.has_images())

    def test_get_img_at_idx_should_return_correct_image(self):
        test_sequence = DigitalTyphoonSequence(
            '200801', 2008, 5, spectrum='Infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        test_image = test_sequence.get_image_at_idx(4)

        read_in_image = test_image.image()
        first_values = [295.52186672208387, 295.41941557506,
                        295.41941557506, 295.41941557506, 295.41941557506]
        last_values = [288.0905295158757, 283.8272408836852,
                       285.0799629800007, 286.7644375372904, 283.8272408836852]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(
                    f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(
                    f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_transform_func(self):
        def dummy_transform_func(x):
            return x

        print("\n=== TESTING TRANSFORM_FUNC WITH LOAD_IMGS_INTO_MEM=TRUE ===")
        test_sequence = DigitalTyphoonSequence(
            '200801', 2008, 5, spectrum='Infrared', verbose=False)
        
        print("Processing track data...")
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        
        print("Processing image directory with load_imgs_into_mem=True...")
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        
        print(f"Number of images in sequence: {len(test_sequence.images)}")
        
        if len(test_sequence.images) > 4:
            image_obj = test_sequence.get_image_at_idx(4)
            print(f"Image 4 filepath: {image_obj.filepath()}")
            print(f"Image 4 exists: {os.path.exists(image_obj.filepath())}")
            print(f"Image 4 load_imgs_into_mem: {image_obj.load_imgs_into_mem}")
            print(f"Image 4 has preloaded array: {image_obj.image_array is not None}")
            
            print("Loading image...")
            read_in_image = image_obj.image()
            print(f"Loaded image type: {type(read_in_image)}")
            print(f"Loaded image shape: {read_in_image.shape if hasattr(read_in_image, 'shape') else 'None'}")
            print(f"Loaded image is empty: {read_in_image.size == 0 if hasattr(read_in_image, 'size') else 'Unknown'}")
            
            # Call debug method if it exists
            if hasattr(image_obj, 'debug_track_data'):
                image_obj.debug_track_data()
        else:
            print("Not enough images in sequence to get index 4!")

    def test_get_img_at_idx_as_numpy_should_return_correct_image(self):
        test_sequence = DigitalTyphoonSequence(
            '200801', 2008, 5, spectrum='Infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        read_in_image = test_sequence.get_image_at_idx(4).image()

        first_values = [295.52186672208387, 295.41941557506,
                        295.41941557506, 295.41941557506, 295.41941557506]
        last_values = [288.0905295158757, 283.8272408836852,
                       285.0799629800007, 286.7644375372904, 283.8272408836852]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(
                    f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(
                    f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_process_track_data_track_entries_should_be_assigned_to_correct_images(self):
        test_sequence = DigitalTyphoonSequence(
            '200801', 2008, 5, spectrum='Infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)

        image1 = test_sequence.get_image_at_idx(0)
        should_be = [2008, 4, 13, 0, 2, 8.6, 128., 1006., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., '2008041300-200801-MTS1-1.h5', 0, 0.]
        for i in range(len(image1.track_array())):
            self.assertEqual(image1.track_array()[i], should_be[i])

        image2 = test_sequence.get_image_at_idx(1)
        should_be = [2008, 4, 13, 1, 2, 8.64, 127.71, 1005.7, 0.0, 0, 0,
                     0, 0, 0, 0, 0, 1, '2008041301-200801-MTS1-1.h5', 0, 0.000000]
        for i in range(len(image2.track_array())):
            self.assertEqual(image2.track_array()[i], should_be[i])

        image3 = test_sequence.get_image_at_idx(2)
        should_be = [2008, 4, 13, 2, 2, 8.68, 127.42, 1005.3, 0.0, 0, 0,
                     0, 0, 0, 0, 0, 1, '2008041302-200801-MTS1-1.h5', 0, 0.000000]
        for i in range(len(image3.track_array())):
            self.assertEqual(image3.track_array()[i], should_be[i])

    def test_get_all_images_in_sequence_should_return_correct_list(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        cwd = os.getcwd()
        should_be = [
            f'{cwd}/test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041301-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041302-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
            f'{cwd}/test_data_files/image/200801/2008041304-200801-MTS1-1.h5'
        ]
        images = test_sequence.get_all_images_in_sequence()[:5]
        for i, image in enumerate(images):
            self.assertEqual(should_be[i], image.filepath())

    def test_get_all_images_in_sequence_as_np_should_return_correct_list(self):
        test_sequence = DigitalTyphoonSequence(
            '200801', 2008, 5, spectrum='Infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)

        images = test_sequence.return_all_images_in_sequence_as_np()
        should_be_front = [293.6596315500573,
                           295.2145126209939,
                           295.2145126209939,
                           295.41941557506,
                           295.52186672208387]
        should_be_back = [256.01710896039054,
                          285.30545727949624,
                          287.648504096486,
                          284.7417187706507,
                          283.8272408836852]

        for i, image in enumerate(images[:5]):
            self.assertEqual(should_be_front[i], images[i][0][0])
            self.assertEqual(should_be_back[i], images[i][-1][-1])

    def test_num_images_match_num_frames(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 157)
        self.assertFalse(test_sequence.num_images_match_num_expected())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertTrue(test_sequence.num_images_match_num_expected())

    def test_sequence_filter_filters_images(self):
        # no filter
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence(
            'test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(157, test_sequence.get_num_images())

        # Filter all images out
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True,
                                                        filter_func=lambda x: False)
        self.assertEqual(0, test_sequence.get_num_images())

        # filter lats lower than 8.7 out
        cwd = os.getcwd()
        should_have_filepaths = [f'{cwd}/test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
                                 f'{cwd}/test_data_files/image/200801/2008041304-200801-MTS1-1.h5']

        def filter_func(image):
            return image.lat() > 8.7

        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True,
                                                        filter_func=filter_func)

        result = sorted([str(test_sequence.get_image_at_idx(i).image_filepath)
                        for i in range(test_sequence.get_num_images())])
        self.assertEqual(result[:2], should_have_filepaths)
