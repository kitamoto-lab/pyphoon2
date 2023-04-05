import numpy as np
from unittest import TestCase

from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence


class TestDigitalTyphoonSequence(TestCase):

    def test_get_sequence_str_should_return_right_str(self):
        sequence_str = '123456'
        test_sequence = DigitalTyphoonSequence(sequence_str, 0, 0)
        if test_sequence.get_sequence_str() != sequence_str:
            self.fail(f'Sequence string should be {sequence_str}. Program gave {test_sequence.get_sequence_str()}')

    def test_load_images_into_memory_on_startup(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)

        if len(test_sequence.images) != 5:
            self.fail(f'Sequence should have 5 images loaded in. Program gave {len(test_sequence.image_filenames)}')

    def test_process_seq_img_dir_into_seq_no_image_loading_should_process_correctly(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/')
        should_be = [
            'test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041301-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041302-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041304-200801-MTS1-1.h5'
        ]
        filenames = test_sequence.get_image_filepaths()
        if should_be != filenames:
            self.fail(f'Processed filenames is incorrect. Program gave: \n {filenames} \n Should be: \n {should_be}')

    def test_process_seq_img_dir_into_seq_with_image_loading_should_load_correct_number(self):
        should_have = 5
        test_sequence = DigitalTyphoonSequence('200801', 2008, should_have)
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        if len(test_sequence.images) != should_have:
            self.fail(f'Sequence should have {should_have}. Program gave {len(test_sequence.image_arrays)}')

    def test_add_track_data(self):
        test_sequence = DigitalTyphoonSequence('200802', 2008, 157)
        test_sequence.add_track_data('test_data_files/track/200802.csv')

        should_be = np.array([[2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.],
                              [2008., 5., 7., 1., 2., 7.79, 133.17, 1003.3, 0.0, 0., 0., 0., 0., 0., 0., 0., 1.],
                              [2008., 5., 7., 2., 2., 7.78, 133.04, 1002.7, 0.0, 0., 0., 0., 0., 0., 0., 0., 1.]])

        if not np.array_equal(test_sequence.get_track_data(),should_be):
            self.fail(f'Read in data does not match. Should be: \n'
                      f'{should_be}\n'
                      f'Program gave: \n'
                      f'{test_sequence.get_track_data()}')

    def test_return_all_images_in_sequence_as_np(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=False)

        if len(test_sequence.images) != 5:
            self.fail(f'Sequence should have 5 images. Program gave {len(test_sequence.image_filenames)}')

        sequence_imgs = test_sequence.return_all_images_in_sequence_as_np()
        if sequence_imgs.shape[0] != 5:
            self.fail(f'Returned sequence np array should have 5 frames in it. Shape of array is {sequence_imgs.shape}')
