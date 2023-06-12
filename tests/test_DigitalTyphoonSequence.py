import numpy as np
from unittest import TestCase

from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence


class TestDigitalTyphoonSequence(TestCase):

    def test_get_sequence_str_should_return_right_str(self):
        sequence_str = '123456'
        test_sequence = DigitalTyphoonSequence(sequence_str, 0, 0)
        if test_sequence.get_sequence_str() != sequence_str:
            self.fail(f'Sequence string should be {sequence_str}. Program gave {test_sequence.get_sequence_str()}')

    def test_load_images_into_memory_on_startup(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)

        if len(test_sequence.images) != 5:
            self.fail(f'Sequence should have 5 images loaded in. Program gave {len(test_sequence.image_filenames)}')

    def test_process_seq_img_dir_into_seq_no_image_loading_should_process_correctly(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
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
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        if len(test_sequence.images) != should_have:
            self.fail(f'Sequence should have {should_have}. Program gave {len(test_sequence.images)}')

    def test_add_track_data(self):
        test_sequence = DigitalTyphoonSequence('000001', 0000, 157)
        test_sequence.add_track_data('test_data_files/metadata/000001.csv')

        should_be = [[2008,5,7,0,2,7.80,133.30,1004.0,0.0,0,0,0,0,0,0,0,0,'2008050700-200802-MTS1-1.h5',0,0.000000],
                  [2008,5,7,1,2,7.79,133.17,1003.3,0.0,0,0,0,0,0,0,0,1,'2008050701-200802-MTS1-1.h5',0,0.000000],
                  [2008,5,7,2,2,7.78,133.04,1002.7,0.0,0,0,0,0,0,0,0,1,'2008050702-200802-MTS1-1.h5',0,0.000000]]

        for row in range(len(should_be)):
            for i in range(len(should_be[row])):
                self.assertEqual(should_be[row][i], test_sequence.get_track_data()[row][i])

    def test_return_all_images_in_sequence_as_np(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=False)

        if len(test_sequence.images) != 5:
            self.fail(f'Sequence should have 5 images. Program gave {len(test_sequence.image_filenames)}')

        sequence_imgs = test_sequence.return_all_images_in_sequence_as_np()
        if sequence_imgs.shape[0] != 5:
            self.fail(f'Returned sequence np array should have 5 frames in it. Shape of array is {sequence_imgs.shape}')

    def test_get_start_year_should_return_correct_year(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(2008, test_sequence.get_start_season())

    def test_get_num_images_should_return_correct_amounts(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(0, test_sequence.get_num_images())
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(5, test_sequence.get_num_images())

    def test_get_num_original_frames_should_return_5(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertEqual(5, test_sequence.get_num_original_images())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(5, test_sequence.get_num_original_images())

    def test_has_images_should_return_false_then_true(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertFalse(test_sequence.has_images())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertTrue(test_sequence.has_images())

    def test_get_img_at_idx_should_return_correct_image(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        test_image = test_sequence.get_image_at_idx(4)

        read_in_image = test_image.image()
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_transform_func(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        test_image = test_sequence.get_image_at_idx(4)

        read_in_image = test_image.image()
        should_be_shape = read_in_image.shape
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i - 1] != last_values[-i - 1]:
                self.fail(f'Value produced was {read_in_image[-1][-i - 1]}. Should be {last_values[-i - 1]}')

        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, transform_func=lambda img: np.ones(img.shape), spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        test_image = test_sequence.get_image_at_idx(4)
        self.assertTrue(np.array_equal(np.ones(should_be_shape), test_image.image()))

    def test_get_img_at_idx_as_numpy_should_return_correct_image(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        read_in_image = test_sequence.get_image_at_idx(4).image()

        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')


    def test_process_track_data_track_entries_should_be_assigned_to_correct_images(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)

        image1 = test_sequence.get_image_at_idx(0)
        should_be = [2008, 4, 13, 0, 2, 8.6, 128., 1006., 0., 0., 0., 0., 0., 0., 0., 0., 0.,'2008041300-200801-MTS1-1.h5', 0 ,0.]
        for i in range(len(image1.track_array())):
            self.assertEqual(image1.track_array()[i], should_be[i])

        image2 = test_sequence.get_image_at_idx(1)
        should_be = [2008,4,13,1,2,8.64,127.71,1005.7,0.0,0,0,0,0,0,0,0,1,'2008041301-200801-MTS1-1.h5',0,0.000000]
        for i in range(len(image2.track_array())):
            self.assertEqual(image2.track_array()[i], should_be[i])

        image3 = test_sequence.get_image_at_idx(2)
        should_be = [2008,4,13,2,2,8.68,127.42,1005.3,0.0,0,0,0,0,0,0,0,1,'2008041302-200801-MTS1-1.h5',0,0.000000]
        for i in range(len(image3.track_array())):
            self.assertEqual(image3.track_array()[i], should_be[i])

    def test_get_all_images_in_sequence_should_return_correct_list(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        should_be = [
            'test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041301-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041302-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041304-200801-MTS1-1.h5'
        ]
        images = test_sequence.get_all_images_in_sequence()
        for i, image in enumerate(images):
            self.assertEqual(should_be[i], image.filepath())

    def test_get_all_images_in_sequence_as_np_should_return_correct_list(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5, spectrum='infrared')
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)

        images = test_sequence.return_all_images_in_sequence_as_np()
        should_be_front = [294.60495000000003,
                             295.970988,
                             296.083902,
                             296.083902,
                             296.30972999999994]
        should_be_back = [274.92027599999994,
                            283.5636017647059,
                            285.31107,
                            282.8017252941176,
                            282.0398488235294]

        for i, image in enumerate(images):
            self.assertEqual(should_be_front[i], images[i][0][0])
            self.assertEqual(should_be_back[i], images[i][-1][-1])

    def test_num_images_match_num_frames(self):
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        self.assertFalse(test_sequence.num_images_match_num_expected())

        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertTrue(test_sequence.num_images_match_num_expected())

    def test_sequence_filter_filters_images(self):
        # no filter
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True)
        self.assertEqual(5, test_sequence.get_num_images())

        # Filter all images out
        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True,
                                                        filter_func=lambda x: False)
        self.assertEqual(0, test_sequence.get_num_images())

        # filter lats lower than 8.7 out

        should_have_filepaths = ['test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
                                 'test_data_files/image/200801/2008041304-200801-MTS1-1.h5']

        def filter_func(image):
            return image.lat() > 8.7

        test_sequence = DigitalTyphoonSequence('200801', 2008, 5)
        test_sequence.process_track_data('test_data_files/metadata/200801.csv')
        test_sequence.process_seq_img_dir_into_sequence('test_data_files/image/200801/', load_imgs_into_mem=True,
                                                        filter_func=filter_func)

        result = sorted([str(test_sequence.get_image_at_idx(i).image_filepath) for i in range(test_sequence.get_num_images())])
        self.assertEqual(result, should_have_filepaths)
