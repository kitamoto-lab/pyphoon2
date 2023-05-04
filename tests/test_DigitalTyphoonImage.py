from datetime import datetime

import numpy as np
from unittest import TestCase

from DigitalTyphoonDataloader.DigitalTyphoonImage import DigitalTyphoonImage


class TestDigitalTyphoonImage(TestCase):
    def test_initialization_should_succeed(self):
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
                                         np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

    def test_initialization_load_image_into_memory_should_fail(self):
        with self.assertRaises(FileNotFoundError):
            test_image = DigitalTyphoonImage('nonexistent/file',
                                             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                             load_imgs_into_mem=True)

    def test_initialization_load_image_into_memory_should_succeed(self):
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                         load_imgs_into_mem=True)

        read_in_image = test_image.image()
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_transform_func(self):
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                         load_imgs_into_mem=True)

        read_in_image = test_image.image()
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        shape = read_in_image.shape
        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i - 1] != last_values[-i - 1]:
                self.fail(f'Value produced was {read_in_image[-1][-i - 1]}. Should be {last_values[-i - 1]}')

        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                         load_imgs_into_mem=True,
                                         transform_func=lambda img: np.ones(img.shape))

        read_in_image = test_image.image()
        self.assertTrue(np.array_equal(np.ones(shape), read_in_image))

    def test_give_track_entry_on_init(self):
        should_be = np.array([2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.])
        track_entry = np.array([2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.])
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         track_entry,
                                         load_imgs_into_mem=False)
        self.assertTrue(np.array_equal(should_be, test_image.track_array()))

    def test_give_track_entry_later_should_succeed(self):
        should_be = np.array([2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.])
        track_entry = np.array([2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.])
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         None,
                                         load_imgs_into_mem=False)
        test_image.set_track_data(track_entry)
        self.assertTrue(np.array_equal(should_be, test_image.track_array()))

    def test_give_track_entry_later_should_fail_not_enough_columns(self):
        track_entry = np.array([2008., 5., 7.80, 133.30, 1004.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0.])
        with self.assertRaises(ValueError) as err:
            test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                             track_entry,
                                             load_imgs_into_mem=False)
        self.assertEqual(str(err.exception), f'Number of columns in the track entry (14) is not equal '
                                             f'to expected amount (17)')

    def test_track_getters_return_correct_values(self):
        track_entry = np.array([2008., 5., 7., 0., 2., 7.80, 133.30, 1004.0, 0.1, 2., 3., 4., 5., 6., 7., 8., 1])
        test_image = DigitalTyphoonImage('test_data_files/image/200801/2008041304-200801-MTS1-1.h5',
                                         track_entry,
                                         load_imgs_into_mem=False)
        self.assertEqual(2008, test_image.year())
        self.assertEqual(5, test_image.month())
        self.assertEqual(7, test_image.day())
        self.assertEqual(0, test_image.hour())
        self.assertEqual(datetime(2008, 5, 7, 0), test_image.datetime())
        self.assertEqual(2, test_image.grade())
        self.assertEqual(7.80, test_image.lat())
        self.assertEqual(133.30, test_image.long())
        self.assertEqual(1004.0, test_image.pressure())
        self.assertEqual(0.1, test_image.wind())
        self.assertEqual(2., test_image.dir50())
        self.assertEqual(3., test_image.long50())
        self.assertEqual(4., test_image.short50())
        self.assertEqual(5., test_image.dir30())
        self.assertEqual(6., test_image.long30())
        self.assertEqual(7., test_image.short30())
        self.assertEqual(8., test_image.landfall())
        self.assertTrue(test_image.interpolated())


