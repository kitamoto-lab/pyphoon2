from unittest import TestCase
from unittest.mock import MagicMock

from DigitalTyphoonSequence import DigitalTyphoonSequence


class TestTyphoonSequence(TestCase):
    def test_get_sequence(self):
        sequence_str = '123456'
        test_sequence = DigitalTyphoonSequence(sequence_str, MagicMock(), '', '')
        if test_sequence.get_sequence() != sequence_str:
            self.fail()

    def test_append_image_to_sequence(self):
        test_typhoon_image = MagicMock()
        test_sequence = DigitalTyphoonSequence('', MagicMock(), '', '')
        returned_image = test_sequence.append_image_to_sequence(test_typhoon_image)
        if test_typhoon_image != returned_image:
            self.fail()

    def test_get_track_data_path(self):
        path_string = '/test/directory/'
        test_sequence = DigitalTyphoonSequence('', MagicMock(), '', path_string)
        if test_sequence.get_track_data_path() != path_string:
            self.fail()

    def test_set_track_data_path(self):
        path_string = '/test/directory/'
        test_sequence = DigitalTyphoonSequence('', MagicMock(), '', '')
        test_sequence.set_track_data_path(path_string)
        if test_sequence.get_track_data_path() != path_string:
            self.fail()

    def test_has_images(self):
        test_sequence = DigitalTyphoonSequence('', MagicMock(), '', '')
        if test_sequence.has_images():
            self.fail()

        test_typhoon_image = MagicMock()
        test_sequence.append_image_to_sequence(test_typhoon_image)
        if not test_sequence.has_images():
            self.fail()