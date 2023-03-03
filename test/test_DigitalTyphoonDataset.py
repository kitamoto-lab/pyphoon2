from unittest import TestCase

from DigitalTyphoonDataset import DigitalTyphoonDataset


class TestDigitalTyphoonDataset(TestCase):
    def test__populate_images_into_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/jma/", verbose=True)
        print("success")

    def test__populate_track_data_into_sequences(self):
        self.fail()

    def test_parse_image_filename(self):
        self.fail()

    def test_parse_track_data_filename(self):
        self.fail()

    def test_is_image_file(self):
        self.fail()
