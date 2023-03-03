from datetime import datetime
from unittest import TestCase
from unittest.mock import MagicMock

from DigitalTyphoonImage import DigitalTyphoonImage


class TestTyphoonImage(TestCase):
    def test_get_date(self):
        test_date = datetime(1990, 1, 2)
        test_image = DigitalTyphoonImage('', test_date, '', '')
        if test_image.get_date() != test_date:
            self.fail()

    def test_get_sequence(self):
        test_sequence = "1234567"
        test_image = DigitalTyphoonImage('', MagicMock(), test_sequence, '')
        print(test_image.get_sequence(), test_image)
        if test_image.get_sequence() != test_sequence:
            self.fail()

    def test_get_satellite(self):
        test_satellite = "test_sat"
        test_image = DigitalTyphoonImage('', MagicMock(), '', test_satellite)
        if test_image.get_satellite() != test_satellite:
            self.fail()

    def test_set_date(self):
        test_set_date = datetime(1990, 1, 2)
        test_image = DigitalTyphoonImage('', datetime(2000, 2, 15), '', '')
        test_image.set_date(test_set_date)
        if test_image.get_date() != test_set_date:
            self.fail()

    def test_set_sequence(self):
        test_set_seq = "132435"
        test_image = DigitalTyphoonImage('', MagicMock(), '', '')
        test_image.set_sequence(test_set_seq)
        if test_image.get_sequence() != test_set_seq:
            self.fail()

    def test_set_satellite(self):
        test_set_sat = "example_sat"
        test_image = DigitalTyphoonImage('', MagicMock(), '', '')
        test_image.set_satellite(test_set_sat)
        if test_image.get_satellite() != test_set_sat:
            self.fail()
