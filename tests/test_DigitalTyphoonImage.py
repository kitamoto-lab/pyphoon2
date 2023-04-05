import numpy as np
from unittest import TestCase

from DigitalTyphoonDataloader.DigitalTyphoonImage import DigitalTyphoonImage


class TestDigitalTyphoonImage(TestCase):
    def test_year(self):
        test_image = DigitalTyphoonImage('', np.array([]))
        print(test_image)
