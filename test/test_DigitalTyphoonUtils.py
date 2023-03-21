import csv
import pandas as pd
from unittest import TestCase

from DigitalTyphoonUtils import read_metadata_file, read_track_file


class Test(TestCase):
    def test_read_metadata_file(self):
        metadata_file = "../data/metadata.json"
        data = read_metadata_file(metadata_file)

        read_track_file('../data/track/197830.csv')


