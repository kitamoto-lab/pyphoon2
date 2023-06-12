from unittest import TestCase

from pyphoon2.DigitalTyphoonUtils import *


class TestDigitalTyphoonUtils(TestCase):

    def test_parse_image_filename(self):
        filename = '2008041301-200801-MTS1-1.h5'
        sequence_num, sequence_datetime, satellite = parse_image_filename(filename)
        self.assertEqual('200801', sequence_num)
        self.assertEqual(datetime(2008, 4, 13, 1), sequence_datetime)
        self.assertEqual('MTS1', satellite)

    def test_get_seq_str_from_track_filename(self):
        filename = '200801.csv'
        seq_str = get_seq_str_from_track_filename(filename)
        self.assertEqual('200801', seq_str)

    def test_is_image_file(self):
        filename = '200801.csv'
        self.assertFalse(is_image_file(filename))
        filename = '2008041302-200801-MTS1-1.h5'
        self.assertTrue(is_image_file(filename))

    def test_split_unit_has_value(self):
        self.assertTrue(SPLIT_UNIT.has_value('sequence'))
        self.assertFalse(SPLIT_UNIT.has_value('nonexistent_value'))

    def test_load_unit_has_value(self):
        self.assertTrue(LOAD_DATA.has_value('track'))
        self.assertFalse(LOAD_DATA.has_value('nonexistent_value'))
