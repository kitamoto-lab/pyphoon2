from unittest import TestCase

from DigitalTyphoonDataset import DigitalTyphoonDataset
from DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonUtils import parse_image_filename


class TestDigitalTyphoonDataset(TestCase):
    def test__initialize_and_populate_images_into_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)

    def test__find_sequence_from_index_should_return_proper_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)
        test_dataset._delete_all_sequences()
        for i in range(10):  # range of frames indices is 0 ~ 199
            test_dataset.sequences.append((DigitalTyphoonSequence(str(i), 1990, 20),
                                           (20*i, 20*i + 19)))

        # test first idx
        result = test_dataset._find_sequence_str_from_index(0)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # test end of first interval
        result = test_dataset._find_sequence_str_from_index(19)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # test middle of one interval
        result = test_dataset._find_sequence_str_from_index(53)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # test end of one interval in the middle
        result = test_dataset._find_sequence_str_from_index(49)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # test last index
        result = test_dataset._find_sequence_str_from_index(199)
        if result != '9':
            self.fail(f'Should be \'9\'. Returned \'{result}\'')

    def test_populate_images_seq_images_are_read_in_chronological_order(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)
        sequences_list = test_dataset._get_list_of_sequence_objs()
        for sequence in sequences_list:
            image_paths = sequence.get_image_filenames()
            datelist = [parse_image_filename(image_path)[1] for image_path in image_paths]
            sorted_datelist = sorted(datelist)
            for i in range(0, len(datelist)):
                if datelist[i] != sorted_datelist[i]:
                    self.fail(f'Sequence \'{sequence.get_sequence_str()}\' was not read in chronological order.')

    def test__populate_track_data_into_sequences(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/track/',
                                             'test_data_files/metadata.json', verbose=True)

        seq200801 = test_dataset._get_seq_from_seq_str('200801')
        seq200802 = test_dataset._get_seq_from_seq_str('200802')

        if not (seq200801.get_track_path() == 'test_data_files/track/200801.csv'):
            self.fail(f'Sequence 200801 does not have the right path. '
                      f'Should be \'test_data_files/track/200801.csv\'. '
                      f'Path given is \'{seq200801.get_track_path()}\'')

        if not (seq200802.get_track_path() == 'test_data_files/track/200802.csv'):
            self.fail(f'Sequence 200801 does not have the right path. '
                      f'Should be \'test_data_files/track/200802.csv\'. '
                      f'Path given is \'{seq200802.get_track_path()}\'')

    def test_populate_images_reads_file_correctly(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/track/',
                                             'test_data_files/metadata.json', verbose=True)
        read_in_image = test_dataset._get_image_from_idx_as_numpy(9)

        first_values = [291.70111421052627, 291.2303478947368, 289.92525315789476, 289.92525315789476, 291.1126563157894]
        last_values = [252.34170272727272, 258.19479750000005, 247.553022, 242.84900999999996, 238.26003375]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value given was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value given was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_parse_image_filename(self):
        self.fail()

    def test_parse_track_data_filename(self):
        self.fail()

    def test_is_image_file(self):
        self.fail()

