import os.path
from unittest import TestCase

from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset
from DigitalTyphoonDataloader.DigitalTyphoonSequence import DigitalTyphoonSequence
from DigitalTyphoonDataloader.DigitalTyphoonUtils import parse_image_filename


class TestDigitalTyphoonDataset(TestCase):
    def test__initialize_and_populate_images_into_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)

    def test__find_sequence_from_index_should_return_proper_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)
        test_dataset._delete_all_sequences()
        for i in range(10):  # range of frames indices is 0 ~ 199
            sequence_obj = (DigitalTyphoonSequence(str(i), 1990, 20))
            test_dataset.sequences.append(sequence_obj)
            for j in range(20*i, 20*i+20):
                test_dataset._frame_idx_to_sequence[j] = sequence_obj
            test_dataset._seq_str_to_first_total_idx[str(i)] = 20 * i

        # tests first idx
        result = test_dataset._find_sequence_str_from_frame_index(0)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # tests end of first interval
        result = test_dataset._find_sequence_str_from_frame_index(19)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # tests middle of one interval
        result = test_dataset._find_sequence_str_from_frame_index(53)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # tests end of one interval in the middle
        result = test_dataset._find_sequence_str_from_frame_index(49)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # tests last index
        result = test_dataset._find_sequence_str_from_frame_index(199)
        if result != '9':
            self.fail(f'Should be \'9\'. Returned \'{result}\'')

    def test_populate_images_seq_images_are_read_in_chronological_order(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/track/", "../data/metadata.json", verbose=True)
        sequences_list = test_dataset._get_list_of_sequence_objs()
        for sequence in sequences_list:
            image_paths = sequence.get_image_filepaths()
            datelist = [parse_image_filename(os.path.basename(image_path))[1] for image_path in image_paths]
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
        read_in_image = test_dataset._get_image_from_idx_as_numpy(4)
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_random_split_by_frame_random_produces_nonidentical_indices(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/track",
                                             "test_data_files/metadata.json",
                                             split_dataset_by='frame',
                                             verbose=True)

        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        bucket1_2, bucket2_2 = test_dataset.random_split([0.7, 0.3])

        i = 0
        all_same = True
        while i < len(bucket1_1) and i < len(bucket1_2):
            if bucket1_1[i] != bucket1_2[i]:
                all_same = False
            i += 1

        self.assertFalse(all_same)

        bucket1_1, bucket2_1, bucket3_1= test_dataset.random_split([0.5, 0.25, 0.25])
        bucket1_2, bucket2_2, bucket3_2 = test_dataset.random_split([0.5, 0.25, 0.25])

        i = 0
        all_same = True
        while i < len(bucket1_1) and i < len(bucket1_2):
            if bucket1_1[i] != bucket1_2[i]:
                all_same = False
            i += 1
        self.assertFalse(all_same)

        i = 0
        while i < len(bucket2_1) and i < len(bucket2_2):
            if bucket2_1[i] != bucket2_2[i]:
                all_same = False
            i += 1

        self.assertFalse(all_same)




    def test_random_split_by_sequence_no_leakage(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/track",
                                             "test_data_files/metadata.json",
                                             split_dataset_by='sequence',
                                             verbose=True)
        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        bucket1_1_sequences = set()
        bucket2_1_sequences = set()
        for i in range(0, len(bucket1_1.indices)):
            bucket1_1_sequences.add(test_dataset._find_sequence_str_from_frame_index(bucket1_1.indices[i]))
        for i in range(0, len(bucket2_1.indices)):
            bucket2_1_sequences.add(test_dataset._find_sequence_str_from_frame_index(bucket2_1.indices[i]))

        self.assertTrue(len(bucket1_1_sequences.intersection(bucket2_1_sequences)) == 0)

    def test_random_split_by_year_no_leakage(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/track",
                                             "test_data_files/metadata.json",
                                             split_dataset_by='year',
                                             verbose=True)
        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        bucket1_1_years = set()
        bucket2_1_years = set()
        for i in range(0, len(bucket1_1.indices)):
            bucket1_1_years.add(test_dataset._find_sequence_str_from_frame_index(bucket1_1.indices[i]))
        for i in range(0, len(bucket2_1.indices)):
            bucket2_1_years.add(test_dataset._find_sequence_str_from_frame_index(bucket2_1.indices[i]))

        self.assertTrue(len(bucket1_1_years.intersection(bucket2_1_years)) == 0)


