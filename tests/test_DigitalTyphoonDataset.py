import os.path
from unittest import TestCase

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize

from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
from pyphoon2.DigitalTyphoonUtils import parse_image_filename

class TestDigitalTyphoonDataset(TestCase):

    def test__initialize_and_populate_images_into_sequences(self):
        # tests process_metadata_file, populate_images_into_sequences, _populate_track_data_into_sequences, and
        #  _assign_all_images_a_dataset_idx

        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             get_images_by_sequence=True,
                                             split_dataset_by='image',
                                             verbose=False)

        def filter_func(image):
            return image.grade() < 7
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade', verbose=False, filter_func=filter_func)
        test, train = test_dataset.random_split([0.8, 0.2], split_by='sequence')

        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)

        self.assertEqual(5, len(test_dataset.sequences))
        self.assertEqual(4, len(test_dataset.season_to_sequence_nums))
        self.assertEqual(428, test_dataset.number_of_images)
        self.assertEqual(5, len(test_dataset._sequence_str_to_seq_idx))
        self.assertEqual(428, len(test_dataset._image_idx_to_sequence))
        self.assertEqual(5, len(test_dataset._seq_str_to_first_total_idx))

    def test_len(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertEqual(len(test_dataset), 428)

    def test_getitem(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertTrue(np.array_equal(test_dataset[0][0], test_dataset.get_image_from_idx(0).image()))
        self.assertTrue(test_dataset[0][1], test_dataset.get_image_from_idx(0).grade())

    def test_setlabel(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertTrue(test_dataset[0][1], test_dataset.get_image_from_idx(0).grade())
        test_dataset.set_label(('lat', 'lng'))
        self.assertTrue(np.array_equal(test_dataset[0][1], (test_dataset.get_image_from_idx(0).lat(), test_dataset.get_image_from_idx(0).long())))
        with self.assertRaises(KeyError) as err:
            test_dataset.set_label('nonexistent_label')
        self.assertEqual(str(err.exception), "'nonexistent_label is not a valid column name.'")

    def test__find_sequence_from_index_should_return_proper_sequences(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade', verbose=False)
        test_dataset._delete_all_sequences()
        for i in range(10):  # range of images indices is 0 ~ 199
            sequence_obj = (DigitalTyphoonSequence(str(i), 1990, 20))
            test_dataset.sequences.append(sequence_obj)
            for j in range(20*i, 20*i+20):
                test_dataset._image_idx_to_sequence[j] = sequence_obj
            test_dataset._seq_str_to_first_total_idx[str(i)] = 20 * i

        # tests first idx
        result = test_dataset._find_sequence_str_from_image_index(0)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # tests end of first interval
        result = test_dataset._find_sequence_str_from_image_index(19)
        if result != '0':
            self.fail(f'Should be \'0\'. Returned \'{result}\'')

        # tests middle of one interval
        result = test_dataset._find_sequence_str_from_image_index(53)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # tests end of one interval in the middle
        result = test_dataset._find_sequence_str_from_image_index(49)
        if result != '2':
            self.fail(f'Should be \'2\'. Returned \'{result}\'')

        # tests last index
        result = test_dataset._find_sequence_str_from_image_index(199)
        if result != '9':
            self.fail(f'Should be \'9\'. Returned \'{result}\'')

    def test_populate_images_seq_images_are_read_in_chronological_order(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade', verbose=False)
        sequences_list = test_dataset._get_list_of_sequence_objs()
        for sequence in sequences_list:
            image_paths = sequence.get_image_filepaths()
            datelist = [parse_image_filename(os.path.basename(image_path))[1] for image_path in image_paths]
            sorted_datelist = sorted(datelist)
            for i in range(0, len(datelist)):
                if datelist[i] != sorted_datelist[i]:
                    self.fail(f'Sequence \'{sequence.get_sequence_str()}\' was not read in chronological order.')


    def test__populate_track_data_into_sequences(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', verbose=False)

        seq200801 = test_dataset._get_seq_from_seq_str('200801')
        seq200802 = test_dataset._get_seq_from_seq_str('200802')

        if not (seq200801.get_track_path() == 'test_data_files/metadata/200801.csv'):
            self.fail(f'Sequence 200801 does not have the right path. '
                      f'Should be \'test_data_files/metadata/200801.csv\'. '
                      f'Path given is \'{seq200801.get_track_path()}\'')

        if not (seq200802.get_track_path() == 'test_data_files/metadata/200802.csv'):
            self.fail(f'Sequence 200801 does not have the right path. '
                      f'Should be \'test_data_files/metadata/200802.csv\'. '
                      f'Path given is \'{seq200802.get_track_path()}\'')

    def test_populate_images_reads_file_correctly(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', verbose=False,
                                             spectrum='infrared')
        read_in_image = test_dataset._get_image_from_idx_as_numpy(54)
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')

    def test_transform_func_transforms(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', verbose=False,
                                             spectrum='infrared')
        read_in_image = test_dataset._get_image_from_idx_as_numpy(54)
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]
        should_be_shape = read_in_image.shape
        for i in range(len(first_values)):
            if read_in_image[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image[0][i]}. Should be {first_values[i]}')
            if read_in_image[-1][-i-1] != last_values[-i-1]:
                self.fail(f'Value produced was {read_in_image[-1][-i-1]}. Should be {last_values[-i-1]}')
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', transform_func=lambda img: np.ones(img.shape),
                                             spectrum='infrared', verbose=False)
        read_in_image = test_dataset._get_image_from_idx_as_numpy(4)
        self.assertTrue(np.array_equal(np.ones(should_be_shape), read_in_image))


    def test_random_split_by_image_random_produces_nonidentical_indices(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             "grade",
                                             split_dataset_by='image',
                                             spectrum='infrared',
                                             verbose=False)
        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        bucket1_2, bucket2_2 = test_dataset.random_split([0.7, 0.3])

        i = 0
        all_same = True
        while i < len(bucket1_1) and i < len(bucket1_2):
            if not np.array_equal(bucket1_1[i][0], bucket1_2[i][0]):
                all_same = False
            i += 1

        self.assertFalse(all_same)

        bucket1_1, bucket2_1, bucket3_1= test_dataset.random_split([0.5, 0.25, 0.25])
        bucket1_2, bucket2_2, bucket3_2 = test_dataset.random_split([0.5, 0.25, 0.25])

        i = 0
        all_same = True
        while i < len(bucket1_1) and i < len(bucket1_2):
            if not np.array_equal(bucket1_1[i][0], bucket1_2[i][0]):
                all_same = False
            i += 1
        self.assertFalse(all_same)

        i = 0
        while i < len(bucket2_1) and i < len(bucket2_2):
            if not np.array_equal(bucket2_1[i][0], bucket2_2[i][0]):
                all_same = False
            i += 1

        self.assertFalse(all_same)

    def test_read_in_sorted_order(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             "grade",
                                             split_dataset_by='image',
                                             spectrum='infrared',
                                             verbose=False)
        num_seq = test_dataset.get_number_of_sequences()
        seq_strs = []
        for i in range(num_seq):
            seq_strs.append(test_dataset.get_ith_sequence(0).get_sequence_str())

        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadatashuffled.json",
                                             "grade",
                                             split_dataset_by='image',
                                             spectrum='infrared',
                                             verbose=False)

        for i in range(num_seq):
            self.assertEqual(seq_strs[i], test_dataset.get_ith_sequence(0).get_sequence_str())

    def test_random_split_by_sequence_no_leakage(self):
        # test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
        #                                      "test_data_files/metadata.json",
        #                                      'grade',
        #                                      split_dataset_by='sequence',
        #                                      verbose=False)
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade',
                                             split_dataset_by='sequence', verbose=False)

        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        self.assertEqual(len(test_dataset), len(bucket1_1)+len(bucket2_1))
        bucket1_1_sequences = set()
        bucket2_1_sequences = set()
        for i in range(0, len(bucket1_1.indices)):
            bucket1_1_sequences.add(test_dataset._find_sequence_str_from_image_index(bucket1_1.indices[i]))
        for i in range(0, len(bucket2_1.indices)):
            bucket2_1_sequences.add(test_dataset._find_sequence_str_from_image_index(bucket2_1.indices[i]))

        self.assertTrue(len(bucket1_1_sequences.intersection(bucket2_1_sequences)) == 0)

    def test_random_split_by_season_no_leakage(self):
        # test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
        #                                      "test_data_files/metadata.json",
        #                                      'grade',
        #                                      split_dataset_by='season',
        #                                      verbose=False)
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade',
                                             split_dataset_by='season', verbose=False)

        bucket1_1, bucket2_1 = test_dataset.random_split([0.7, 0.3])
        self.assertEqual(len(test_dataset), len(bucket1_1)+len(bucket2_1))
        bucket1_1_seasons = set()
        bucket2_1_seasons = set()
        for i in range(0, len(bucket1_1.indices)):
            bucket1_1_seasons.add(test_dataset._find_sequence_str_from_image_index(bucket1_1.indices[i]))
        for i in range(0, len(bucket2_1.indices)):
            bucket2_1_seasons.add(test_dataset._find_sequence_str_from_image_index(bucket2_1.indices[i]))
        self.assertTrue(len(bucket1_1_seasons.intersection(bucket2_1_seasons)) == 0)

    def test_random_split_get_sequence(self):
        test_dataset = DigitalTyphoonDataset("../data/image/", "../data/metadata/", "../data/metadata.json", 'grade',
                                             get_images_by_sequence=True,
                                             verbose=False)
        test, train = test_dataset.random_split([0.8, 0.2], split_by='season')
        test_seasons = set([test_dataset.get_ith_sequence(i) for i in test.indices])
        train_seasons = set([test_dataset.get_ith_sequence(i) for i in train.indices])
        self.assertTrue(len(test_seasons.intersection(train_seasons)) == 0)

        test, train = test_dataset.random_split([0.8, 0.2], split_by='image')
        self.assertEqual(len(test.indices), len(set(test.indices)))
        self.assertEqual(len(train.indices), len(set(train.indices)))

        test, train = test_dataset.random_split([0.8, 0.2], split_by='sequence')
        self.assertEqual(len(test.indices), len(set(test.indices)))
        self.assertEqual(len(train.indices), len(set(train.indices)))

    def test_ignore_filenames_should_ignore_correct_images(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)

        images_to_ignore = [
            'test_data_files/image/200801/2008041300-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041301-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041302-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041303-200801-MTS1-1.h5',
            'test_data_files/image/200801/2008041304-200801-MTS1-1.h5'
        ]
        images_to_ignore_set = set(images_to_ignore)

        # All images are present
        self.assertTrue(len(test_dataset) == 428)
        image_filenames = []
        for i in range(len(test_dataset)):
            image_filenames.append(test_dataset.get_image_from_idx(i).filepath())
        all_image_filenames = set(image_filenames)
        self.assertTrue(len(all_image_filenames) == 428)

        # Ensure that all the to ignore images are currently present
        self.assertEqual(5, len(images_to_ignore_set.intersection(all_image_filenames)))
        images_to_ignore = [
            '2008041300-200801-MTS1-1.h5',
            '2008041301-200801-MTS1-1.h5',
            '2008041302-200801-MTS1-1.h5',
            '2008041303-200801-MTS1-1.h5',
            '2008041304-200801-MTS1-1.h5'
        ]
        #
        # Create new dataset ignoring those images
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             ignore_list=images_to_ignore,
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertTrue(len(test_dataset) == 423)
        image_filenames = []
        for i in range(len(test_dataset)):
            image_filenames.append(test_dataset.get_image_from_idx(i).filepath())
        all_image_filenames = set(image_filenames)
        self.assertTrue(len(all_image_filenames) == 423)

        # Ensure that all the to ignore images are not present
        self.assertEqual(0, len(images_to_ignore_set.intersection(all_image_filenames)))

    def test_return_images_from_season(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                            "test_data_files/metadata.json",
                                            'season',
                                            split_dataset_by='image',
                                            verbose=False)

        filenames = {'test_data_files/image/202222/2022102600-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102601-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102602-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102603-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102604-202222-HMW8-1.h5'}

        season_images = test_dataset.images_from_season(2022)
        self.assertEqual(len(filenames), len(season_images))

        season_images = test_dataset.images_from_season(2008)
        self.assertEqual(len(season_images), 180)

    def test_return_images_from_season(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'year',
                                             split_dataset_by='image',
                                             verbose=False)
        season_subset = test_dataset.images_from_seasons([1979, 2022])
        self.assertEqual(len(season_subset), 55)

    def test_get_seq_ids_from_season(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'year',
                                             split_dataset_by='image',
                                             verbose=False)
        ids = test_dataset.get_seq_ids_from_season(2008)
        ids.sort()
        self.assertEqual(['200801', '200802'], ids)

    def test_images_from_sequence(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)

        filenames = {'test_data_files/image/202222/2022102600-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102601-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102602-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102603-202222-HMW8-1.h5',
                     'test_data_files/image/202222/2022102604-202222-HMW8-1.h5'}

        seq_images = test_dataset.images_from_sequence('202222')
        self.assertEqual(len(filenames), len(seq_images))

    def test_images_from_sequences(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        seq_subset = test_dataset.images_from_sequences(['197918', '202222'])
        self.assertEqual(len(seq_subset), 55)

    def test_image_objects_from_sequence(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        should_be = [test_dataset.get_image_from_idx(423),
                     test_dataset.get_image_from_idx(424),
                     test_dataset.get_image_from_idx(425),
                     test_dataset.get_image_from_idx(426),
                     test_dataset.get_image_from_idx(427)]

        img_list = test_dataset.image_objects_from_sequence('202222')
        for i in range(len(should_be)):
            self.assertEqual(img_list[i], should_be[i])

    def test_image_objects_from_season(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        image_list = test_dataset.image_objects_from_season(2022)
        should_be = test_dataset.image_objects_from_sequence('202222')
        for i in range(len(should_be)):
            self.assertEqual(image_list[i], should_be[i])

    def test_images_as_tensor(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             spectrum='infrared',
                                             split_dataset_by='image',
                                             verbose=False)

        img_1 = test_dataset.get_image_from_idx(0).image()
        img_2 = test_dataset.get_image_from_idx(5).image()
        img_3 = test_dataset.get_image_from_idx(15).image()
        should_be = torch.Tensor([img_1, img_2, img_3])

        img_tensor = test_dataset.images_as_tensor([0, 5, 15])
        self.assertTrue(torch.equal(img_tensor, should_be))

    def test_labels_as_tensor(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             spectrum='infrared',
                                             split_dataset_by='image',
                                             verbose=False)

        label_1 = test_dataset.get_image_from_idx(0).grade()
        label_2 = test_dataset.get_image_from_idx(5).grade()
        label_3 = test_dataset.get_image_from_idx(40).grade()
        should_be = torch.Tensor([label_1, label_2, label_3])
        label_tensor = test_dataset.labels_as_tensor([0, 5, 40], 'grade')
        self.assertTrue(torch.equal(label_tensor, should_be))


    def test_get_num_sequences(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertEqual(5, test_dataset.get_number_of_sequences())

        test_dataset = DigitalTyphoonDataset("dummy_test_data/image/", "dummy_test_data/metadata/",
                                             "dummy_test_data/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertEqual(0, test_dataset.get_number_of_sequences())

    def test_sequence_exists(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertTrue(test_dataset.sequence_exists('202222'))
        self.assertTrue(test_dataset.sequence_exists('197918'))
        self.assertTrue(test_dataset.sequence_exists('200801'))
        self.assertTrue(test_dataset.sequence_exists('200802'))
        self.assertTrue(test_dataset.sequence_exists('201323'))
        self.assertFalse(test_dataset.sequence_exists('201324'))

    def test_seq_indices_to_total_indices(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        should_be = [50, 51, 52, 53, 54]
        sequence = test_dataset._get_seq_from_seq_str('200801')
        return_indices = test_dataset.seq_indices_to_total_indices(sequence)
        self.assertEqual(should_be, return_indices)

    def test_get_list_of_seq_obj(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertEqual(5, len(test_dataset._get_list_of_sequence_objs()))
        sequence_names = [seq.sequence_str for seq in test_dataset._get_list_of_sequence_objs()]
        self.assertEqual(['197918', '200801', '200802', '201323', '202222'], sequence_names)

    def test_assign_all_images_dataset_idx(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        seq_count = {'200801':0,
                     '200802':0,
                     '197918':0,
                     '201323':0,
                     '202222':0}

        self.assertEqual(428, len(test_dataset._image_idx_to_sequence))
        for i in range(len(test_dataset._image_idx_to_sequence)):
            seq_count[test_dataset._image_idx_to_sequence[i].get_sequence_str()] += 1

        counts_should_be = {'200801':5,
                             '200802':175,
                             '197918':50,
                             '201323':193,
                             '202222':5}
        self.assertEqual(counts_should_be, seq_count)

    def test_get_seq_from_seq_str(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        seq_strs = ['200801', '200802','197918', '201323', '202222']
        for seq in seq_strs:
            self.assertEqual(seq, test_dataset._get_seq_from_seq_str(seq).get_sequence_str())

    def test_get_seasons(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        season_list = [1979, 2008, 2013, 2022]
        dataset_season_list = test_dataset.get_seasons()
        self.assertEqual(season_list, dataset_season_list)

    def test_find_seq_str_from_image(self):
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             'grade',
                                             split_dataset_by='image',
                                             verbose=False)
        self.assertEqual('197918', test_dataset._find_sequence_str_from_image_index(0))
        self.assertEqual('200801', test_dataset._find_sequence_str_from_image_index(51))

    def test_get_image_from_idx(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', verbose=False)
        read_in_image = test_dataset.get_image_from_idx(4)

        correct_image = test_dataset.sequences[0].get_image_at_idx(4)
        self.assertEqual(read_in_image, correct_image)

    def test_get_image_from_idx_as_numpy(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', spectrum='infrared', verbose=False)
        read_in_image_array = test_dataset._get_image_from_idx_as_numpy(54)
        first_values = [296.30972999999994, 296.196816, 296.083902, 296.083902, 296.083902]
        last_values = [285.80799, 284.56569, 285.18684, 281.78588999999994, 282.0398488235294]

        for i in range(len(first_values)):
            if read_in_image_array[0][i] != first_values[i]:
                self.fail(f'Value produced was {read_in_image_array[0][i]}. Should be {first_values[i]}')
            if read_in_image_array[-1][-i - 1] != last_values[-i - 1]:
                self.fail(f'Value produced was {read_in_image_array[-1][-i - 1]}. Should be {last_values[-i - 1]}')

    def test_get_nonempty_seasons_and_sequences(self):
        def filter_func(image):
            return image.year() != 2008

        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', filter_func=filter_func, verbose=False)
        self.assertEqual(test_dataset.get_number_of_nonempty_sequences(), 3)
        self.assertEqual(test_dataset.get_nonempty_seasons(), 3)

        def filter_func(image):
            return image.sequence_id() != '200801'

        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', filter_func=filter_func, verbose=False)
        self.assertEqual(test_dataset.get_number_of_nonempty_sequences(), 4)
        self.assertEqual(test_dataset.get_nonempty_seasons(), 4)

    def test_random_split_doesnt_add_empty_sequences(self):
        def filter_func(image):
            return image.year() != 2008
        #
        test_dataset = DigitalTyphoonDataset("test_data_files/image/", "test_data_files/metadata/",
                                             "test_data_files/metadata.json",
                                             "grade",
                                             get_images_by_sequence=True,
                                             split_dataset_by='sequence',
                                             spectrum='infrared',
                                             filter_func=filter_func,
                                             verbose=False)

        bucket_1, bucket_2 = test_dataset.random_split([0.7, 0.3], split_by='sequence')
        should_contain = {'197918', '201323', '202222'}
        does_contain = set()
        for idx in bucket_1.indices:
            self.assertNotEqual(test_dataset.get_image_from_idx(int(idx)).year(), 2008)
            does_contain.add(test_dataset.get_ith_sequence(int(idx)).get_sequence_str())
        for idx in bucket_2.indices:
            self.assertNotEqual(test_dataset.get_image_from_idx(int(idx)).year(), 2008)
            does_contain.add(test_dataset.get_ith_sequence(int(idx)).get_sequence_str())
        self.assertEqual(should_contain, does_contain)

        bucket_1, bucket_2 = test_dataset.random_split([0.7, 0.3], split_by='season')
        should_contain = {'197918', '201323', '202222'}
        does_contain = set()
        for idx in bucket_1.indices:
            self.assertNotEqual(test_dataset.get_image_from_idx(int(idx)).year(), 2008)
            does_contain.add(test_dataset.get_ith_sequence(int(idx)).get_sequence_str())
        for idx in bucket_2.indices:
            self.assertNotEqual(test_dataset.get_image_from_idx(int(idx)).year(), 2008)
            does_contain.add(test_dataset.get_ith_sequence(int(idx)).get_sequence_str())
        self.assertEqual(should_contain, does_contain)


    def test_delete_all_sequence(self):
        test_dataset = DigitalTyphoonDataset('test_data_files/image/', 'test_data_files/metadata/',
                                             'test_data_files/metadata.json', 'grade', verbose=False)
        self.assertEqual(5, test_dataset.get_number_of_sequences())

        test_dataset._delete_all_sequences()
        self.assertEqual(0, test_dataset.get_number_of_sequences())
        self.assertEqual(test_dataset.sequences, [])
        self.assertEqual(test_dataset._sequence_str_to_seq_idx, {})
        self.assertEqual(test_dataset._image_idx_to_sequence, {})
        self.assertEqual(test_dataset._seq_str_to_first_total_idx, {})
        self.assertEqual(test_dataset.number_of_sequences, 0)
        self.assertEqual(test_dataset.number_of_original_images, 0)
        self.assertEqual(test_dataset.number_of_images, 0)
