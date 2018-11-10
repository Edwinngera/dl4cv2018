from ..dataset import Sample, Subset, ClassificationDataset

import os
import numpy as np


LABEL_CAT = 0
LABEL_DOG = 1


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        if not os.path.exists(fdir):
            raise ValueError('"{}" does not exist'.format(fdir))

        # Load labels
        path = os.path.join(fdir, 'batches.meta')
        if not os.path.exists(path):
            raise ValueError('"{}" does not exist'.format(path))

        meta = unpickle(path)
        assert meta is not None

        label_names = meta[b'label_names']
        assert label_names is not None

        assert b'cat' in label_names
        assert b'dog' in label_names

        cat_index = label_names.index(b'cat')
        dog_index = label_names.index(b'dog')

        subset_mapping = {
            Subset.TRAINING:    ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'],
            Subset.VALIDATION:  ['data_batch_5'],
            Subset.TEST:  ['test_batch'],
        }

        filenames = subset_mapping[subset]

        self.samples = []

        for filename in filenames:
            path = os.path.join(fdir, filename)

            if not os.path.exists(path):
                raise ValueError('"{}" does not exist'.format(path))

            batch = unpickle(path)

            labels = batch[b'labels']
            data = batch[b'data']

            for i, label in enumerate(labels):
                sample_label = None

                if label == cat_index:
                    sample_label = LABEL_CAT
                elif label == dog_index:
                    sample_label = LABEL_DOG
                else:
                    continue

                img = np.zeros((32, 32, 3), dtype=np.uint8)
                rgb = np.reshape(data[i], (3, 32, 32))

                # Ordering should be BGR
                img[:, :, 0] = rgb[2] # blue channel
                img[:, :, 1] = rgb[1] # green channel
                img[:, :, 2] = rgb[0] # red channel

                self.samples.append(Sample(idx=len(self.samples),
                                           data=img,
                                           label=sample_label))

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''
        return self.samples[idx]

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        return 2
