from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import cv2

if __name__ == "__main__":
    dataset = PetsDataset('../data/cifar-10-batches-py',
                          Subset.TEST)

    print('Found dataset directory')
    print(f'{len(dataset)} samples')

    test_sample = dataset[1]

    print(f'Index of test sample: {test_sample.idx}')
    print(f'Label of test sample: {test_sample.label}')
    cv2.imshow('Sample Image', test_sample.data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
