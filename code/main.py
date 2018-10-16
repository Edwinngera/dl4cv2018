from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
import cv2

if __name__ == "__main__":
    dataset = PetsDataset('../data/cifar-10-batches-py',
                          Subset.TRAINING)

    print('Found dataset directory')
    print(f'{len(dataset)} samples')

    test_img = dataset[1]
    cv2.imshow('Sample Image', test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
