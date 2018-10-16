from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset

if __name__ == "__main__":
    datasets = PetsDataset('../data/cifar-10-batches-py',
                           Subset.TRAINING)
    print('Found dataset directory')
