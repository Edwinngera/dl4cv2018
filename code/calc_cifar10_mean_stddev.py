import numpy as np

from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc import ops, batches

dataset = PetsDataset('../data/cifar-10-batches-py', Subset.TRAINING)

op = ops.chain([ops.mul(1 / 255), ops.type_cast(np.float32)])

batch_generator = batches.BatchGenerator(dataset, 7959, True, op)

training_images = []

for batch in batch_generator:
    training_images.append(batch.data)

training_images = np.array(training_images, dtype=np.float32)
training_images = training_images.reshape(training_images.shape[1:])

train_mean = np.mean(training_images, axis=(0, 1, 2))
train_std = np.std(training_images, axis=(0, 1, 2))

print(train_mean, train_std)