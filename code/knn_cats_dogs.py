from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.test import Accuracy
from dlvc.models import knn
from dlvc import ops, batches

import time
import numpy as np

INPUT_DIM = 3072
NUM_CLASSES = 2


def load_dataset(subset: Subset) -> batches.BatchGenerator:
    dataset = PetsDataset('../data/cifar-10-batches-py', subset)

    op = ops.chain([
        ops.vectorize(),
        ops.type_cast(np.float32)
    ])

    return batches.BatchGenerator(dataset, len(dataset), True, op)

if __name__ == "__main__":
    training_batch = load_dataset(Subset.TRAINING)
    validation_batch = load_dataset(Subset.VALIDATION)
    test_batch = load_dataset(Subset.TEST)

    training_data = next(iter(training_batch))
    validation_data = next(iter(validation_batch))
    test_data = next(iter(test_batch))

    measure = Accuracy()
    accuracies = []

    ks = [1, 5, 20, 50, 100]

    print(f'Starting grid search with k={ks}')
    print()
    for k in ks:
        print(f'Train classifier with k={k}')
        knn_cl = knn.KnnClassifier(k=k, input_dim=INPUT_DIM,
                                   num_classes=NUM_CLASSES)

        # train the classifier
        print('\tRun training... ', end='', flush=True)
        knn_cl.train(training_data.data, training_data.label)
        print('done')

        # test on validation set
        print('\tRun validation... ', end='', flush=True)
        t0 = time.time()
        predictions = knn_cl.predict(validation_data.data)
        print(f'done (took {time.time() - t0})')

        measure.reset()
        measure.update(predictions, validation_data.label)
        print(f'\tAccuracy: {measure.accuracy()}')
        print()

        accuracies.append(measure.accuracy())

    best_accuracy_index = np.argmax(accuracies)
    best_accuracy = accuracies[best_accuracy_index]
    best_k = ks[best_accuracy_index]

    print(f"Best accuracy '{best_accuracy}' with k='{best_k}'")
    print()
    print(f'Evaluating on test set with k={best_k}... ', end='', flush=True)
    t0 = time.time()
    predictions = knn_cl.predict(test_data.data)
    print(f'done (took {time.time() - t0})')

    measure.reset()
    measure.update(predictions, test_data.label)
    print(f'\tAccuracy: {measure.accuracy()}')
    print()
