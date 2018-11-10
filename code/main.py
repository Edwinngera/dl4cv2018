from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset

from dlvc import ops, batches
import numpy as np
from dlvc.models import knn

if __name__ == "__main__":
    dataset = PetsDataset('../data/cifar-10-batches-py',
                          Subset.TEST)

    print('Found dataset directory')
    print('{} samples'.format(len(dataset)))

    test_sample = dataset[1]

    print('Index of test sample: {.idx}'.format(test_sample))
    print('Label of test sample: {}'.format(test_sample.label))
    #cv2.imshow('Sample Image', test_sample.data)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    op = ops.chain([
        ops.vectorize(),
        ops.type_cast(np.float32)
    ])

    #The number of training batches is 1 if the batch size is set to the number of samples in the dataset
    generator = batches.BatchGenerator(dataset, len(dataset), True, op)
    print(len(generator))

    #The number of training batches is 16 if the batch size is set to 500
    generator = batches.BatchGenerator(dataset, 500, True, op)
    print(len(generator))

    #The data and label shapes are (500, 3072) and (500,), respectively, unless for the last batch
    #The data type is always np.float32 and the label type is integral (one of the np.int and np.uint variants)
    #generator = batches.BatchGenerator(dataset, 20, False, op)
    for i in iter(generator):
        pass
    #    print(i.idx.shape)
    #    print(i.data)
    #    print(i.data.dtype)
    #    print(i.label.shape)
    #    print(i.label.dtype)

    knn_cl = knn.KnnClassifier(k=15, input_dim=len(op(dataset[1][1])), num_classes=2)
    print( knn_cl.input_shape(), knn_cl.output_shape())

    knn_cl.train(i.data, i.label)

    test_sample = np.array((op(dataset[2].data), op(dataset[3].data)))
    print(test_sample.shape)

    scores = knn_cl.predict(test_sample )
    print(scores)



