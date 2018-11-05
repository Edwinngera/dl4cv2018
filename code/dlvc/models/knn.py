
from ..model import Model

import numpy as np

class KnnClassifier(Model):
    '''
    k nearest neighbors classifier.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, k: int, input_dim: int, num_classes: int):
        '''
        Ctor.
        k is the number of nearest neighbors to consult (>= 1).
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        '''

        if k < 1:
            raise ValueError ("Number of neighbors must be at least 1")
        if input_dim < 1:
            raise ValueError ("Length of input vectors must be at least 1")
        if num_classes < 2:
            raise ValueError ("Number of classes must be at least 2")

        self.k = k
        self.input_dim = input_dim
        self.num_classes = num_classes


    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return (0, self.input_dim)


    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes,)


    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        As training simply entails storing the data, the model is reset each time this method is called.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns 0 as there is no training loss to compute.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        if data.dtype != np.float32:
            raise TypeError ("data: datatype 'float32' expected, but %s given" % data.dtype)

        if len(data.shape) != 2 or data.shape[1] != self.input_dim:
            raise TypeError ("data: Expected input shape (n, %s)" % self.input_dim)

        #check LABELS
        if labels.shape != (data.shape[0],):
            raise TypeError ("labels: Expected input shape (%s, )" % data.shape[0])

        if not (np.issubdtype(labels.dtype, np.integer) and max(labels)<self.num_classes):
            raise ValueError ("labels must have integral values between 0 and num_classes - 1")

        self.traindata = data
        self.labels = labels

        ###return 0 ?????


    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        if not (len(data.shape) == 2 and data.shape[1] == self.traindata.shape[1]):
            pass
        elif not (len(data.shape) == 1 and data.shape[0] == self.traindata.shape[1]):
            pass
        else:
            raise ValueError ("data: shape %s must compatible with input_shape() %s" % ( data.shape, self.input_shape()))

        if len(data.shape) == 2:
            label_scores = np.zeros((data.shape[0],self.num_classes)) #nr of samples, nr of classes
        elif len(data.shape) == 1:
            label_scores = np.zeros((1,self.num_classes))
            data = np.array(data).reshape(1,-1)

        distances = []

        #loop over all test rows
        for idx_sample, sample in enumerate(data):

            # find the k nearest training image to the the sample test image
            # using the L1 distance
            distances =  np.sum(np.abs(self.traindata - sample), axis=1)

            #get the indices for the k nearest distances
            neighbor_indices = np.argpartition(distances.ravel(), self.k)[:self.k]

            #get labels for k nearest neighbors
            neighbor_labels = self.labels[neighbor_indices].tolist()

            #count the appearance of the class labels
            label_count=[]
            for label_nr  in range(0,self.num_classes):
                label_count.append(neighbor_labels.count(label_nr))

            #calculate soft max
            e_x = np.exp(label_count - np.max(label_count))
            label_scores[idx_sample,:] = e_x / e_x.sum(axis=0)

        return label_scores
