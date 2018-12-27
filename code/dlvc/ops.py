
import numpy as np

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)
    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    return np.ravel


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.transpose(2, 0, 1)

    return op
    # return np.transpose(2,0,1)


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    return np.transpose(1, 2, 0)


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample + val

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample * val

    return op

def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if np.random.rand() >= 0.5:
            return np.flip(sample, axis=1)
        
        return sample

    return op

def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # TODO implement
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.pad.html will be helpful

    def op(sample: np.ndarray) -> np.ndarray:
        sample = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
        height, width = sample.shape[0], sample.shape[1]

        rand_y = np.random.randint( (height - sz) + 1)
        rand_x = np.random.randint( (width - sz) + 1)

        cropped = sample[rand_y:rand_y+sz, rand_x:rand_x+sz]
        return cropped

    return op

def normalize(mean: np.ndarray, std: np.ndarray) -> Op:
    '''
    Mean/stddev normalization for multi-dimensional arrays
    See https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize
    '''
    if mean.shape != std.shape:
        raise ValueError('mean and std shapes must match')

    def op(sample: np.ndarray) -> np.ndarray:
        sample -= mean
        sample /= std
        return sample

    return op

def resize(new_size: np.ndarray):
    '''
    Rescale image to given size
    See http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
    '''
    from skimage.transform import resize

    def op(sample: np.ndarray) -> np.ndarray:
        return resize(sample, new_size)

    return op
