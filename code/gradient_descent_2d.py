
import cv2
import numpy as np

import os
import time
import math
from collections import namedtuple

Vec2 = namedtuple('Vec2', ['x1', 'x2'])

class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self._fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self._fn = self._fn.astype(np.float32)
        self._fn /= (2**16-1)

        self.height = self._fn.shape[0]
        self.width = self._fn.shape[1]

    def visualize(self) -> np.ndarray:
        '''
        Return a visualization as a color image.
        Use the result to visualize the progress of gradient descent.
        '''

        vis = self._fn - self._fn.min()
        vis /= self._fn.max()
        vis *= 255
        vis = vis.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_HOT)

        return vis

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        if loc.x1 <0 and loc.x1 >= self.width:
            raise ValueError('x1 is out of range')

        if loc.x2 <0 and loc.x2 >= self.height:
            raise ValueError('x2 is out of range')
        
        x1_low = int(math.floor(loc.x1))
        x1_high = int(math.ceil(loc.x1))

        x2_low = int(math.floor(loc.x2))
        x2_high = int(math.ceil(loc.x2))

        # TODO: interpolate
        return self._fn[x2_low][x1_low]

def grad(fn: Fn, loc: Vec2, eps: float) -> Vec2:
    '''
    Compute the numerical gradient of a 2D function fn at location loc,
    using the given epsilon. See lecture 5 slides.
    Raises ValueError if loc is out of bounds of fn or if eps <= 0.
    '''

    if eps <= 0:
        raise ValueError('eps has to be positive')

    if loc.x1 <0 and loc.x1 >= fn.width:
        raise ValueError('x1 is out of range')

    if loc.x2 <0 and loc.x2 >= fn.height:
        raise ValueError('x2 is out of range')

    dx1 = ( fn(Vec2(loc.x1 + eps, loc.x2)) - fn(Vec2(loc.x1 - eps, loc.x2)) ) / (2 * eps)
    dx2 = ( fn(Vec2(loc.x1, loc.x2 + eps)) - fn(Vec2(loc.x1, loc.x2 - eps)) ) / (2 * eps)

    return Vec2(dx1, dx2)

if __name__ == '__main__':
    # parse args

    import argparse

    parser = argparse.ArgumentParser(description='Perform gradient descent on a 2D function.')
    parser.add_argument('fpath', help='Path to a PNG file encoding the function')
    parser.add_argument('sx1', type=float, help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon for computing numeric gradients')
    parser.add_argument('--step_size', type=float, default=10.0, help='Step size')
    parser.add_argument('--beta', type=float, default=0, help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true', help='Use Nesterov momentum')
    args = parser.parse_args()

    # init

    fn = Fn(args.fpath)
    vis = fn.visualize()
    loc = Vec2(args.sx1, args.sx2)
    eps = args.eps
    step_size = args.step_size
    beta = args.beta
    use_nesterov = args.nesterov

    # perform gradient descent

    _velocity = Vec2(0, 0)

    min_step = 1e-6

    while True:
        gradient_eval_loc = loc

        if use_nesterov:
            gradient_eval_loc = Vec2(loc.x1 + _velocity.x1, loc.x2 + _velocity.x2)
        
        _gradient = grad(fn, gradient_eval_loc, eps)

        _velocity = Vec2(_velocity.x1 * beta - step_size * _gradient.x1, _velocity.x2 * beta- step_size * _gradient.x2)
        new_loc = Vec2(loc.x1 + _velocity.x1, loc.x2 + _velocity.x2)

        cv2.line(vis, (int(loc.x1), int(loc.x2)), (int(new_loc.x1), int(new_loc.x2)), (255, 255, 0), 2)
        
        loc_delta_dist = math.sqrt(math.pow(loc.x1 - new_loc.x1, 2) * math.pow(loc.x2 - new_loc.x2, 2))
        loc = new_loc

        cv2.imshow('Progress', vis)
        cv2.waitKey(50)  # 20 fps, tune according to your liking

        print(loc_delta_dist)
        if loc_delta_dist <= min_step:
            cv2.circle(vis, (int(new_loc.x1), int(new_loc.x2)), 4, (255, 255, 0), -1)
            cv2.imshow('Progress', vis)
            break

    print('Finished', loc, _gradient)
    print('Press any key to quit')
    cv2.waitKey(0)
    