'''
Includes:
* A batch generator for SSD model training and inference which can perform online data agumentation
* An offline image processor that saves processed images and adjusted labels to disk

Copyright (C) 2017 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import division
import numpy as np
import cv2

class ConvertColor:
    '''
    Converts an image between RGB and HSV color spaces. This is just a wrapper
    around `cv2.cvtColor()`.
    '''
    def __init__(self, current='RGB', to='HSV'):
        if not ((current == 'RGB' and to == 'HSV') or (current == 'HSV' and to == 'RGB'):
            raise NotImplementedError
        self.current = current
        self.to = to

    def __call__(self, image, labels=None):
        if self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertDataType:
    '''
    Converts an image represented as a Numpy array between `uint8` and `float`.
    This is just a wrapper around `np.ndarray.astype()`.
    '''
    def __init__(self, to='uint8'):
        if not (to == 'uint8' or to == 'float'):
            raise ValueError("`to` can be either of 'uint8' or 'float'.")
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float)
        if labels is None:
            return image
        else:
            return image, labels

class RandomSaturation:
    '''
    Applies random saturation to an image.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            saturation = np.random.uniform(self.lower, self.upper)
            image[:,:,1] = np.clip(image[:,:,1] * saturation, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomHue:
    '''
    Randomly changes the hue of an image.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, max_delta=18, prob=0.5):
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            hue = np.random.uniform(-self.max_delta, self.max_delta)
            image[:, :, 0] = (image[:, :, 0] + hue) % 180.0
        if labels is None:
            return image
        else:
            return image, labels

class RandomChannelSwap:
    '''
    Randomly swaps the channels of an image.

    Important: Expects RGB input.
    '''
    def __init__(self, prob=0.5):
        self.prob = prob
        # All possible permutations of the three image channels except the original order.
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5) # There are 6 possible permutations.
            permutation = self.permutations[i]
            image = image[:,:,permutation]
        if labels is None:
            return image
        else:
            return image, labels

class RandomContrast:
    '''
    Randomly changes the contrast of an image.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            factor = np.random.uniform(self.lower, self.upper)
            image = np.clip(127.5 + factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomBrightness:
    '''
    Randomly changes the brightness of an image.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=-84, upper=84, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            delta = np.random.uniform(self.lower, self.upper)
            image = np.clip(image + delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels
