'''
Various geometric image transformations for 2D object detection, both deterministic
and probabilistic.

Copyright (C) 2018 Pierluigi Ferrari

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

class Resize:
    '''
    Resize an image to a specified height and width in pixels.
    '''

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.out_height = height
        self.out_width = width
        self.interpolation = interpolation

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation)

        if labels is None:
            return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = labels[:, [ymin, ymax]] * (self.out_height / img_height)
            labels[:, [xmin, xmax]] = labels[:, [xmin, xmax]] * (self.out_width / img_width)
            return image, labels

class Flip:
    '''
    Flips an image horizontally or vertically.
    '''
    def __init__(self, dim='horizontal'):
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels

class RandomFlip:
    '''
    Randomly flips an image horizontally or vertically. The randomness only refers
    to whether or not the image will be flipped.
    '''
    def __init__(self, dim='horizontal', prob=0.5):
        self.dim = dim
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            flip = Flip(dim=self.dim)
            return flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Translate:
    '''
    Translates an image horizontally and/or vertically.
    '''
    def __init__(self, dy, dx, background=(0,0,0)):
        self.dy = dy
        self.dx = dx
        self.M = np.float32([[1, 0, dx],
                             [0, 1, dy]])
        self.background = background

    def __call__(self, image, labels=None):
        img_height, img_width = image.shape[:2]
        image = cv2.warpAffine(image,
                               M=self.M,
                               dsize=(img_width, img_height),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=self.background)
        if labels is None:
            return image
        else:
            return image, labels
