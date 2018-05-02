'''
Various photometric image transformations, both deterministic and probabilistic.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
import cv2

class ConvertColor:
    '''
    Converts images between RGB, HSV and grayscale color spaces. This is just a wrapper
    around `cv2.cvtColor()`.
    '''
    def __init__(self, current='RGB', to='HSV', keep_3ch=True):
        '''
        Arguments:
            current (str, optional): The current color space of the images. Can be
                one of 'RGB' and 'HSV'.
            to (str, optional): The target color space of the images. Can be one of
                'RGB', 'HSV', and 'GRAY'.
            keep_3ch (bool, optional): Only relevant if `to == GRAY`.
                If `True`, the resulting grayscale images will have three channels.
        '''
        if not ((current in {'RGB', 'HSV'}) and (to in {'RGB', 'HSV', 'GRAY'})):
            raise NotImplementedError
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None):
        if self.current == 'RGB' and self.to == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'RGB' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        elif self.current == 'HSV' and self.to == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        elif self.current == 'HSV' and self.to == 'GRAY':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2GRAY)
            if self.keep_3ch:
                image = np.stack([image] * 3, axis=-1)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertDataType:
    '''
    Converts images represented as Numpy arrays between `uint8` and `float32`.
    Serves as a helper for certain photometric distortions. This is just a wrapper
    around `np.ndarray.astype()`.
    '''
    def __init__(self, to='uint8'):
        '''
        Arguments:
            to (string, optional): To which datatype to convert the input images.
                Can be either of 'uint8' and 'float32'.
        '''
        if not (to == 'uint8' or to == 'float32'):
            raise ValueError("`to` can be either of 'uint8' or 'float32'.")
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertTo3Channels:
    '''
    Converts 1-channel and 4-channel images to 3-channel images. Does nothing to images that
    already have 3 channels. In the case of 4-channel images, the fourth channel will be
    discarded.
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:,:,:3]
        if labels is None:
            return image
        else:
            return image, labels

class Hue:
    '''
    Changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer in the closed interval `[-180, 180]` that determines the hue change, where
                a change by integer `delta` means a change by `2 * delta` degrees. Read up on the HSV color format
                if you need more information.
        '''
        if not (-180 <= delta <= 180): raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels

class RandomHue:
    '''
    Randomly changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, max_delta=18, prob=0.5):
        '''
        Arguments:
            max_delta (int): An integer in the closed interval `[0, 180]` that determines the maximal absolute
                hue change.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Saturation:
    '''
    Changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines saturation change, where
                values less than one result in less saturation and values greater than one result
                in more saturation.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomSaturation:
    '''
    Randomly changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                saturation change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                saturation change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Brightness:
    '''
    Changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer, the amount to add to or subtract from the intensity
                of every pixel.
        '''
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomBrightness:
    '''
    Randomly changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=-84, upper=84, prob=0.5):
        '''
        Arguments:
            lower (int, optional): An integer, the lower bound for the random brightness change.
            upper (int, optional): An integer, the upper bound for the random brightness change.
                Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Contrast:
    '''
    Changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines contrast change, where
                values less than one result in less contrast and values greater than one result
                in more contrast.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomContrast:
    '''
    Randomly changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                contrast change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                contrast change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Gamma:
    '''
    Changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, gamma):
        '''
        Arguments:
            gamma (float): A float greater than zero that determines gamma change.
        '''
        if gamma <= 0.0: raise ValueError("It must be `gamma > 0`.")
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma
        # Build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values.
        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, labels=None):
        image = cv2.LUT(image, table)
        if labels is None:
            return image
        else:
            return image, labels

class RandomGamma:
    '''
    Randomly changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, lower=0.25, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                gamma change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                gamma change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            gamma = np.random.uniform(self.lower, self.upper)
            change_gamma = Gamma(gamma=gamma)
            return change_gamma(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class HistogramEqualization:
    '''
    Performs histogram equalization on HSV images.

    Importat: Expects HSV input.
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        if labels is None:
            return image
        else:
            return image, labels

class RandomHistogramEqualization:
    '''
    Randomly performs histogram equalization on HSV images. The randomness only refers
    to whether or not the equalization is performed.

    Importat: Expects HSV input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        self.equalize = HistogramEqualization()

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.equalize(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class ChannelSwap:
    '''
    Swaps the channels of images.
    '''
    def __init__(self, order):
        '''
        Arguments:
            order (tuple): A tuple of integers that defines the desired channel order
                of the input images after the channel swap.
        '''
        self.order = order

    def __call__(self, image, labels=None):
        image = image[:,:,self.order]
        if labels is None:
            return image
        else:
            return image, labels

class RandomChannelSwap:
    '''
    Randomly swaps the channels of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        # All possible permutations of the three image channels except the original order.
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5) # There are 6 possible permutations.
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels
