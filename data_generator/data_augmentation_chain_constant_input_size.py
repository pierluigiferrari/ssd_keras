'''
The data augmentation operations of the original SSD implementation.

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

from data_generator.object_detection_2d_photometric_ops import ConvertColor, ConvertDataType, ConvertTo3Channels, RandomBrightness, RandomContrast, RandomHue, RandomSaturation
from data_generator.object_detection_2d_geometric_ops import RandomFlip, RandomTranslate, RandomScale
from data_generator.object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class DataAugmentationConstantInputSize:
    '''
    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.

    Important: This augmentation chain is suitable for constant-size images only.
    '''

    def __init__(self,
                 random_brightness=(-48, 48, 0.5),
                 random_contrast=(0.5, 1.8, 0.5),
                 random_saturation=(0.5, 1.8, 0.5),
                 random_hue=(18, 0.5),
                 random_flip=0.5,
                 random_translate=((0.03,0.5), (0.03,0.5), 0.5),
                 random_scale=(0.5, 2.0, 0.5),
                 n_trials_max=3,
                 clip_boxes=True,
                 overlap_criterion='area',
                 bounds_box_filter=(0.3, 1.0),
                 bounds_validator=(0.5, 1.0),
                 n_boxes_min=1,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        if (random_scale[0] >= 1) or (random_scale[1] <= 1):
            raise ValueError("This sequence of transformations only makes sense if the minimum scaling factor is <1 and the maximum scaling factor is >1.")

        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.overlap_criterion = overlap_criterion
        self.bounds_box_filter = bounds_box_filter
        self.bounds_validator = bounds_validator
        self.n_boxes_min = n_boxes_min
        self.background = background
        self.labels_format = labels_format

        # Determines which boxes are kept in an image after the transformations have been applied.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=True,
                                    check_degenerate=True,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds_box_filter,
                                    min_area=16,
                                    labels_format=self.labels_format)

        # Determines whether the result of the transformations is a valid training image.
        self.image_validator = ImageValidator(overlap_criterion=self.overlap_criterion,
                                              bounds=self.bounds_validator,
                                              n_boxes_min=self.n_boxes_min,
                                              labels_format=self.labels_format)

        # Utility distortions
        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels() # Make sure all images end up having 3 channels.

        # Photometric transformations
        self.random_brightness = RandomBrightness(lower=random_brightness[0], upper=random_brightness[1], prob=random_brightness[2])
        self.random_contrast = RandomContrast(lower=random_contrast[0], upper=random_contrast[1], prob=random_contrast[2])
        self.random_saturation = RandomSaturation(lower=random_saturation[0], upper=random_saturation[1], prob=random_saturation[2])
        self.random_hue = RandomHue(max_delta=random_hue[0], prob=random_hue[1])

        # Geometric transformations
        self.random_flip = RandomFlip(dim='horizontal', prob=random_flip, labels_format=self.labels_format)
        self.random_translate = RandomTranslate(dy_minmax=random_translate[0],
                                                dx_minmax=random_translate[1],
                                                prob=random_translate[2],
                                                clip_boxes=self.clip_boxes,
                                                box_filter=self.box_filter,
                                                image_validator=self.image_validator,
                                                n_trials_max=self.n_trials_max,
                                                background=self.background,
                                                labels_format=self.labels_format)
        self.random_zoom_in = RandomScale(min_factor=1.0,
                                          max_factor=random_scale[1],
                                          prob=random_scale[2],
                                          clip_boxes=self.clip_boxes,
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          n_trials_max=self.n_trials_max,
                                          background=self.background,
                                          labels_format=self.labels_format)
        self.random_zoom_out = RandomScale(min_factor=random_scale[0],
                                           max_factor=1.0,
                                           prob=random_scale[2],
                                           clip_boxes=self.clip_boxes,
                                           box_filter=self.box_filter,
                                           image_validator=self.image_validator,
                                           n_trials_max=self.n_trials_max,
                                           background=self.background,
                                           labels_format=self.labels_format)

        # If we zoom in, do translation before scaling.
        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_translate,
                          self.random_zoom_in,
                          self.random_flip]

        # If we zoom out, do scaling before translation.
        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.convert_to_float32,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.random_zoom_out,
                          self.random_translate,
                          self.random_flip]

    def __call__(self, image, labels=None):

        self.random_translate.labels_format = self.labels_format
        self.random_zoom_in.labels_format = self.labels_format
        self.random_zoom_out.labels_format = self.labels_format
        self.random_flip.labels_format = self.labels_format

        # Choose sequence 1 with probability 0.5.
        if np.random.choice(2):

            if not (labels is None):
                for transform in self.sequence1:
                    image, labels = transform(image, labels)
                return image, labels
            else:
                for transform in self.sequence1:
                    image = transform(image)
                return image
        # Choose sequence 2 with probability 0.5.
        else:

            if not (labels is None):
                for transform in self.sequence2:
                    image, labels = transform(image, labels)
                return image, labels
            else:
                for transform in self.sequence2:
                    image = transform(image)
                return image
