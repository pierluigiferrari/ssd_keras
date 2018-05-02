'''
A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
no "up" or "down" in the images.

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
from data_generator.object_detection_2d_geometric_ops import Resize, RandomFlip, RandomRotate
from data_generator.object_detection_2d_patch_sampling_ops import PatchCoordinateGenerator, RandomPatch
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator

class DataAugmentationSatellite:
    '''
    A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
    no "up" or "down" in the images.

    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.
    '''

    def __init__(self,
                 resize_height,
                 resize_width,
                 random_brightness=(-48, 48, 0.5),
                 random_contrast=(0.5, 1.8, 0.5),
                 random_saturation=(0.5, 1.8, 0.5),
                 random_hue=(18, 0.5),
                 random_flip=0.5,
                 random_rotate=([90, 180, 270], 0.5),
                 min_scale=0.3,
                 max_scale=2.0,
                 min_aspect_ratio = 0.8,
                 max_aspect_ratio = 1.25,
                 n_trials_max=3,
                 clip_boxes=True,
                 overlap_criterion='area',
                 bounds_box_filter=(0.3, 1.0),
                 bounds_validator=(0.5, 1.0),
                 n_boxes_min=1,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.overlap_criterion = overlap_criterion
        self.bounds_box_filter = bounds_box_filter
        self.bounds_validator = bounds_validator
        self.n_boxes_min = n_boxes_min
        self.background = background
        self.labels_format = labels_format

        # Determines which boxes are kept in an image after the transformations have been applied.
        self.box_filter_patch = BoxFilter(check_overlap=True,
                                          check_min_area=False,
                                          check_degenerate=False,
                                          overlap_criterion=self.overlap_criterion,
                                          overlap_bounds=self.bounds_box_filter,
                                          labels_format=self.labels_format)

        self.box_filter_resize = BoxFilter(check_overlap=False,
                                           check_min_area=True,
                                           check_degenerate=True,
                                           min_area=16,
                                           labels_format=self.labels_format)

        # Determines whether the result of the transformations is a valid training image.
        self.image_validator = ImageValidator(overlap_criterion=self.overlap_criterion,
                                              bounds=self.bounds_validator,
                                              n_boxes_min=self.n_boxes_min,
                                              labels_format=self.labels_format)

        # Utility transformations
        self.convert_to_3_channels  = ConvertTo3Channels() # Make sure all images end up having 3 channels.
        self.convert_RGB_to_HSV     = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB     = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32     = ConvertDataType(to='float32')
        self.convert_to_uint8       = ConvertDataType(to='uint8')
        self.resize                 = Resize(height=resize_height,
                                             width=resize_width,
                                             box_filter=self.box_filter_resize,
                                             labels_format=self.labels_format)

        # Photometric transformations
        self.random_brightness      = RandomBrightness(lower=random_brightness[0], upper=random_brightness[1], prob=random_brightness[2])
        self.random_contrast        = RandomContrast(lower=random_contrast[0], upper=random_contrast[1], prob=random_contrast[2])
        self.random_saturation      = RandomSaturation(lower=random_saturation[0], upper=random_saturation[1], prob=random_saturation[2])
        self.random_hue             = RandomHue(max_delta=random_hue[0], prob=random_hue[1])

        # Geometric transformations
        self.random_horizontal_flip = RandomFlip(dim='horizontal', prob=random_flip, labels_format=self.labels_format)
        self.random_vertical_flip   = RandomFlip(dim='vertical', prob=random_flip, labels_format=self.labels_format)
        self.random_rotate          = RandomRotate(angles=random_rotate[0], prob=random_rotate[1], labels_format=self.labels_format)
        self.patch_coord_generator  = PatchCoordinateGenerator(must_match='w_ar',
                                                               min_scale=min_scale,
                                                               max_scale=max_scale,
                                                               scale_uniformly=False,
                                                               min_aspect_ratio = min_aspect_ratio,
                                                               max_aspect_ratio = max_aspect_ratio)
        self.random_patch           = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                                  box_filter=self.box_filter_patch,
                                                  image_validator=self.image_validator,
                                                  n_trials_max=self.n_trials_max,
                                                  clip_boxes=self.clip_boxes,
                                                  prob=1.0,
                                                  can_fail=False,
                                                  labels_format=self.labels_format)

        # Define the processing chain.
        self.transformations = [self.convert_to_3_channels,
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
                                self.random_horizontal_flip,
                                self.random_vertical_flip,
                                self.random_rotate,
                                self.random_patch,
                                self.resize]

    def __call__(self, image, labels=None):

        self.random_patch.labels_format = self.labels_format
        self.random_horizontal_flip.labels_format = self.labels_format
        self.random_vertical_flip.labels_format = self.labels_format
        self.random_rotate.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        if not (labels is None):
            for transform in self.transformations:
                image, labels = transform(image, labels)
            return image, labels
        else:
            for transform in self.sequence1:
                image = transform(image)
            return image
