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
import cv2
import inspect

from data_generator.object_detection_2d_photometric_ops import ConvertColor, ConvertDataType, ConvertTo3Channels, RandomBrightness, RandomContrast, RandomHue, RandomSaturation, RandomChannelSwap
from data_generator.object_detection_2d_patch_sampling_ops import PatchCoordinateGenerator, RandomPatch, RandomPatchInf
from data_generator.object_detection_2d_geometric_ops import ResizeRandomInterp, RandomFlip
from data_generator.object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class SSDRandomCrop:
    '''
    Performs the same random crops as defined by the `batch_sampler` instructions
    of the original Caffe implementation of SSD. A description of this random cropping
    strategy can also be found in the data augmentation section of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self, labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        # This randomly samples one of the lower IoU bounds defined
        # by the `sample_space` every time it is called.
        self.bound_generator = BoundGenerator(sample_space=((None, None),
                                                            (0.1, None),
                                                            (0.3, None),
                                                            (0.5, None),
                                                            (0.7, None),
                                                            (0.9, None)),
                                              weights=None)

        # Produces coordinates for candidate patches such that the height
        # and width of the patches are between 0.3 and 1.0 of the height
        # and width of the respective image and the aspect ratio of the
        # patches is between 0.5 and 2.0.
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=0.3,
                                                              max_scale=1.0,
                                                              scale_uniformly=False,
                                                              min_aspect_ratio = 0.5,
                                                              max_aspect_ratio = 2.0)

        # Filters out boxes whose center point does not lie within the
        # chosen patches.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion='center_point',
                                    labels_format=self.labels_format)

        # Determines whether a given patch is considered a valid patch.
        # Defines a patch to be valid if at least one ground truth bounding box
        # (n_boxes_min == 1) has an IoU overlap with the patch that
        # meets the requirements defined by `bound_generator`.
        self.image_validator = ImageValidator(overlap_criterion='iou',
                                              n_boxes_min=1,
                                              labels_format=self.labels_format,
                                              border_pixels='half')

        # Performs crops according to the parameters set in the objects above.
        # Runs until either a valid patch is found or the original input image
        # is returned unaltered. Runs a maximum of 50 trials to find a valid
        # patch for each new sampled IoU threshold. Every 50 trials, the original
        # image is returned as is with probability (1 - prob) = 0.143.
        self.random_crop = RandomPatchInf(patch_coord_generator=self.patch_coord_generator,
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          bound_generator=self.bound_generator,
                                          n_trials_max=50,
                                          clip_boxes=True,
                                          prob=0.857,
                                          labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.random_crop.labels_format = self.labels_format
        return self.random_crop(image, labels, return_inverter)

class SSDExpand:
    '''
    Performs the random image expansion as defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD. A description of this expansion strategy
    can also be found in section 3.6 ("Data Augmentation for Small Object Accuracy") of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self, background=(123, 117, 104), labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        # Generate coordinates for patches that are between 1.0 and 4.0 times
        # the size of the input image in both spatial dimensions.
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=1.0,
                                                              max_scale=4.0,
                                                              scale_uniformly=True)

        # With probability 0.5, place the input image randomly on a canvas filled with
        # mean color values according to the parameters set above. With probability 0.5,
        # return the input image unaltered.
        self.expand = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                  box_filter=None,
                                  image_validator=None,
                                  n_trials_max=1,
                                  clip_boxes=False,
                                  prob=0.5,
                                  background=background,
                                  labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.expand.labels_format = self.labels_format
        return self.expand(image, labels, return_inverter)

class SSDPhotometricDistortions:
    '''
    Performs the photometric distortions defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD.
    '''

    def __init__(self):

        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.convert_to_3_channels = ConvertTo3Channels()
        self.random_brightness = RandomBrightness(lower=-32, upper=32, prob=0.5)
        self.random_contrast = RandomContrast(lower=0.5, upper=1.5, prob=0.5)
        self.random_saturation = RandomSaturation(lower=0.5, upper=1.5, prob=0.5)
        self.random_hue = RandomHue(max_delta=18, prob=0.5)
        self.random_channel_swap = RandomChannelSwap(prob=0.0)

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
                          self.random_channel_swap]

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
                          self.random_channel_swap]

    def __call__(self, image, labels):

        # Choose sequence 1 with probability 0.5.
        if np.random.choice(2):

            for transform in self.sequence1:
                image, labels = transform(image, labels)
            return image, labels
        # Choose sequence 2 with probability 0.5.
        else:

            for transform in self.sequence2:
                image, labels = transform(image, labels)
            return image, labels

class SSDDataAugmentation:
    '''
    Reproduces the data augmentation pipeline used in the training of the original
    Caffe implementation of SSD.
    '''

    def __init__(self,
                 img_height=300,
                 img_width=300,
                 background=(123, 117, 104),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the
                background pixels of the translated images.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.labels_format = labels_format

        self.photometric_distortions = SSDPhotometricDistortions()
        self.expand = SSDExpand(background=background, labels_format=self.labels_format)
        self.random_crop = SSDRandomCrop(labels_format=self.labels_format)
        self.random_flip = RandomFlip(dim='horizontal', prob=0.5, labels_format=self.labels_format)

        # This box filter makes sure that the resized images don't contain any degenerate boxes.
        # Resizing the images could lead the boxes to becomes smaller. For boxes that are already
        # pretty small, that might result in boxes with height and/or width zero, which we obviously
        # cannot allow.
        self.box_filter = BoxFilter(check_overlap=False,
                                    check_min_area=False,
                                    check_degenerate=True,
                                    labels_format=self.labels_format)

        self.resize = ResizeRandomInterp(height=img_height,
                                         width=img_width,
                                         interpolation_modes=[cv2.INTER_NEAREST,
                                                              cv2.INTER_LINEAR,
                                                              cv2.INTER_CUBIC,
                                                              cv2.INTER_AREA,
                                                              cv2.INTER_LANCZOS4],
                                         box_filter=self.box_filter,
                                         labels_format=self.labels_format)

        self.sequence = [self.photometric_distortions,
                         self.expand,
                         self.random_crop,
                         self.random_flip,
                         self.resize]

    def __call__(self, image, labels, return_inverter=False):
        self.expand.labels_format = self.labels_format
        self.random_crop.labels_format = self.labels_format
        self.random_flip.labels_format = self.labels_format
        self.resize.labels_format = self.labels_format

        inverters = []

        for transform in self.sequence:
            if return_inverter and ('return_inverter' in inspect.signature(transform).parameters):
                image, labels, inverter = transform(image, labels, return_inverter=True)
                inverters.append(inverter)
            else:
                image, labels = transform(image, labels)

        if return_inverter:
            return image, labels, inverters[::-1]
        else:
            return image, labels
