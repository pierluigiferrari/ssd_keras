'''
The data augmentation operations of the original SSD implementation.

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

from object_detection_2d_patch_sample_ops import BoundGenerator, PatchCoordinateGenerator, ValidBoxesPatch, RandomPatch, RandomPatchInf

class SSDRandomCrop:
    '''
    Performs the same random crops as defined by the `batch_sampler` instructions
    of the original Caffe implementation of SSD. A description of this random cropping
    strategy can also be found in the data augmentation section of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self):

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
        self.box_filter = ValidBoxesPatch(mode='boxes',
                                          box_criterion='center_point')

        # Determines whether a given patch is considered a valid patch.
        # Defines a patch to be valid if at least one ground truth bounding box
        # (patch_criterion == 1) has an IoU overlap with the patch that
        # meets the requirements defined by `bound_generator`.
        self.patch_validator = ValidBoxesPatch(mode='patch',
                                               box_criterion='iou',
                                               patch_criterion=1)

        # Performs crops according to the parameters set in the objects above.
        # Runs until either a valid patch is found or the original input image
        # is returned unaltered. Runs a maximum of 50 trials to find a valid
        # patch for each new sampled IoU threshold. Every 50 trials, the original
        # image is returned as is with probability (1 - prob) = 0.143.
        # Does not clip the ground truth bounding boxes to lie within the patch.
        self.random_crop = RandomPatchInf(patch_coord_generator=self.patch_coord_generator,
                                          box_filter=self.box_filter,
                                          patch_validator=self.patch_validator,
                                          bound_generator=self.bound_generator,
                                          n_trials_max=50,
                                          clip_boxes=False,
                                          prob=0.857)

    def __call__(self, image, labels=None):
        return self.random_crop(image, labels)

class SSDExpand:
    '''
    Performs the random image expansion as defined by the `train_transform_param` instructions
    of the original Caffe implementation of SSD. A description of this expansion strategy
    can also be found in section 3.6 ("Data Augmentation for Small Object Accuracy") of the paper:
    https://arxiv.org/abs/1512.02325
    '''

    def __init__(self):

        # Generate coordinates for patches that are between 1.0 and 4.0 times
        # the size of the input image in both spatial dimensions.
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=1.0,
                                                              max_scale=4.0,
                                                              scale_uniformly=True)

        # With probability 0.5, place the input image randomly on a patch according
        # to the parameters set above. With probability 0.5, return the input image
        # unaltered.
        self.expand = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                  box_filter=None,
                                  patch_validator=None,
                                  n_trials_max=1,
                                  clip_boxes=False,
                                  prob=0.5)

    def __call__(self, image, labels=None):
        return self.expand(image, labels)
