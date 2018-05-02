'''
Utilities for 2D object detection related to answering the following questions:
1. Given an image size and bounding boxes, which bounding boxes meet certain
   requirements with respect to the image size?
2. Given an image size and bounding boxes, is an image of that size valid with
   respect to the bounding boxes according to certain requirements?

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

from bounding_box_utils.bounding_box_utils import iou

class BoundGenerator:
    '''
    Generates pairs of floating point values that represent lower and upper bounds
    from a given sample space.
    '''
    def __init__(self,
                 sample_space=((0.1, None),
                               (0.3, None),
                               (0.5, None),
                               (0.7, None),
                               (0.9, None),
                               (None, None)),
                 weights=None):
        '''
        Arguments:
            sample_space (list or tuple): A list, tuple, or array-like object of shape
                `(n, 2)` that contains `n` samples to choose from, where each sample
                is a 2-tuple of scalars and/or `None` values.
            weights (list or tuple, optional): A list or tuple representing the distribution
                over the sample space. If `None`, a uniform distribution will be assumed.
        '''

        if (not (weights is None)) and len(weights) != len(sample_space):
            raise ValueError("`weights` must either be `None` for uniform distribution or have the same length as `sample_space`.")

        self.sample_space = []
        for bound_pair in sample_space:
            if len(bound_pair) != 2:
                raise ValueError("All elements of the sample space must be 2-tuples.")
            bound_pair = list(bound_pair)
            if bound_pair[0] is None: bound_pair[0] = 0.0
            if bound_pair[1] is None: bound_pair[1] = 1.0
            if bound_pair[0] > bound_pair[1]:
                raise ValueError("For all sample space elements, the lower bound cannot be greater than the upper bound.")
            self.sample_space.append(bound_pair)

        self.sample_space_size = len(self.sample_space)

        if weights is None:
            self.weights = [1.0/self.sample_space_size] * self.sample_space_size
        else:
            self.weights = weights

    def __call__(self):
        '''
        Returns:
            An item of the sample space, i.e. a 2-tuple of scalars.
        '''
        i = np.random.choice(self.sample_space_size, p=self.weights)
        return self.sample_space[i]

class BoxFilter:
    '''
    Returns all bounding boxes that are valid with respect to a the defined criteria.
    '''

    def __init__(self,
                 check_overlap=True,
                 check_min_area=True,
                 check_degenerate=True,
                 overlap_criterion='center_point',
                 overlap_bounds=(0.3, 1.0),
                 min_area=16,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Arguments:
            check_overlap (bool, optional): Whether or not to enforce the overlap requirements defined by
                `overlap_criterion` and `overlap_bounds`. Sometimes you might want to use the box filter only
                to enforce a certain minimum area for all boxes (see next argument), in such cases you can
                turn the overlap requirements off.
            check_min_area (bool, optional): Whether or not to enforce the minimum area requirement defined
                by `min_area`. If `True`, any boxes that have an area (in pixels) that is smaller than `min_area`
                will be removed from the labels of an image. Bounding boxes below a certain area aren't useful
                training examples. An object that takes up only, say, 5 pixels in an image is probably not
                recognizable anymore, neither for a human, nor for an object detection model. It makes sense
                to remove such boxes.
            check_degenerate (bool, optional): Whether or not to check for and remove degenerate bounding boxes.
                Degenerate bounding boxes are boxes that have `xmax <= xmin` and/or `ymax <= ymin`. In particular,
                boxes with a width and/or height of zero are degenerate. It is obviously important to filter out
                such boxes, so you should only set this option to `False` if you are certain that degenerate
                boxes are not possible in your data and processing chain.
            overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within the given `overlap_bounds`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within the given `overlap_bounds`.
            overlap_bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            min_area (int, optional): Only relevant if `check_min_area` is `True`. Defines the minimum area in
                pixels that a bounding box must have in order to be valid. Boxes with an area smaller than this
                will be removed.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        '''
        if not isinstance(overlap_bounds, (list, tuple, BoundGenerator)):
            raise ValueError("`overlap_bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(overlap_bounds, (list, tuple)) and (overlap_bounds[0] > overlap_bounds[1]):
            raise ValueError("The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError("`overlap_criterion` must be one of 'iou', 'area', or 'center_point'.")
        self.overlap_criterion = overlap_criterion
        self.overlap_bounds = overlap_bounds
        self.min_area = min_area
        self.check_overlap = check_overlap
        self.check_min_area = check_min_area
        self.check_degenerate = check_degenerate
        self.labels_format = labels_format
        self.border_pixels = border_pixels

    def __call__(self,
                 labels,
                 image_height=None,
                 image_width=None):
        '''
        Arguments:
            labels (array): The labels to be filtered. This is an array with shape `(m,n)`, where
                `m` is the number of bounding boxes and `n` is the number of elements that defines
                each bounding box (box coordinates, class ID, etc.). The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): Only relevant if `check_overlap == True`. The height of the image
                (in pixels) to compare the box coordinates to.
            image_width (int): `check_overlap == True`. The width of the image (in pixels) to compare
                the box coordinates to.

        Returns:
            An array containing the labels of all boxes that are valid.
        '''

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Record the boxes that pass all checks here.
        requirements_met = np.ones(shape=labels.shape[0], dtype=np.bool)

        if self.check_degenerate:

            non_degenerate = (labels[:,xmax] > labels[:,xmin]) * (labels[:,ymax] > labels[:,ymin])
            requirements_met *= non_degenerate

        if self.check_min_area:

            min_area_met = (labels[:,xmax] - labels[:,xmin]) * (labels[:,ymax] - labels[:,ymin]) >= self.min_area
            requirements_met *= min_area_met

        if self.check_overlap:

            # Get the lower and upper bounds.
            if isinstance(self.overlap_bounds, BoundGenerator):
                lower, upper = self.overlap_bounds()
            else:
                lower, upper = self.overlap_bounds

            # Compute which boxes are valid.

            if self.overlap_criterion == 'iou':
                # Compute the patch coordinates.
                image_coords = np.array([0, 0, image_width, image_height])
                # Compute the IoU between the patch and all of the ground truth boxes.
                image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]], coords='corners', mode='element-wise', border_pixels=self.border_pixels)
                requirements_met *= (image_boxes_iou > lower) * (image_boxes_iou <= upper)

            elif self.overlap_criterion == 'area':
                if self.border_pixels == 'half':
                    d = 0
                elif self.border_pixels == 'include':
                    d = 1 # If border pixels are supposed to belong to the bounding boxes, we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
                elif self.border_pixels == 'exclude':
                    d = -1 # If border pixels are not supposed to belong to the bounding boxes, we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
                # Compute the areas of the boxes.
                box_areas = (labels[:,xmax] - labels[:,xmin] + d) * (labels[:,ymax] - labels[:,ymin] + d)
                # Compute the intersection area between the patch and all of the ground truth boxes.
                clipped_boxes = np.copy(labels)
                clipped_boxes[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=image_height-1)
                clipped_boxes[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=image_width-1)
                intersection_areas = (clipped_boxes[:,xmax] - clipped_boxes[:,xmin] + d) * (clipped_boxes[:,ymax] - clipped_boxes[:,ymin] + d) # +1 because the border pixels belong to the box areas.
                # Check which boxes meet the overlap requirements.
                if lower == 0.0:
                    mask_lower = intersection_areas > lower * box_areas # If `self.lower == 0`, we want to make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
                else:
                    mask_lower = intersection_areas >= lower * box_areas # Especially for the case `self.lower == 1` we want the ">=" sign, otherwise no boxes would count at all.
                mask_upper = intersection_areas <= upper * box_areas
                requirements_met *= mask_lower * mask_upper

            elif self.overlap_criterion == 'center_point':
                # Compute the center points of the boxes.
                cy = (labels[:,ymin] + labels[:,ymax]) / 2
                cx = (labels[:,xmin] + labels[:,xmax]) / 2
                # Check which of the boxes have center points within the cropped patch remove those that don't.
                requirements_met *= (cy >= 0.0) * (cy <= image_height-1) * (cx >= 0.0) * (cx <= image_width-1)

        return labels[requirements_met]

class ImageValidator:
    '''
    Returns `True` if a given minimum number of bounding boxes meets given overlap
    requirements with an image of a given height and width.
    '''

    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 n_boxes_min=1,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4},
                 border_pixels='half'):
        '''
        Arguments:
            overlap_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given image. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the image.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the image and its own area is within `lower` and `upper`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the image is within `lower` and `upper`.
            bounds (list or BoundGenerator, optional): Only relevant if `overlap_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `overlap_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
            n_boxes_min (int or str, optional): Either a non-negative integer or the string 'all'.
                Determines the minimum number of boxes that must meet the `overlap_criterion` with respect to
                an image of the given height and width in order for the image to be a valid image.
                If set to 'all', an image is considered valid if all given boxes meet the `overlap_criterion`.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
        '''
        if not ((isinstance(n_boxes_min, int) and n_boxes_min > 0) or n_boxes_min == 'all'):
            raise ValueError("`n_boxes_min` must be a positive integer or 'all'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format
        self.border_pixels = border_pixels
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds,
                                    labels_format=self.labels_format,
                                    border_pixels=self.border_pixels)

    def __call__(self,
                 labels,
                 image_height,
                 image_width):
        '''
        Arguments:
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the image's coordinate system.
            image_height (int): The height of the image to compare the box coordinates to.
            image_width (int): The width of the image to compare the box coordinates to.

        Returns:
            A boolean indicating whether an imgae of the given height and width is
            valid with respect to the given bounding boxes.
        '''

        self.box_filter.overlap_bounds = self.bounds
        self.box_filter.labels_format = self.labels_format

        # Get all boxes that meet the overlap requirements.
        valid_labels = self.box_filter(labels=labels,
                                       image_height=image_height,
                                       image_width=image_width)

        # Check whether enough boxes meet the requirements.
        if isinstance(self.n_boxes_min, int):
            # The image is valid if at least `self.n_boxes_min` ground truth boxes meet the requirements.
            if len(valid_labels) >= self.n_boxes_min:
                return True
            else:
                return False
        elif self.n_boxes_min == 'all':
            # The image is valid if all ground truth boxes meet the requirements.
            if len(valid_labels) == len(labels):
                return True
            else:
                return False
