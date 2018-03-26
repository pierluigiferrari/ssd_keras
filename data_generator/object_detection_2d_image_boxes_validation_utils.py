'''
Utilities for 2D object detection related to answering the following questions:
1. Given an image size and bounding boxes, which bounding boxes meet certain
   requirements with respect to the image size?
2. Given an image size and bounding boxes, is an image of that size valid with
   respect to the bounding boxes according to certain requirements?

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
    Returns all bounding boxes that are valid with respect to a given image height
    and width according to the given criteria.
    '''

    def __init__(self,
                 overlap_criterion='center_point',
                 bounds=(0.3, 1.0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
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
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not isinstance(bounds, (list, tuple, BoundGenerator)):
            raise ValueError("`bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(bounds, (list, tuple)) and (bounds[0] > bounds[1]):
            raise ValueError("The lower bound must not be greater than the upper bound.")
        if not (overlap_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError("`overlap_criterion` must be one of 'iou', 'area', or 'center_point'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.labels_format = labels_format

    def __call__(self,
                 image_height,
                 image_width,
                 labels):
        '''
        Arguments:
            image_height (int): The height of the image to compare the box coordinates to.
            image_width (int): The width of the image to compare the box coordinates to.
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the image's coordinate system.

        Returns:
            An array containing the labels of all boxes that are valid.
        '''

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Get the lower and upper bounds.
        if isinstance(self.bounds, BoundGenerator):
            lower, upper = self.bounds()
        else:
            lower, upper = self.bounds

        # Compute which boxes are valid.

        if self.overlap_criterion == 'iou':
            # Compute the patch coordinates.
            image_coords = np.array([0, 0, image_width, image_height])
            # Compute the IoU between the patch and all of the ground truth boxes.
            image_boxes_iou = iou(image_coords, labels[:, [xmin, ymin, xmax, ymax]], coords='corners')
            requirements_met = (image_boxes_iou > lower) * (image_boxes_iou <= upper)

        elif self.overlap_criterion == 'area':
            # Compute the areas of the boxes.
            box_areas = (labels[:,xmax] - labels[:,xmin]) * (labels[:,ymax] - labels[:,ymin])
            # Compute the intersection area between the patch and all of the ground truth boxes.
            clipped_boxes = np.copy(labels)
            clipped_boxes[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=image_height-1)
            clipped_boxes[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=image_width-1)
            intersection_areas = (clipped_boxes[:,xmax] - clipped_boxes[:,xmin]) * (clipped_boxes[:,ymax] - clipped_boxes[:,ymin])
            # Check which boxes meet the overlap requirements.
            if lower == 0.0:
                mask_lower = intersection_areas > lower * box_areas # If `self.lower == 0`, we want to make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
            else:
                mask_lower = intersection_areas >= lower * box_areas # Especially for the case `self.lower == 1` we want the ">=" sign, otherwise no boxes would count at all.
            mask_upper = intersection_areas <= upper * box_areas
            requirements_met = mask_lower * mask_upper

        elif self.overlap_criterion == 'center_point':
            # Compute the center points of the boxes.
            cy = (labels[:,ymin] + labels[:,ymax]) / 2
            cx = (labels[:,xmin] + labels[:,xmax]) / 2
            # Check which of the boxes have center points within the cropped patch remove those that don't.
            requirements_met = (cy >= 0.0) * (cy <= image_height-1) * (cx >= 0.0) * (cx <= image_width-1)

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
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
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
        '''
        if not ((isinstance(n_boxes_min, int) and n_boxes_min > 0) or n_boxes_min == 'all'):
            raise ValueError("`n_boxes_min` must be a positive integer or 'all'.")
        self.overlap_criterion = overlap_criterion
        self.bounds = bounds
        self.n_boxes_min = n_boxes_min
        self.labels_format = labels_format

    def __call__(self,
                 image_height,
                 image_width,
                 labels):
        '''
        Arguments:
            image_height (int): The height of the image to compare the box coordinates to.
            image_width (int): The width of the image to compare the box coordinates to.
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the image's coordinate system.

        Returns:
            A boolean indicating whether an imgae of the given height and width is
            valid with respect to the given bounding boxes.
        '''

        box_filter = BoxFilter(overlap_criterion=self.overlap_criterion,
                               bounds=self.bounds,
                               labels_format=self.labels_format)

        # Get all boxes that meet the overlap requirements.
        valid_labels = box_filter(image_height=image_height,
                                  image_width=image_width,
                                  labels=labels)

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
