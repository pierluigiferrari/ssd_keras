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

class PatchGenerator:

    def __init__(self,
                 img_height,
                 img_width,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        '''
        Arguments:
            img_height (int): The height of the image for which this patch shall be
                generated.
            img_width (int): The width of the image for which this patch shall be
                generated.
            must_match (str, optional): Can be either of 'h_w', 'h_ar', and 'w_ar'.
                Specifies which two of the three quantities height, width, and aspect
                ratio determine the shape of the generated patch. The respective third
                quantity will be computed from the other two. For example,
                if `must_match == 'h_w'`, then the patch's height and width will be
                set to lie within [min_scale, max_scale] of the image size or to
                `patch_height` and/or `patch_width`, if given. The patch's aspect ratio
                is the dependent variable in this case, it will be computed from the
                height and width. Any given values for `patch_aspect_ratio`,
                `min_aspect_ratio`, or `max_aspect_ratio` will be ignored.
            min_scale (float, optional): The minimum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `min_scale == 0.5`,
                then the width of the generated patch will be at least 100. If `min_scale == 1.5`,
                the width of the generated patch will be at least 300.
            max_scale (float, optional): The maximum size of a dimension of the patch
                as a fraction of the respective dimension of the image. Can be greater
                than 1. For example, if the image width is 200 and `max_scale == 1.0`,
                then the width of the generated patch will be at most 200. If `max_scale == 1.5`,
                the width of the generated patch will be at most 300. Must be greater than
                `min_scale`.
            min_aspect_ratio (float, optional): Determines the minimum aspect ratio
                for the generated patches.
            max_aspect_ratio (float, optional): Determines the maximum aspect ratio
                for the generated patches.
            patch_ymin (int, optional): `None` or the vertical coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the vertical axis is fixed. If this is `None`, then the
                vertical position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the vertical dimension is
                always maximal.
            patch_xmin (int, optional): `None` or the horizontal coordinate of the top left
                corner of the generated patches. If this is not `None`, the position of the
                patches along the horizontal axis is fixed. If this is `None`, then the
                horizontal position of generated patches will be chosen randomly such that
                the overlap of a patch and the image along the horizontal dimension is
                always maximal.
            patch_height (int, optional): `None` or the fixed height of the generated patches.
            patch_width (int, optional): `None` or the fixed width of the generated patches.
            patch_aspect_ratio (float, optional): `None` or the fixed aspect ratio of the
                generated patches.
        '''

        if not (must_match in {'h_w', 'h_ar', 'w_ar'}):
            raise ValueError("`must_match` must be either of 'h_w', 'h_ar' and 'w_ar'.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError("It must be `min_aspect_ratio < max_aspect_ratio`.")
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_aspect_ratio = patch_aspect_ratio

    def __call__(self):
        '''
        Returns:
            A 4-tuple `(ymin, xmin, height, width)` that represents the coordinates
            of the generated patch.
        '''

        # Get the patch height and width.

        if self.must_match == 'h_w': # Aspect is the dependent variable.
            # Get the height.
            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
            else:
                patch_height = self.patch_height
            # Get the width.
            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
            else:
                patch_width = self.patch_width

        if self.must_match == 'h_ar': # Width is the dependent variable.
            # Get the height.
            if self.patch_height is None:
                patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_height)
            else:
                patch_height = self.patch_height
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the width.
            patch_width = int(patch_height * patch_aspect_ratio)

        if self.must_match == 'w_ar': # Height is the dependent variable.
            # Get the width.
            if self.patch_width is None:
                patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * self.img_width)
            else:
                patch_width = self.patch_width
            # Get the aspect ratio.
            if self.patch_aspect_ratio is None:
                patch_aspect_ratio = np.random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            else:
                patch_aspect_ratio = self.patch_aspect_ratio
            # Get the height.
            patch_height = int(patch_width / patch_aspect_ratio)

        # Get the top left corner coordinates of the patch.

        if self.patch_ymin is None:
            # Compute how much room we have along the vertical axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the vertical dimension, in which case the patch will be placed such that it fully contains the
            # image in the vertical dimension.
            y_range = img_height - patch_height
            # Select a random top left corner for the sample position from the possible positions.
            if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension.
            else: patch_ymin = np.random.randint(y_range, 1) # The possible positions for the image on the background canvas in the vertical dimension.
        else:
            patch_ymin = self.patch_ymin

        if self.patch_xmin is None:
            # Compute how much room we have along the horizontal axis to place the patch.
            # A negative number here means that we want to sample a patch that is larger than the original image
            # in the horizontal dimension, in which case the patch will be placed such that it fully contains the
            # image in the horizontal dimension.
            x_range = img_width - patch_width
            # Select a random top left corner for the sample position from the possible positions.
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)

class ValidBoxesPatch:
    '''
    Does either of two things:
    1. Tells you whether a given patch is valid based on the given criteria ('bool' mode).
    2. Returns all bounding boxes that are valid according to the given criteria ('boxes' mode).
    '''

    def __init__(self,
                 mode='boxes',
                 box_criterion='center_point',
                 patch_criterion=1,
                 bounds=(0.3, 1.0)):
        '''
        Arguments:
            mode (str, optional): Can be 'patch' or 'boxes'. In 'patch' mode, returns a boolean
                indicating whether a given patch is valid with respect to given labels. In 'boxes' mode,
                returns all valid boxes with respect to the given patch.
            box_criterion (str, optional): Can be either of 'center_point', 'iou', or 'area'. Determines
                which boxes are considered valid with respect to a given patch. If set to 'center_point',
                a given bounding box is considered valid if its center point lies within the patch.
                If set to 'area', a given bounding box is considered valid if the quotient of its intersection
                area with the patch and its own area is within `lower` and `upper`. If set to 'iou', a given
                bounding box is considered valid if its IoU with the patch is within `lower` and `upper`.
            patch_criterion (int or str, optional): Either a non-negative integer or the string 'all'.
                Determines the minimum number of boxes that must meet `box_criterion` with respect to a given
                patch for the patch to be considered a valid patch. If set to 'all', a patch is considered
                valid if all given boxes meet `box_criterion`.
            bounds (list or BoundGenerator, optional): Only relevant if `box_criterion` is 'area' or 'iou'.
                Determines the lower and upper bounds for `box_criterion`. Can be either a 2-tuple of scalars
                representing a lower bound and an upper bound, or a `BoundGenerator` object, which provides
                the possibility to generate bounds randomly.
        '''

        if not isinstance(bounds, (list, tuple, BoundGenerator)):
            raise ValueError("`bounds` must be either a 2-tuple of scalars or a `BoundGenerator` object.")
        if isinstance(bounds, (list, tuple)) and (bounds[0] > bounds[1]):
            raise ValueError("The lower bound must not be greater than the upper bound.")
        if not (mode in {'patch', 'boxes'}):
            raise ValueError("`mode` must be one of 'patch' or 'boxes'.")
        if not (box_criterion in {'iou', 'area', 'center_point'}):
            raise ValueError("`box_criterion` must be one of 'iou', 'area', or 'center_point'.")
        if not ((isinstance(patch_criterion, int) and patch_criterion > 0) or patch_criterion == 'all'):
            raise ValueError("`patch_criterion` must be a positive integer or 'all'.")
        self.mode = mode
        self.box_criterion = box_criterion
        self.patch_criterion = patch_criterion
        self.bounds = bounds

    def __call__(self,
                 patch_height,
                 patch_width,
                 labels):
        '''
        Arguments:
            patch_height (int): The height of the patch to be tested.
            patch_width (int): The width of the patch to be tested.
            labels (array): The labels to be tested. The box coordinates are expected
                to be in the patch's coordinate system.

        Returns:
            If `mode == bool`: A boolean indicating whether the given patch is valid.
            If `mode == boxes`: An array containing the labels of all boxes that are
                considered valid.
        '''

        labels = np.copy(labels)

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        # Compute the lower and upper bounds.
        if isinstance(self.bounds, BoundGenerator):
            lower, upper = self.bounds()
        else:
            lower, upper = self.bounds

        # Compute which boxes are valid.

        if self.box_criterion == 'iou':
            # Compute the patch coordinates.
            patch_coords = np.array([0, 0, patch_width, patch_height])
            # Compute the IoU between the patch and all of the ground truth boxes.
            patch_iou = iou(patch_coords, labels[:, 1:], coords='corners')
            requirements_met = (patch_iou > lower) * (patch_iou <= upper)

        elif self.box_criterion == 'area':
            # Compute the areas of the boxes.
            box_areas = (labels[:,xmax] - labels[:,xmin]) * (labels[:,ymax] - labels[:,ymin])
            # Compute the intersection area between the patch and all of the ground truth boxes.
            clipped_boxes = np.copy(labels)
            clipped_boxes[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=patch_height-1)
            clipped_boxes[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=patch_width-1)
            intersection_areas = (clipped_boxes[:,xmax] - clipped_boxes[:,xmin]) * (clipped_boxes[:,ymax] - clipped_boxes[:,ymin])
            # Check which boxes meet the overlap requirements.
            if self.lower == 0.0:
                mask_lower = intersection_areas > lower * box_areas # If `self.lower == 0`, we want to make sure that boxes with area 0 don't count, hence the ">" sign instead of the ">=" sign.
            else:
                mask_lower = intersection_areas >= lower * box_areas # Especially for the case `self.lower == 1` we want the ">=" sign, otherwise no boxes would count at all.
            mask_upper = intersection_areas <= upper * box_areas
            requirements_met = mask_lower * mask_upper

        elif self.box_criterion == 'center_point':
            # Compute the center points of the boxes.
            cy = (labels[:,ymin] + labels[:,ymax]) / 2
            cx = (labels[:,xmin] + labels[:,xmax]) / 2
            # Check which of the boxes have center points within the cropped patch remove those that don't.
            requirements_met = (cy >= 0.0) * (cy <= patch_height-1) * (cx >= 0.0) * (cx <= patch_width-1)

        if self.mode == 'boxes': # Return all boxes that meet the criteria.
            return labels[requirements_met]

        # Compute whether the patch is valid.

        elif self.mode == 'patch': # Return a boolean that indicates whether or not the patch matches the criteria.
            # Check whether enough boxes meet the requirements.
            if isinstance(self.patch_criterion, int):
                n_requirements_met = np.count_nonzero(requirements_met)
                # The patch is valid if at least `self.number_criterion` ground truth boxes meet the requirements.
                if n_requirements_met >= self.patch_criterion:
                    return True
                else:
                    return False
            elif self.patch_criterion == 'all':
                # The patch is valid if all ground truth boxes meet the requirements.
                if np.all(requirements_met):
                    return True
                else:
                    return False

class CropPad:
    '''
    Crops and/or pads an image deterministically.

    Depending on the given output patch size and the position (top left corner) relative
    to the input image, the image will be cropped and/or padded along one or both spatial
    dimensions.

    For example, if the output patch lies entirely within the input image, this will result
    in a regular crop. If the input image lies entirely within the output patch, this will
    result in the image being padded in every direction. All other cases are mixed cases
    where the image might be cropped in some directions and padded in others.

    The output patch can be arbitrary in both size and position as long as it overlaps
    with the input image.
    '''

    def __init__(self,
                 patch_ymin,
                 patch_xmin,
                 patch_height,
                 patch_width,
                 clip_boxes=False,
                 box_filter=None):
        '''
        Arguments:
            patch_ymin (int, optional): The vertical coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_ymin (int, optional): The horizontal coordinate of the top left corner of the output
                patch relative to the image coordinate system. Can be negative (i.e. lie outside the image)
                as long as the resulting patch still overlaps with the image.
            patch_height (int): The height of the patch to be sampled from the image. Can be greater
                than the height of the input image.
            patch_width (int): The width of the patch to be sampled from the image. Can be greater
                than the width of the input image.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            box_filter (ValidBoxesPatch, optional): Only relevant if ground truth bounding boxes are given.
                A `ValidBoxesPatch` object in 'boxes' mode that filters out bounding boxes that don't meet
                the overlap criteria with the sampled patch.
        '''

        if (patch_height <= 0) or (patch_width <= 0):
            raise ValueError("Patch height and width must both be positive.")
        if (patch_ymin + patch_height < 0) or (patch_xmin + patch_width < 0):
            raise ValueError("A patch with the given coordinates cannot overlap with an input image.")
        if not (isinstance(box_filter, ValidBoxesPatch) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `ValidBoxesPatch` object.")
        if (not box_filter is None) and (box_filter.mode != 'boxes'):
            raise ValueError("`box_filter` must be in 'boxes' mode.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter

    def __call__(self, image, labels=None):

        img_height, img_width, img_channels = image.shape

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError("The given patch doesn't overlap with the input image.")
        if (self.box_filter is None) and (not labels is None):
            raise ValueError("If labels are given, `box_filter` must not be `None`.")

        labels = np.copy(labels)

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        # Top left corner of the patch relative to the image coordinate system:
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        # Create a canvas of the size of the patch we want to end up with.
        canvas = np.zeros((self.patch_height, self.patch_width, img_channels), dtype=np.uint8)

        # Perform the crop.
        if patch_ymin < 0 and patch_xmin < 0: # Pad the image at the top and on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[:image_crop_height, :image_crop_width]

        elif patch_ymin < 0 and patch_xmin >= 0: # Pad the image at the top and crop it on the left.
            image_crop_height = min(img_height, self.patch_height + patch_ymin)  # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[-patch_ymin:-patch_ymin + image_crop_height, :image_crop_width] = image[:image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        elif patch_ymin >= 0 and patch_xmin < 0: # Crop the image at the top and pad it on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(img_width, self.patch_width + patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, -patch_xmin:-patch_xmin + image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, :image_crop_width]

        elif patch_ymin >= 0 and patch_xmin >= 0: # Crop the image at the top and on the left.
            image_crop_height = min(self.patch_height, img_height - patch_ymin) # The number of pixels of the image that will end up on the canvas in the vertical direction.
            image_crop_width = min(self.patch_width, img_width - patch_xmin) # The number of pixels of the image that will end up on the canvas in the horizontal direction.
            canvas[:image_crop_height, :image_crop_width] = image[patch_ymin:patch_ymin + image_crop_height, patch_xmin:patch_xmin + image_crop_width]

        image = canvas

        if not (labels is None):

            # Translate the box coordinates to the patch's coordinate system.
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            # Compute all valid boxes for this patch.
            labels = box_filter(patch_height=self.patch_height,
                                patch_width=self.patch_width,
                                labels=labels)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=self.patch_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=self.patch_width-1)

            return image, labels

        else:
            return image

class RandomPatchFixedSize:
    '''
    Randomly samples a patch of a fixed height and width from an image. The randomness
    refers to the position of the cropped patch relative to the input image.

    Depending on the given output patch size the image will be cropped and/or padded
    along the respective spatial dimensions.

    The patch will always be sampled in a way such that the overlap of the patch and
    the input image is maximal.

    The patch height and width can be arbitrary. In particular, the patch height
    and/or width can be greater than the height and width of the input image.
    '''

    def __init__(self,
                 patch_height,
                 patch_width,
                 clip_boxes=False,
                 box_filter=None,
                 patch_validator=None,
                 n_trials_max=3):
        '''
        Arguments:
            patch_height (int): The height of the patch to be sampled from the image. Can be greater
                than the height of the input image, in which case the original image will be randomly
                padded to produce the desired patch height.
            patch_width (int): The width of the patch to be sampled from the image. Can be greater
                than the width of the input image, in which case the original image will be randomly
                padded to produce the desired patch width.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            box_filter (ValidBoxesPatch, optional): Only relevant if ground truth bounding boxes are given.
                A `ValidBoxesPatch` object in 'boxes' mode that filters out bounding boxes that don't meet
                the overlap criteria with the sampled patch.
            patch_validator (ValidBoxesPatch, optional): Only relevant if ground truth bounding boxes are given.
                A `ValidBoxesPatch` object in 'patch' mode that determines whether a sampled patch is
                considered valid. If `None`, any sampled patch is considered valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to sample a valid patch. If no valid patch could
                be sampled in `n_trials_max` trials, returns `None`.
        '''
        if not (isinstance(patch_validator, ValidBoxesPatch) or patch_validator is None):
            raise ValueError("`patch_validator` must be either `None` or a `ValidBoxesPatch` object.")
        if (not patch_validator is None) and (patch_validator.mode != 'patch'):
            raise ValueError("`patch_validator` must be in 'patch' mode.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.patch_validator = patch_validator
        self.n_trials_max = n_trials_max

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        # Compute how much room we have in both dimensions to sample a patch.
        # A negative number here means that we want to sample a patch that is larger than the original image
        # in the respective dimension, in which case the image will be padded along that dimension.
        y_range = img_height - self.patch_height
        x_range = img_width - self.patch_width

        for _ in range(max(1, self.n_trials_max)):

            # Select a random top left corner for the sample position from the possible positions.
            if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension.
            else: patch_ymin = np.random.randint(y_range, 1) # The possible positions for the image on the background canvas in the vertical dimension.
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.

            if labels is None:
                # Create a patch sampler object.
                sample_patch = CropPad(patch_ymin=patch_ymin,
                                       patch_xmin=patch_xmin,
                                       patch_height=self.patch_height,
                                       patch_width=self.patch_width,
                                       clip_boxes=self.clip_boxes,
                                       box_filter=self.box_filter)
                # Sample the patch.
                return sample_patch(image)
            else:
                if self.patch_validator is None: # We will accept any patch as valid.
                    # Create a patch sampler object.
                    sample_patch = CropPad(patch_ymin=patch_ymin,
                                           patch_xmin=patch_xmin,
                                           patch_height=self.patch_height,
                                           patch_width=self.patch_width,
                                           clip_boxes=self.clip_boxes,
                                           box_filter=self.box_filter)
                    # Sample the patch.
                    return sample_patch(image, labels)
                else:
                    # Translate the box coordinates to the patch's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] -= patch_ymin
                    new_labels[:, [xmin, xmax]] -= patch_xmin
                    # Check if the patch contains the minimum number of boxes we require.
                    if self.patch_validator(patch_height=self.patch_height,
                                            patch_width=self.patch_width,
                                            labels=new_labels):
                        # Create a patch sampler object.
                        sample_patch = CropPad(patch_ymin=patch_ymin,
                                               patch_xmin=patch_xmin,
                                               patch_height=self.patch_height,
                                               patch_width=self.patch_width,
                                               clip_boxes=self.clip_boxes,
                                               box_filter=self.box_filter)
                        # Sample the patch.
                        return sample_patch(image, labels)

        return None # If we weren't able to sample a valid patch, return `None`.

class RandomPatchVariableSize:
    '''
    Randomly
    '''

    def __init__(self,
                 min_scale=0.3,
                 max_scale=1.0,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 box_filter=None,
                 patch_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=False,
                 prob=0.857):
        if min_aspect_ratio >= max_aspect_ratio:
            raise ValueError("`max_aspect_ratio` must be greater than `min_aspect_ratio`.")
        if min_scale >= max_scale:
            raise ValueError("It must be `min_scale < max_scale`.")
        if not (isinstance(patch_validator, ValidBoxesPatch) or patch_validator is None):
            raise ValueError("`patch_validator` must be either `None` or a `ValidBoxesPatch` object.")
        if (not patch_validator is None) and (patch_validator.mode != 'patch'):
            raise ValueError("`patch_validator` must be in 'patch' mode.")
        if not (isinstance(bound_generator, BoundGenerator) or bound_generator is None):
            raise ValueError("`bound_generator` must be either `None` or a `BoundGenerator` object.")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.box_filter = box_filter
        self.patch_validator = patch_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob

    def __call__(self, image, labels=None):

        img_height, img_width = image.shape[:2]

        # Coordinates are expected to be in the 'corners' format.
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        counter = -1

        while True: # Keep going until we either find a valid patch or return the original image.

            counter += 1

            p = np.random.uniform(0,1)
            if p >= (1.0-self.prob):

                # In case we have a bound generator, pick a lower and upper bound for the patch validator.
                if not ((self.patch_validator is None) or (self.bound_generator is None)):
                    self.patch_validator.bounds = self.bound_generator()

                print("trying bounds:", self.patch_validator.bounds)

                # Use at most `self.n_trials_max` attempts to find a crop
                # that meets our requirements.
                for _ in range(self.n_trials_max):

                    # Determine the size of the sample patch.
                    patch_height = int(np.random.uniform(self.min_scale, self.max_scale) * img_height)
                    patch_width = int(np.random.uniform(self.min_scale, self.max_scale) * img_width)

                    # Check if the resulting patch meets the aspect ratio requirements.
                    aspect_ratio = patch_height / patch_width
                    if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                        continue

                    # Compute how much room we have in both dimensions to sample a patch.
                    # A negative number here means that we want to sample a patch that is larger than the original image
                    # in the respective dimension, in which case the image will be padded along that dimension.
                    y_range = img_height - patch_height
                    x_range = img_width - patch_width

                    # Select a random top left corner for the sample position from the possible positions.
                    if y_range >= 0: patch_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension.
                    else: patch_ymin = np.random.randint(y_range, 1) # The possible positions for the image on the background canvas in the vertical dimension.
                    if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
                    else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.

                    if labels is None:
                        # Create a patch sampler object.
                        sample_patch = CropPad(patch_ymin=patch_ymin,
                                               patch_xmin=patch_xmin,
                                               patch_height=patch_height,
                                               patch_width=patch_width,
                                               clip_boxes=self.clip_boxes,
                                               box_filter=self.box_filter)
                        # Sample the patch.
                        return sample_patch(image)
                    else:
                        if self.patch_validator is None: # We will accept any patch as valid.
                            # Create a patch sampler object.
                            sample_patch = CropPad(patch_ymin=patch_ymin,
                                                   patch_xmin=patch_xmin,
                                                   patch_height=patch_height,
                                                   patch_width=patch_width,
                                                   clip_boxes=self.clip_boxes,
                                                   box_filter=self.box_filter)
                            # Sample the patch.
                            return sample_patch(image, labels)
                        else:
                            # Translate the box coordinates to the patch's coordinate system.
                            new_labels = np.copy(labels)
                            new_labels[:, [ymin, ymax]] -= patch_ymin
                            new_labels[:, [xmin, xmax]] -= patch_xmin
                            # Check if the patch contains the minimum number of boxes we require.
                            if self.patch_validator(patch_height=patch_height,
                                                    patch_width=patch_width,
                                                    labels=new_labels):
                                print("success with rbounds:", self.patch_validator.bounds)
                                print("counter:", counter)
                                print("trial number:", _)
                                # Create a patch sampler object.
                                sample_patch = CropPad(patch_ymin=patch_ymin,
                                                       patch_xmin=patch_xmin,
                                                       patch_height=patch_height,
                                                       patch_width=patch_width,
                                                       clip_boxes=self.clip_boxes,
                                                       box_filter=self.box_filter)
                                # Sample the patch.
                                return sample_patch(image, labels)

            else:
                print("success with original image.")
                print("counter:", counter)
                if labels is None:
                    return image
                else:
                    return image, labels
