'''
Various patch sampling operations for data augmentation in 2D object detection.

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

from data_generator.object_detection_2d_image_boxes_validation_utils import BoundGenerator, BoxFilter, ImageValidator

class PatchCoordinateGenerator:
    '''
    Generates random patch coordinates that meet specified requirements.
    '''

    def __init__(self,
                 img_height=None,
                 img_width=None,
                 must_match='h_w',
                 min_scale=0.3,
                 max_scale=1.0,
                 scale_uniformly=False,
                 min_aspect_ratio = 0.5,
                 max_aspect_ratio = 2.0,
                 patch_ymin=None,
                 patch_xmin=None,
                 patch_height=None,
                 patch_width=None,
                 patch_aspect_ratio=None):
        '''
        Arguments:
            img_height (int): The height of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
            img_width (int): The width of the image for which the patch coordinates
                shall be generated. Doesn't have to be known upon construction.
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
            scale_uniformly (bool, optional): If `True` and if `must_match == 'h_w'`,
                the patch height and width will be scaled uniformly, otherwise they will
                be scaled independently.
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
        if scale_uniformly and not ((patch_height is None) and (patch_width is None)):
            raise ValueError("If `scale_uniformly == True`, `patch_height` and `patch_width` must both be `None`.")
        self.img_height = img_height
        self.img_width = img_width
        self.must_match = must_match
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_uniformly = scale_uniformly
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
            if not self.scale_uniformly:
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
            else:
                scaling_factor = np.random.uniform(self.min_scale, self.max_scale)
                patch_height = int(scaling_factor * self.img_height)
                patch_width = int(scaling_factor * self.img_width)

        elif self.must_match == 'h_ar': # Width is the dependent variable.
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

        elif self.must_match == 'w_ar': # Height is the dependent variable.
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
            y_range = self.img_height - patch_height
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
            x_range = self.img_width - patch_width
            # Select a random top left corner for the sample position from the possible positions.
            if x_range >= 0: patch_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension.
            else: patch_xmin = np.random.randint(x_range, 1) # The possible positions for the image on the background canvas in the horizontal dimension.
        else:
            patch_xmin = self.patch_xmin

        return (patch_ymin, patch_xmin, patch_height, patch_width)

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
                 clip_boxes=True,
                 box_filter=None,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
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
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        #if (patch_height <= 0) or (patch_width <= 0):
        #    raise ValueError("Patch height and width must both be positive.")
        #if (patch_ymin + patch_height < 0) or (patch_xmin + patch_width < 0):
        #    raise ValueError("A patch with the given coordinates cannot overlap with an input image.")
        if not (isinstance(box_filter, BoxFilter) or box_filter is None):
            raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_ymin = patch_ymin
        self.patch_xmin = patch_xmin
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.background = background
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if (self.patch_ymin > img_height) or (self.patch_xmin > img_width):
            raise ValueError("The given patch doesn't overlap with the input image.")

        labels = np.copy(labels)

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Top left corner of the patch relative to the image coordinate system:
        patch_ymin = self.patch_ymin
        patch_xmin = self.patch_xmin

        # Create a canvas of the size of the patch we want to end up with.
        if image.ndim == 3:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width, 3), dtype=np.uint8)
            canvas[:, :] = self.background
        elif image.ndim == 2:
            canvas = np.zeros(shape=(self.patch_height, self.patch_width), dtype=np.uint8)
            canvas[:, :] = self.background[0]

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

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] += patch_ymin
                labels[:, [xmin+1, xmax+1]] += patch_xmin
                return labels

        if not (labels is None):

            # Translate the box coordinates to the patch's coordinate system.
            labels[:, [ymin, ymax]] -= patch_ymin
            labels[:, [xmin, xmax]] -= patch_xmin

            # Compute all valid boxes for this patch.
            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.patch_height,
                                         image_width=self.patch_width)

            if self.clip_boxes:
                labels[:,[ymin,ymax]] = np.clip(labels[:,[ymin,ymax]], a_min=0, a_max=self.patch_height-1)
                labels[:,[xmin,xmax]] = np.clip(labels[:,[xmin,xmax]], a_min=0, a_max=self.patch_width-1)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

        else:
            if return_inverter:
                return image, inverter
            else:
                return image

class Crop:
    '''
    Crops off the specified numbers of pixels from the borders of images.

    This is just a convenience interface for `CropPad`.
    '''

    def __init__(self,
                 crop_top,
                 crop_bottom,
                 crop_left,
                 crop_right,
                 clip_boxes=True,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.clip_boxes = clip_boxes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.crop = CropPad(patch_ymin=self.crop_top,
                            patch_xmin=self.crop_left,
                            patch_height=None,
                            patch_width=None,
                            clip_boxes=self.clip_boxes,
                            box_filter=self.box_filter,
                            labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.crop.patch_height = img_height - self.crop_top - self.crop_bottom
        self.crop.patch_width = img_width - self.crop_left - self.crop_right
        self.crop.labels_format = self.labels_format

        return self.crop(image, labels, return_inverter)

class Pad:
    '''
    Pads images by the specified numbers of pixels on each side.

    This is just a convenience interface for `CropPad`.
    '''

    def __init__(self,
                 pad_top,
                 pad_bottom,
                 pad_left,
                 pad_right,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.background = background
        self.labels_format = labels_format
        self.pad = CropPad(patch_ymin=-self.pad_top,
                           patch_xmin=-self.pad_left,
                           patch_height=None,
                           patch_width=None,
                           clip_boxes=False,
                           box_filter=None,
                           background=self.background,
                           labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        self.pad.patch_height = img_height + self.pad_top + self.pad_bottom
        self.pad.patch_width = img_width + self.pad_left + self.pad_right
        self.pad.labels_format = self.labels_format

        return self.pad(image, labels, return_inverter)

class RandomPatch:
    '''
    Randomly samples a patch from an image. The randomness refers to whatever
    randomness may be introduced by the patch coordinate generator, the box filter,
    and the patch validator.

    Input images may be cropped and/or padded along either or both of the two
    spatial dimensions as necessary in order to obtain the required patch.

    As opposed to `RandomPatchInf`, it is possible for this transform to fail to produce
    an output image at all, in which case it will return `None`. This is useful, because
    if this transform is used to generate patches of a fixed size or aspect ratio, then
    the caller needs to be able to rely on the output image satisfying the set size or
    aspect ratio. It might therefore not be an option to return the unaltered input image
    as other random transforms do when they fail to produce a valid transformed image.
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 prob=1.0,
                 background=(0,0,0),
                 can_fail=False,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): A `PatchCoordinateGenerator` object
                to generate the positions and sizes of the patches to be sampled from the input images.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to sample a valid patch. If no valid patch could
                be sampled in `n_trials_max` trials, returns one `None` in place of each regular output.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            can_fail (bool, optional): If `True`, will return `None` if no valid patch could be found after
                `n_trials_max` trials. If `False`, will return the unaltered input image in such a case.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.can_fail = can_fail
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):

            img_height, img_width = image.shape[:2]
            self.patch_coord_generator.img_height = img_height
            self.patch_coord_generator.img_width = img_width

            xmin = self.labels_format['xmin']
            ymin = self.labels_format['ymin']
            xmax = self.labels_format['xmax']
            ymax = self.labels_format['ymax']

            # Override the preset labels format.
            if not self.image_validator is None:
                self.image_validator.labels_format = self.labels_format
            self.sample_patch.labels_format = self.labels_format

            for _ in range(max(1, self.n_trials_max)):

                # Generate patch coordinates.
                patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                self.sample_patch.patch_ymin = patch_ymin
                self.sample_patch.patch_xmin = patch_xmin
                self.sample_patch.patch_height = patch_height
                self.sample_patch.patch_width = patch_width

                if (labels is None) or (self.image_validator is None):
                    # We either don't have any boxes or if we do, we will accept any outcome as valid.
                    return self.sample_patch(image, labels, return_inverter)
                else:
                    # Translate the box coordinates to the patch's coordinate system.
                    new_labels = np.copy(labels)
                    new_labels[:, [ymin, ymax]] -= patch_ymin
                    new_labels[:, [xmin, xmax]] -= patch_xmin
                    # Check if the patch is valid.
                    if self.image_validator(labels=new_labels,
                                            image_height=patch_height,
                                            image_width=patch_width):
                        return self.sample_patch(image, labels, return_inverter)

            # If we weren't able to sample a valid patch...
            if self.can_fail:
                # ...return `None`.
                if labels is None:
                    if return_inverter:
                        return None, None
                    else:
                        return None
                else:
                    if return_inverter:
                        return None, None, None
                    else:
                        return None, None
            else:
                # ...return the unaltered input image.
                if labels is None:
                    if return_inverter:
                        return image, None
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, None
                    else:
                        return image, labels

        else:
            if return_inverter:
                def inverter(labels):
                    return labels

            if labels is None:
                if return_inverter:
                    return image, inverter
                else:
                    return image
            else:
                if return_inverter:
                    return image, labels, inverter
                else:
                    return image, labels

class RandomPatchInf:
    '''
    Randomly samples a patch from an image. The randomness refers to whatever
    randomness may be introduced by the patch coordinate generator, the box filter,
    and the patch validator.

    Input images may be cropped and/or padded along either or both of the two
    spatial dimensions as necessary in order to obtain the required patch.

    This operation is very similar to `RandomPatch`, except that:
    1. This operation runs indefinitely until either a valid patch is found or
       the input image is returned unaltered, i.e. it cannot fail.
    2. If a bound generator is given, a new pair of bounds will be generated
       every `n_trials_max` iterations.
    '''

    def __init__(self,
                 patch_coord_generator,
                 box_filter=None,
                 image_validator=None,
                 bound_generator=None,
                 n_trials_max=50,
                 clip_boxes=True,
                 prob=0.857,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_coord_generator (PatchCoordinateGenerator): A `PatchCoordinateGenerator` object
                to generate the positions and sizes of the patches to be sampled from the input images.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            bound_generator (BoundGenerator, optional): A `BoundGenerator` object to generate upper and
                lower bound values for the patch validator. Every `n_trials_max` trials, a new pair of
                upper and lower bounds will be generated until a valid patch is found or the original image
                is returned. This bound generator overrides the bound generator of the patch validator.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                The sampler will run indefinitely until either a valid patch is found or the original image
                is returned, but this determines the maxmial number of trials to sample a valid patch for each
                selected pair of lower and upper bounds before a new pair is picked.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        if not isinstance(patch_coord_generator, PatchCoordinateGenerator):
            raise ValueError("`patch_coord_generator` must be an instance of `PatchCoordinateGenerator`.")
        if not (isinstance(image_validator, ImageValidator) or image_validator is None):
            raise ValueError("`image_validator` must be either `None` or an `ImageValidator` object.")
        if not (isinstance(bound_generator, BoundGenerator) or bound_generator is None):
            raise ValueError("`bound_generator` must be either `None` or a `BoundGenerator` object.")
        self.patch_coord_generator = patch_coord_generator
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.bound_generator = bound_generator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.prob = prob
        self.background = background
        self.labels_format = labels_format
        self.sample_patch = CropPad(patch_ymin=None,
                                    patch_xmin=None,
                                    patch_height=None,
                                    patch_width=None,
                                    clip_boxes=self.clip_boxes,
                                    box_filter=self.box_filter,
                                    background=self.background,
                                    labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]
        self.patch_coord_generator.img_height = img_height
        self.patch_coord_generator.img_width = img_width

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        # Override the preset labels format.
        if not self.image_validator is None:
            self.image_validator.labels_format = self.labels_format
        self.sample_patch.labels_format = self.labels_format

        while True: # Keep going until we either find a valid patch or return the original image.

            p = np.random.uniform(0,1)
            if p >= (1.0-self.prob):

                # In case we have a bound generator, pick a lower and upper bound for the patch validator.
                if not ((self.image_validator is None) or (self.bound_generator is None)):
                    self.image_validator.bounds = self.bound_generator()

                # Use at most `self.n_trials_max` attempts to find a crop
                # that meets our requirements.
                for _ in range(max(1, self.n_trials_max)):

                    # Generate patch coordinates.
                    patch_ymin, patch_xmin, patch_height, patch_width = self.patch_coord_generator()

                    self.sample_patch.patch_ymin = patch_ymin
                    self.sample_patch.patch_xmin = patch_xmin
                    self.sample_patch.patch_height = patch_height
                    self.sample_patch.patch_width = patch_width

                    # Check if the resulting patch meets the aspect ratio requirements.
                    aspect_ratio = patch_width / patch_height
                    if not (self.patch_coord_generator.min_aspect_ratio <= aspect_ratio <= self.patch_coord_generator.max_aspect_ratio):
                        continue

                    if (labels is None) or (self.image_validator is None):
                        # We either don't have any boxes or if we do, we will accept any outcome as valid.
                        return self.sample_patch(image, labels, return_inverter)
                    else:
                        # Translate the box coordinates to the patch's coordinate system.
                        new_labels = np.copy(labels)
                        new_labels[:, [ymin, ymax]] -= patch_ymin
                        new_labels[:, [xmin, xmax]] -= patch_xmin
                        # Check if the patch contains the minimum number of boxes we require.
                        if self.image_validator(labels=new_labels,
                                                image_height=patch_height,
                                                image_width=patch_width):
                            return self.sample_patch(image, labels, return_inverter)
            else:
                if return_inverter:
                    def inverter(labels):
                        return labels

                if labels is None:
                    if return_inverter:
                        return image, inverter
                    else:
                        return image
                else:
                    if return_inverter:
                        return image, labels, inverter
                    else:
                        return image, labels

class RandomMaxCropFixedAR:
    '''
    Crops the largest possible patch of a given fixed aspect ratio
    from an image.

    Since the aspect ratio of the sampled patches is constant, they
    can subsequently be resized to the same size without distortion.
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 box_filter=None,
                 image_validator=None,
                 n_trials_max=3,
                 clip_boxes=True,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_aspect_ratio (float): The fixed aspect ratio that all sampled patches will have.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            image_validator (ImageValidator, optional): Only relevant if ground truth bounding boxes are given.
                An `ImageValidator` object to determine whether a sampled patch is valid. If `None`,
                any outcome is valid.
            n_trials_max (int, optional): Only relevant if ground truth bounding boxes are given.
                Determines the maxmial number of trials to sample a valid patch. If no valid patch could
                be sampled in `n_trials_max` trials, returns `None`.
            clip_boxes (bool, optional): Only relevant if ground truth bounding boxes are given.
                If `True`, any ground truth bounding boxes will be clipped to lie entirely within the
                sampled patch.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.patch_aspect_ratio = patch_aspect_ratio
        self.box_filter = box_filter
        self.image_validator = image_validator
        self.n_trials_max = n_trials_max
        self.clip_boxes = clip_boxes
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=self.box_filter,
                                        image_validator=self.image_validator,
                                        n_trials_max=self.n_trials_max,
                                        clip_boxes=self.clip_boxes,
                                        prob=1.0,
                                        can_fail=False,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        # The ratio of the input image aspect ratio and patch aspect ratio determines the maximal possible crop.
        image_aspect_ratio = img_width / img_height

        if image_aspect_ratio < self.patch_aspect_ratio:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))
        else:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)

class RandomPadFixedAR:
    '''
    Adds the minimal possible padding to an image that results in a patch
    of the given fixed aspect ratio that contains the entire image.

    Since the aspect ratio of the resulting images is constant, they
    can subsequently be resized to the same size without distortion.
    '''

    def __init__(self,
                 patch_aspect_ratio,
                 background=(0,0,0),
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            patch_aspect_ratio (float): The fixed aspect ratio that all sampled patches will have.
            background (list/tuple, optional): A 3-tuple specifying the RGB color value of the potential
                background pixels of the scaled images. In the case of single-channel images,
                the first element of `background` will be used as the background pixel value.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''

        self.patch_aspect_ratio = patch_aspect_ratio
        self.background = background
        self.labels_format = labels_format
        self.random_patch = RandomPatch(patch_coord_generator=PatchCoordinateGenerator(), # Just a dummy object
                                        box_filter=None,
                                        image_validator=None,
                                        n_trials_max=1,
                                        clip_boxes=False,
                                        background=self.background,
                                        prob=1.0,
                                        labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        if img_width < img_height:
            patch_height = img_height
            patch_width = int(round(patch_height * self.patch_aspect_ratio))
        else:
            patch_width = img_width
            patch_height = int(round(patch_width / self.patch_aspect_ratio))

        # Now that we know the desired height and width for the patch,
        # instantiate an appropriate patch coordinate generator.
        patch_coord_generator = PatchCoordinateGenerator(img_height=img_height,
                                                         img_width=img_width,
                                                         must_match='h_w',
                                                         patch_height=patch_height,
                                                         patch_width=patch_width)

        # The rest of the work is done by `RandomPatch`.
        self.random_patch.patch_coord_generator = patch_coord_generator
        self.random_patch.labels_format = self.labels_format
        return self.random_patch(image, labels, return_inverter)
