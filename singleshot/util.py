"""
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
"""

import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from copy import deepcopy
from PIL import Image
import csv
import os
from bs4 import BeautifulSoup


# Image processing functions used by the generator to perform the following image manipulations:
# - Translation
# - Horizontal flip
# - Scaling
# - Brightness change
# - Histogram contrast equalization

def _translate(image, horizontal=(0, 40), vertical=(0, 10)):
    """
    Randomly translate the input image horizontally and vertically.

    Arguments:
        image (array-like): The image to be translated.
        horizontal (int tuple, optinal): A 2-tuple `(min, max)` with the minimum
            and maximum horizontal translation. A random translation value will
            be picked from a uniform distribution over [min, max].
        vertical (int tuple, optional): Analog to `horizontal`.

    Returns:
        The translated image and the horzontal and vertical shift values.
    """
    rows, cols, ch = image.shape

    x = np.random.randint(horizontal[0], horizontal[1] + 1)
    y = np.random.randint(vertical[0], vertical[1] + 1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift


def _flip(image, orientation='horizontal'):
    """
    Flip the input image horizontally or vertically.
    """
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def _scale(image, min=0.9, max=1.1):
    """
    Scale the input image by a random factor picked from a uniform distribution
    over [min, max].

    Returns:
        The scaled image, the associated warp matrix, and the scaling value.
    """

    rows, cols, ch = image.shape

    # Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale


def _brightness(image, min=0.5, max=2.0):
    """
    Randomly change the brightness of the input image.

    Protected against overflow.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def histogram_eq(image):
    """
    Perform histogram equalization on the input image.

    See https://en.wikipedia.org/wiki/Histogram_equalization.
    """

    image1 = np.copy(image)

    image1[:, :, 0] = cv2.equalizeHist(image1[:, :, 0])
    image1[:, :, 1] = cv2.equalizeHist(image1[:, :, 1])
    image1[:, :, 2] = cv2.equalizeHist(image1[:, :, 2])

    return image1


class BatchGenerator:
    """
    A generator to generate batches of samples and corresponding labels indefinitely.

    The labels are read from a CSV file.

    Shuffles the dataset consistently after each complete pass.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    """

    def __init__(self,
                 include_classes=None,
                 box_output_format=None):
        """
        Arguments:
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            box_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, xmax, ymin, ymax in the generated data. The expected strings are
                'xmin', 'xmax', 'ymin', 'ymax', 'class_id'. If you want to train the model, this
                must be the order that the box encoding class requires as input. Defaults to
                `['class_id', 'xmin', 'xmax', 'ymin', 'ymax']`. Note that even though the parser methods are
                able to produce different output formats, the SSDBoxEncoder currently requires the format
                `['class_id', 'xmin', 'xmax', 'ymin', 'ymax']`. This list only specifies the five box parameters
                that are relevant as training targets, a list of filenames is generated separately.
        """
        # These are the variables we always need
        if box_output_format is None:
            box_output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax']
        self.class_map = {v: k+1 for k, v in enumerate(include_classes)} if include_classes else None
        self.class_map_inv = {k+1:v for k, v in enumerate(include_classes)} if include_classes else None
        self.include_classes = include_classes
        self.box_output_format = box_output_format

        # These are the variables that we only need if we want to use parse_csv()
        self.labels_path = None
        self.input_format = None

        # These are the variables that we only need if we want to use parse_xml()
        self.annotations_path = None
        self.image_set_path = None
        self.image_set = None
        self.classes = None

        # The two variables below store the output from the parsers. This is the input for the generate() method
        # `self.filenames` is a list containing all file names of the image samples. Note that it does not contain the actual image files themselves.
        self.filenames = []  # All unique image filenames will go here
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        self.labels = []  # Each entry here will contain a 2D Numpy array with all the ground truth boxes for a given image

    def parse_csv(self,
                  labels_path=None,
                  input_format=None,
                  ret=False):
        '''
        Arguments:
            labels_path (str, optional): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located. Defaults to `None`.
            input_format (list, optional): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
                are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'. Defaults to `None`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.

        Returns:
            None by default, optionally the image filenames and labels.
        '''

        # If we get arguments in this call, set them
        if not labels_path is None: self.labels_path = labels_path
        if not input_format is None: self.input_format = input_format

        # Before we begin, make sure that we have a labels_path and an input_format
        if self.labels_path is None or self.input_format is None:
            raise ValueError(
                "`labels_path` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            k = 0
            for row in csv_reader:  # For every line (i.e for every bounding box) in the CSV file...
                if k == 0:  # Skip the header row
                    k += 1
                    continue
                else:
                    if self.include_classes == 'all' or int(row[self.input_format.index(
                            'class_id')].strip()) in self.include_classes:  # If the class_id is among the classes that are to be included in the dataset...
                        obj = [row[self.input_format.index('image_name')].strip()]
                        # Store the box class and coordinates here
                        for item in self.box_output_format:
                            val = int(row[self.input_format.index(item)].strip())
                            if item == 'class_id' and self.class_map:
                                obj.append(self.class_map[val])
                            else:
                                obj.append(val)
                                # ...select the respective column in the input format and append it to `obj`
                        data.append(obj)

        data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = ''  # The current image for which we're collecting the ground truth boxes
        current_labels = []  # The list where we collect all ground truth boxes for a given image
        for idx, row in enumerate(data):
            if current_file == '':  # If this is the first image file
                current_file = row[0]
                current_labels.append(row[1:])
                if len(data) == 1:  # If there is only one box in the CVS file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
            else:
                if row[0] == current_file:
                    # If this box (i.e. this line of the CSV file) belongs to the current image file
                    current_labels.append(row[1:])
                    if idx == len(data) - 1:  # If this is the last line of the CSV file
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(current_file)
                else:  # If this box belongs to a new image file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
                    current_labels = []
                    current_file = row[0]
                    current_labels.append(row[1:])
        self.count = len(self.filenames)
        if ret:  # In case we want to return these
            return self.filenames, self.labels

    def parse_xml(self,
                  annotations_path=None,
                  image_set_path=None,
                  image_set=None,
                  classes=None,
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False):
        '''
        This is a parser for the Pascal VOC datasets. It might be used for other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            annotations_path (str, optional): The path to the directory that contains the annotation XML files for
                the images. The directory must contain one XML file per image and name of the XML file must be the
                image ID. The content of the XML files must be in the Pascal VOC format. Defaults to `None`.
            image_set_path (str, optional): The path to the directory that contains a text file with the image
                set to be loaded. Defaults to `None`.
            image_set (str, optional): The name of the image set text file to be loaded, ending in '.txt'.
                This text file simply contains one image ID per line and nothing else. Defaults to `None`.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
                Defaults to `False`.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
                Defaults to `False`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.

        Returns:
            None by default, optionally the image filenames and labels.
        '''

        if classes is None:
            classes = ['background',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat',
                       'chair', 'cow', 'diningtable', 'dog',
                       'horse', 'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor']
        if not annotations_path is None: self.annotations_path = annotations_path
        if not image_set_path is None: self.image_set_path = image_set_path
        if not image_set is None: self.image_set = image_set
        if not classes is None: self.classes = classes

        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        # Parse the image set that so that we know all the IDs of all the images to be included in the dataset
        with open(os.path.join(self.image_set_path, self.image_set)) as f:
            image_ids = [line.strip() for line in f]

        # Parse the labels for each image ID from its respective XML file
        for image_id in image_ids:
            # Open the XML file for this image
            with open(os.path.join(self.annotations_path, image_id + '.xml')) as f:
                soup = BeautifulSoup(f, 'xml')

            folder = soup.folder.text  # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
            filename = soup.filename.text
            self.filenames.append(filename)

            boxes = []  # We'll store all boxes for this image here
            objects = soup.find_all('object')  # Get a list of all objects in this image

            # Parse the data for each object
            for obj in objects:
                class_name = obj.find('name').text
                class_id = self.classes.index(class_name)
                # Check if this class is supposed to be included in the dataset
                if (self.include_classes is not None) and (class_id not in self.include_classes): continue
                pose = obj.pose.text
                truncated = int(obj.truncated.text)
                if exclude_truncated and (truncated == 1): continue
                difficult = int(obj.difficult.text)
                if exclude_difficult and (difficult == 1): continue
                xmin = int(obj.bndbox.xmin.text)
                ymin = int(obj.bndbox.ymin.text)
                xmax = int(obj.bndbox.xmax.text)
                ymax = int(obj.bndbox.ymax.text)
                item_dict = {'folder': folder,
                             'image_name': filename,
                             'image_id': image_id,
                             'class_name': class_name,
                             'class_id': class_id,
                             'pose': pose,
                             'truncated': truncated,
                             'difficult': difficult,
                             'xmin': xmin,
                             'ymin': ymin,
                             'xmax': xmax,
                             'ymax': ymax}
                box = []
                for item in self.box_output_format:
                    box.append(item_dict[item])
                boxes.append(box)

            self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels

    def generate(self,
                 batch_size=32,
                 train=True,
                 ssd_box_encoder=None,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 gray=False,
                 limit_boxes=True,
                 include_thresh=0.3,
                 diagnostics=False):
        '''
        Generate batches of samples and corresponding labels indefinitely from
        lists of filenames and labels.

        Returns two numpy arrays, one containing the next `batch_size` samples
        from `filenames`, the other containing the corresponding labels from
        `labels`.

        Shuffles `filenames` and `labels` consistently after each complete pass.

        Can perform image transformations for data conversion and data augmentation.
        `resize`, `gray`, and `equalize` are image conversion tools and should be
        used consistently during training and inference. The remaining transformations
        serve for data augmentation. Each data augmentation process can set its own
        independent application probability. The transformations are performed
        in the order of their arguments, i.e. equalization is performed first,
        grayscale conversion is performed last.

        `prob` works the same way in all arguments in which it appears. It must be a float in [0,1]
        and determines the probability that the respective transform is applied to any given image.

        All conversions and transforms default to `False`.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated. Defaults to 32.
            train (bool, optional): Whether or not the generator is used in training mode. If `True`, then the labels
                will be transformed into the format that the SSD cost function requires. Otherwise,
                the output format of the labels is identical to the input format. Defaults to `True`.
            ssd_box_encoder (SSDBoxEncoder, optional): Only required if `train = True`. An SSDBoxEncoder object
                to encode the ground truth labels to the required format for training an SSD model.
            equalize (bool, optional): If `True`, performs histogram equalization on the images.
                This can improve contrast and lead the improved model performance.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
                The respective box coordinates are adjusted accordingly.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for horizontal shift of the image,
                the second tuple for vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
                The respective box coordinates are adjusted accordingly.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both min and max must be >=0.
            random_crop (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`,
                where `height` and `width` are the height and width of the patch that is to be cropped out at a random
                position in the input image. Note that `height` and `width` can be arbitrary - they are allowed to be larger
                than the image height and width, in which case the original image will be randomly placed on a black background
                canvas of size `(height, width)`. `min_1_object` is either 0 or 1. If 1, there must be at least one detectable
                object remaining in the image for the crop to be valid, and if 0, crops with no detectable objects left in the
                image patch are allowed. `max_#_trials` is only relevant if `min_1_object == 1` and sets the maximum number
                of attempts to get a valid crop. If no valid crop was obtained within this maximum number of attempts,
                the respective image will be removed from the batch without replacement (i.e. for each removed image, the batch
                will be one sample smaller). Defaults to `False`.
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images.
                The targets are adjusted accordingly. Note: Cropping happens before resizing.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected format is `(width, height)`.
                The box coordinates are adjusted accordingly. Note: Resizing happens after cropping.
            gray (bool, optional): If `True`, converts the images to grayscale.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries
                post any transformation. This should always be set to `True`, even if you set `include_thresh`
                to 0. I don't even know why I made this an option. If this is set to `False`, you could
                end up with some boxes that lie entirely outside the image boundaries after a given transformation
                and such boxes would of course not make any sense and have a strongly adverse effect on the learning.
            include_thresh (float, optional): Only relevant if `limit_boxes` is `True`. Determines the minimum
                fraction of the area of a ground truth box that must be left after limiting in order for the box
                to still be included in the batch data. If set to 0, all boxes are kept except those which lie
                entirely outside of the image bounderies after limiting. If set to 1, only boxes that did not
                need to be limited at all are kept. Defaults to 0.3.
            diagnostics (bool, optional): If `True`, yields three additional output items:
                1) A list of the image file names in the batch.
                2) An array with the original, unaltered images.
                3) A list with the original, unaltered labels.
                This can be useful for diagnostic purposes. Defaults to `False`. Only works if `train = True`.

        Yields:
            The next batch as a tuple containing a Numpy array that contains the images and a python list
            that contains the corresponding labels for each image as 2D Numpy arrays. The output format
            of the labels is according to the `box_output_format` that was specified in the constructor.
        '''

        self.filenames, self.labels = shuffle(self.filenames, self.labels)  # Shuffle the data before we begin
        current = 0

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        while True:

            batch_X, batch_y = [], []

            # Shuffle the data after each complete pass
            if current >= len(self.filenames):
                self.filenames, self.labels = shuffle(self.filenames, self.labels)
                current = 0

            for filename in self.filenames[current:current + batch_size]:
                with Image.open('{}'.format(filename)) as img:
                    batch_X.append(np.array(img))
            batch_y = deepcopy(self.labels[current:current + batch_size])

            this_filenames = self.filenames[
                             current:current + batch_size]  # The filenames of the files in the current batch

            if diagnostics:
                original_images = np.copy(batch_X)  # The original, unaltered images
                original_labels = deepcopy(batch_y)  # The original, unaltered labels

            current += batch_size

            # At this point we're done producing the batch. Now perform some
            # optional image transformations:

            batch_items_to_remove = []  # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list

            for i in range(len(batch_X)):

                img_height, img_width, ch = batch_X[i].shape
                batch_y[i] = np.array(batch_y[
                                          i])  # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks

                if equalize:
                    batch_X[i] = histogram_eq(batch_X[i])

                if brightness:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - brightness[2]):
                        batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])

                # Could easily be extended to also allow vertical flipping, but I'm not convinced of the
                # usefulness of vertical flipping either empirically or theoretically, so I'm going for simplicity.
                # If you want to allow vertical flipping, just change this function to pass the respective argument
                # to `_flip()`.
                if flip:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - flip):
                        batch_X[i] = _flip(batch_X[i])
                        batch_y[i][:, [xmin, xmax]] = img_width - batch_y[i][:, [xmax,
                                                                                 xmin]]  # xmin and xmax are swapped when mirrored

                if translate:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - translate[2]):
                        # Translate the image and return the shift values so that we can adjust the labels
                        batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                        # Adjust the labels
                        batch_y[i][:, [xmin, xmax]] += xshift
                        batch_y[i][:, [ymin, ymax]] += yshift
                        # Limit the box coordinates to lie within the image boundaries
                        if limit_boxes:
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:, [xmin, xmax]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:, [xmin, xmax]] = x_coords
                            y_coords = batch_y[i][:, [ymin, ymax]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:, [ymin, ymax]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (
                            before_limiting[:, ymax] - before_limiting[:, ymin])
                            after_area = (batch_y[i][:, xmax] - batch_y[i][:, xmin]) * (
                            batch_y[i][:, ymax] - batch_y[i][:, ymin])
                            if include_thresh == 0:
                                batch_y[i] = batch_y[i][
                                    after_area > include_thresh * before_area]  # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                            else:
                                batch_y[i] = batch_y[i][
                                    after_area >= include_thresh * before_area]  # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if scale:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - scale[2]):
                        # Rescale the image and return the transformation matrix M so we can use it to adjust the box coordinates
                        batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                        # Adjust the box coordinates
                        # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                        toplefts = np.array([batch_y[i][:, xmin], batch_y[i][:, ymin], np.ones(batch_y[i].shape[0])])
                        bottomrights = np.array(
                            [batch_y[i][:, xmax], batch_y[i][:, ymax], np.ones(batch_y[i].shape[0])])
                        new_toplefts = (np.dot(M, toplefts)).T
                        new_bottomrights = (np.dot(M, bottomrights)).T
                        batch_y[i][:, [xmin, ymin]] = new_toplefts.astype(np.int)
                        batch_y[i][:, [xmax, ymax]] = new_bottomrights.astype(np.int)
                        # Limit the box coordinates to lie within the image boundaries
                        if limit_boxes and (
                            scale_factor > 1):  # We don't need to do any limiting in case we shrunk the image
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:, [xmin, xmax]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:, [xmin, xmax]] = x_coords
                            y_coords = batch_y[i][:, [ymin, ymax]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:, [ymin, ymax]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (
                            before_limiting[:, ymax] - before_limiting[:, ymin])
                            after_area = (batch_y[i][:, xmax] - batch_y[i][:, xmin]) * (
                            batch_y[i][:, ymax] - batch_y[i][:, ymin])
                            if include_thresh == 0:
                                batch_y[i] = batch_y[i][
                                    after_area > include_thresh * before_area]  # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                            else:
                                batch_y[i] = batch_y[i][
                                    after_area >= include_thresh * before_area]  # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if random_crop:
                    # Compute how much room we have in both dimensions to make a random crop.
                    # A negative number here means that we want to crop out a patch that is larger than the original image in the respective dimension,
                    # in which case we will create a black background canvas onto which we will randomly place the image.
                    y_range = img_height - random_crop[0]
                    x_range = img_width - random_crop[1]
                    # Keep track of the number of trials and of whether or not the most recent crop contains at least one object
                    min_1_object_fulfilled = False
                    trial_counter = 0
                    while (not min_1_object_fulfilled) and (trial_counter < random_crop[3]):
                        # Select a random crop position from the possible crop positions
                        if y_range >= 0:
                            crop_ymin = np.random.randint(0,
                                                          y_range + 1)  # There are y_range + 1 possible positions for the crop in the vertical dimension
                        else:
                            crop_ymin = np.random.randint(0,
                                                          -y_range + 1)  # The possible positions for the image on the background canvas in the vertical dimension
                        if x_range >= 0:
                            crop_xmin = np.random.randint(0,
                                                          x_range + 1)  # There are x_range + 1 possible positions for the crop in the horizontal dimension
                        else:
                            crop_xmin = np.random.randint(0,
                                                          -x_range + 1)  # The possible positions for the image on the background canvas in the horizontal dimension
                        # Perform the crop
                        if y_range >= 0 and x_range >= 0:  # If the patch to be cropped out is smaller than the original image in both dimenstions, we just perform a regular crop
                            # Crop the image
                            patch_X = np.copy(
                                batch_X[i][crop_ymin:crop_ymin + random_crop[0], crop_xmin:crop_xmin + random_crop[1]])
                            # Translate the box coordinates into the new coordinate system: Cropping shifts the origin by `(crop_ymin, crop_xmin)`
                            patch_y = np.copy(batch_y[i])
                            patch_y[:, [ymin, ymax]] -= crop_ymin
                            patch_y[:, [xmin, xmax]] -= crop_xmin
                            # Limit the box coordinates to lie within the new image boundaries
                            if limit_boxes:
                                # Both the x- and y-coordinates might need to be limited
                                before_limiting = np.copy(patch_y)
                                y_coords = patch_y[:, [ymin, ymax]]
                                y_coords[y_coords < 0] = 0
                                y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                patch_y[:, [ymin, ymax]] = y_coords
                                x_coords = patch_y[:, [xmin, xmax]]
                                x_coords[x_coords < 0] = 0
                                x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                patch_y[:, [xmin, xmax]] = x_coords
                        elif y_range >= 0 and x_range < 0:  # If the crop is larger than the original image in the horizontal dimension only,...
                            # Crop the image
                            patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin + random_crop[
                                0]])  # ...crop the vertical dimension just as before,...
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]),
                                              dtype=np.uint8)  # ...generate a blank background image to place the patch onto,...
                            canvas[:,
                            crop_xmin:crop_xmin + img_width] = patch_X  # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                            patch_X = canvas
                            # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(crop_ymin, -crop_xmin)`
                            patch_y = np.copy(batch_y[i])
                            patch_y[:, [ymin, ymax]] -= crop_ymin
                            patch_y[:, [xmin, xmax]] += crop_xmin
                            # Limit the box coordinates to lie within the new image boundaries
                            if limit_boxes:
                                # Only the y-coordinates might need to be limited
                                before_limiting = np.copy(patch_y)
                                y_coords = patch_y[:, [ymin, ymax]]
                                y_coords[y_coords < 0] = 0
                                y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                patch_y[:, [ymin, ymax]] = y_coords
                        elif y_range < 0 and x_range >= 0:  # If the crop is larger than the original image in the vertical dimension only,...
                            # Crop the image
                            patch_X = np.copy(batch_X[i][:, crop_xmin:crop_xmin + random_crop[
                                1]])  # ...crop the horizontal dimension just as in the first case,...
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]),
                                              dtype=np.uint8)  # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin + img_height,
                            :] = patch_X  # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                            patch_X = canvas
                            # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, crop_xmin)`
                            patch_y = np.copy(batch_y[i])
                            patch_y[:, [ymin, ymax]] += crop_ymin
                            patch_y[:, [xmin, xmax]] -= crop_xmin
                            # Limit the box coordinates to lie within the new image boundaries
                            if limit_boxes:
                                # Only the x-coordinates might need to be limited
                                before_limiting = np.copy(patch_y)
                                x_coords = patch_y[:, [xmin, xmax]]
                                x_coords[x_coords < 0] = 0
                                x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                patch_y[:, [xmin, xmax]] = x_coords
                        else:  # If the crop is larger than the original image in both dimensions,...
                            patch_X = np.copy(batch_X[i])
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]),
                                              dtype=np.uint8)  # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin + img_height,
                            crop_xmin:crop_xmin + img_width] = patch_X  # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                            patch_X = canvas
                            # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, -crop_xmin)`
                            patch_y = np.copy(batch_y[i])
                            patch_y[:, [ymin, ymax]] += crop_ymin
                            patch_y[:, [xmin, xmax]] += crop_xmin
                            # Note that no limiting is necessary in this case
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        if limit_boxes and (y_range >= 0 or x_range >= 0):
                            before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (
                            before_limiting[:, ymax] - before_limiting[:, ymin])
                            after_area = (patch_y[:, xmax] - patch_y[:, xmin]) * (patch_y[:, ymax] - patch_y[:, ymin])
                            if include_thresh == 0:
                                patch_y = patch_y[
                                    after_area > include_thresh * before_area]  # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                            else:
                                patch_y = patch_y[
                                    after_area >= include_thresh * before_area]  # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all
                        trial_counter += 1  # We've just used one of our trials
                        # Check if we have found a valid crop
                        if random_crop[
                            2] == 0:  # If `min_1_object == 0`, break out of the while loop after the first loop because we are fine with whatever crop we got
                            batch_X[i] = patch_X  # The cropped patch becomes our new batch item
                            batch_y[i] = patch_y  # The adjusted boxes become our new labels for this batch item
                            # Update the image size so that subsequent transformations can work correctly
                            img_height = random_crop[0]
                            img_width = random_crop[1]
                            break
                        elif len(
                                patch_y) > 0:  # If we have at least one object left, this crop is valid and we can stop
                            min_1_object_fulfilled = True
                            batch_X[i] = patch_X  # The cropped patch becomes our new batch item
                            batch_y[i] = patch_y  # The adjusted boxes become our new labels for this batch item
                            # Update the image size so that subsequent transformations can work correctly
                            img_height = random_crop[0]
                            img_width = random_crop[1]
                        elif (trial_counter >= random_crop[3]) and (
                        not i in batch_items_to_remove):  # If we've reached the trial limit and still not found a valid crop, remove this image from the batch
                            batch_items_to_remove.append(i)

                if crop:
                    # Crop the image
                    batch_X[i] = np.copy(batch_X[i][crop[0]:img_height - crop[1], crop[2]:img_width - crop[3]])
                    # Translate the box coordinates into the new coordinate system if necessary: The origin is shifted by `(crop[0], crop[2])` (i.e. by the top and left crop values)
                    # If nothing was cropped off from the top or left of the image, the coordinate system stays the same as before
                    if crop[0] > 0:
                        batch_y[i][:, [ymin, ymax]] -= crop[0]
                    if crop[2] > 0:
                        batch_y[i][:, [xmin, xmax]] -= crop[2]
                    # Update the image size so that subsequent transformations can work correctly
                    img_height -= crop[0] + crop[1]
                    img_width -= crop[2] + crop[3]
                    # Limit the box coordinates to lie within the new image boundaries
                    if limit_boxes:
                        before_limiting = np.copy(batch_y[i])
                        # We only need to check those box coordinates that could possibly have been affected by the cropping
                        # For example, if we only crop off the top and/or bottom of the image, there is no need to check the x-coordinates
                        if crop[0] > 0:
                            y_coords = batch_y[i][:, [ymin, ymax]]
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:, [ymin, ymax]] = y_coords
                        if crop[1] > 0:
                            y_coords = batch_y[i][:, [ymin, ymax]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            batch_y[i][:, [ymin, ymax]] = y_coords
                        if crop[2] > 0:
                            x_coords = batch_y[i][:, [xmin, xmax]]
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:, [xmin, xmax]] = x_coords
                        if crop[3] > 0:
                            x_coords = batch_y[i][:, [xmin, xmax]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            batch_y[i][:, [xmin, xmax]] = x_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:, xmax] - before_limiting[:, xmin]) * (
                        before_limiting[:, ymax] - before_limiting[:, ymin])
                        after_area = (batch_y[i][:, xmax] - batch_y[i][:, xmin]) * (
                        batch_y[i][:, ymax] - batch_y[i][:, ymin])
                        if include_thresh == 0:
                            batch_y[i] = batch_y[i][
                                after_area > include_thresh * before_area]  # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                        else:
                            batch_y[i] = batch_y[i][
                                after_area >= include_thresh * before_area]  # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if resize:
                    batch_X[i] = cv2.resize(batch_X[i], dsize=resize)
                    batch_y[i][:, [xmin, xmax]] = (batch_y[i][:, [xmin, xmax]] * (resize[0] / img_width)).astype(np.int)
                    batch_y[i][:, [ymin, ymax]] = (batch_y[i][:, [ymin, ymax]] * (resize[1] / img_height)).astype(
                        np.int)
                    img_width, img_height = resize  # Updating these at this point is unnecessary, but it's one fewer source of error if this method gets expanded in the future

                if gray:
                    batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), 3)

            # If any batch items need to be removed because of failed random cropping, remove them now.
            for j in sorted(batch_items_to_remove, reverse=True):
                batch_X.pop(j)
                batch_y.pop(j)  # This isn't efficient, but this should hopefully not need to be done often anyway

            if train:  # During training we need the encoded labels instead of the format that `batch_y` has
                if ssd_box_encoder is None:
                    raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                y_true = ssd_box_encoder.encode_y(
                    batch_y)  # Encode the labels into the `y_true` tensor that the cost function needs

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes.
            #          At this point, all images have to have the same size, otherwise you will get an error during training.
            if train:
                if diagnostics:
                    yield (np.array(batch_X), y_true, batch_y, this_filenames, original_images, original_labels)
                else:
                    yield (np.array(batch_X), y_true)
            else:
                yield (np.array(batch_X), batch_y, this_filenames)

    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames and the list of labels.
        '''
        return self.filenames, self.labels

    def get_n_samples(self):
        '''
        Returns:
            The number of image files in the initialized dataset.
        '''
        return len(self.filenames)

    def process_offline(self,
                        dest_path='',
                        start=0,
                        stop='all',
                        crop=None,
                        equalize=None,
                        brightness=None,
                        flip=None,
                        translate=None,
                        scale=None,
                        resize=None,
                        gray=False,
                        limit_boxes=True,
                        include_thresh=0.3,
                        diagnostics=False):
        '''
        Perform offline image processing.

        This function the same image processing capabilities as the generator function above,
        but it performs the processing on all items in `filenames` starting at index `start`
        until index `stop` and saves the processed images to disk. The labels are adjusted
        accordingly.

        Processing images offline is useful to reduce the amount of work done by the batch
        generator and thus can speed up training. For example, transformations that are performed
        on all images in a deterministic way, such as resizing or cropping, should be done offline.

        Arguments:
            dest_path (str, optional): The destination directory where the processed images
                and `labels.csv` should be saved, ending on a slash.
            start (int, optional): The inclusive start index from which onward to process the
                items in `filenames`. Defaults to 0.
            stop (int, optional): The exclusive stop index until which to process the
                items in `filenames`. Defaults to 'all', meaning to process all items until the
                end of the list.

        For a description of the other arguments, please refer to the documentation of `generate_batch()` above.

        Returns:
            `None`, but saves all processed images as JPEG files to the specified destination
            directory and generates a `labels.csv` CSV file that is saved to the same directory.
            The format of the lines in the destination CSV file is the same as that of the
            source CSV file, i.e. `[frame, xmin, xmax, ymin, ymax, class_id]`.
        '''

        import gc

        targets_for_csv = []
        if stop == 'all':
            stop = len(self.filenames)

        if diagnostics:
            processed_images = []
            original_images = []
            processed_labels = []

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        for k, filename in enumerate(self.filenames[start:stop]):
            i = k + start
            with Image.open('{}'.format(filename)) as img:
                image = np.array(img)
            targets = np.copy(self.labels[i])

            if diagnostics:
                original_images.append(image)

            img_height, img_width, ch = image.shape

            if equalize:
                image = histogram_eq(image)

            if brightness:
                p = np.random.uniform(0, 1)
                if p >= (1 - brightness[2]):
                    image = _brightness(image, min=brightness[0], max=brightness[1])

            # Could easily be extended to also allow vertical flipping, but I'm not convinced of the
            # usefulness of vertical flipping either empirically or theoretically, so I'm going for simplicity.
            # If you want to allow vertical flipping, just change this function to pass the respective argument
            # to `_flip()`.
            if flip:
                p = np.random.uniform(0, 1)
                if p >= (1 - flip):
                    image = _flip(image)
                    targets[:, [0, 1]] = img_width - targets[:, [1, 0]]  # xmin and xmax are swapped when mirrored

            if translate:
                p = np.random.uniform(0, 1)
                if p >= (1 - translate[2]):
                    image, xshift, yshift = _translate(image, translate[0], translate[1])
                    targets[:, [0, 1]] += xshift
                    targets[:, [2, 3]] += yshift
                    if limit_boxes:
                        before_limiting = np.copy(targets)
                        x_coords = targets[:, [0, 1]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:, [0, 1]] = x_coords
                        y_coords = targets[:, [2, 3]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:, [2, 3]] = y_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:, 1] - before_limiting[:, 0]) * (
                        before_limiting[:, 3] - before_limiting[:, 2])
                        after_area = (targets[:, 1] - targets[:, 0]) * (targets[:, 3] - targets[:, 2])
                        targets = targets[after_area >= include_thresh * before_area]

            if scale:
                p = np.random.uniform(0, 1)
                if p >= (1 - scale[2]):
                    image, M, scale_factor = _scale(image, scale[0], scale[1])
                    # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                    toplefts = np.array([targets[:, 0], targets[:, 2], np.ones(targets.shape[0])])
                    bottomrights = np.array([targets[:, 1], targets[:, 3], np.ones(targets.shape[0])])
                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T
                    targets[:, [0, 2]] = new_toplefts.astype(np.int)
                    targets[:, [1, 3]] = new_bottomrights.astype(np.int)
                    if limit_boxes and (
                        scale_factor > 1):  # We don't need to do any limiting in case we shrunk the image
                        before_limiting = np.copy(targets)
                        x_coords = targets[:, [0, 1]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:, [0, 1]] = x_coords
                        y_coords = targets[:, [2, 3]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:, [2, 3]] = y_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:, 1] - before_limiting[:, 0]) * (
                        before_limiting[:, 3] - before_limiting[:, 2])
                        after_area = (targets[:, 1] - targets[:, 0]) * (targets[:, 3] - targets[:, 2])
                        targets = targets[after_area >= include_thresh * before_area]

            if crop:
                image = image[crop[0]:img_height - crop[1], crop[2]:img_width - crop[3]]
                if limit_boxes:  # Adjust boxes affected by cropping and remove those that will no longer be in the image
                    before_limiting = np.copy(targets)
                    if crop[0] > 0:
                        y_coords = targets[:, [2, 3]]
                        y_coords[y_coords < crop[0]] = crop[0]
                        targets[:, [2, 3]] = y_coords
                    if crop[1] > 0:
                        y_coords = targets[:, [2, 3]]
                        y_coords[y_coords >= (img_height - crop[1])] = img_height - crop[1] - 1
                        targets[:, [2, 3]] = y_coords
                    if crop[2] > 0:
                        x_coords = targets[:, [0, 1]]
                        x_coords[x_coords < crop[2]] = crop[2]
                        targets[:, [0, 1]] = x_coords
                    if crop[3] > 0:
                        x_coords = targets[:, [0, 1]]
                        x_coords[x_coords >= (img_width - crop[3])] = img_width - crop[3] - 1
                        targets[:, [0, 1]] = x_coords
                    # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                    # process that they don't serve as useful training examples anymore, because too little of them is
                    # visible. We'll remove all boxes that we had to limit so much that their area is less than
                    # `include_thresh` of the box area before limiting.
                    before_area = (before_limiting[:, 1] - before_limiting[:, 0]) * (
                    before_limiting[:, 3] - before_limiting[:, 2])
                    after_area = (targets[:, 1] - targets[:, 0]) * (targets[:, 3] - targets[:, 2])
                    targets = targets[after_area >= include_thresh * before_area]
                # Now adjust the box coordinates for the new image size post cropping
                if crop[0] > 0:
                    targets[:, [2, 3]] -= crop[0]
                if crop[2] > 0:
                    targets[:, [0, 1]] -= crop[2]
                img_height -= crop[0] - crop[1]
                img_width -= crop[2] - crop[3]

            if resize:
                image = cv2.resize(image, dsize=resize)
                targets[:, [0, 1]] = (targets[:, [0, 1]] * (resize[0] / img_width)).astype(np.int)
                targets[:, [2, 3]] = (targets[:, [2, 3]] * (resize[1] / img_height)).astype(np.int)

            if gray:
                image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 3)

            if diagnostics:
                processed_images.append(image)
                processed_labels.append(targets)

            img = Image.fromarray(image.astype(np.uint8))
            img.save('{}{}'.format(dest_path, filename), 'JPEG', quality=90)
            del image
            del img
            gc.collect()

            # Transform the labels back to the original CSV file format:
            # One line per ground truth box, i.e. possibly multiple lines per image
            for target in targets:
                target = list(target)
                target = [filename] + target
                targets_for_csv.append(target)

        with open('{}labels.csv'.format(dest_path), 'w', newline='') as csvfile:
            labelswriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labelswriter.writerow(['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
            labelswriter.writerows(targets_for_csv)

        if diagnostics:
            print("Image processing completed.")
            return np.array(processed_images), np.array(original_images), np.array(targets_for_csv), processed_labels
        else:
            print("Image processing completed.")


def iou(boxes1, boxes2, coords='centroids'):
    '''
    Compute the intersection-over-union similarity (also known as Jaccard similarity)
    of two axis-aligned 2D rectangular boxes or of multiple axis-aligned 2D rectangular
    boxes contained in two arrays with broadcast-compatible shapes.

    Three common use cases would be to compute the similarities for 1 vs. 1, 1 vs. `n`,
    or `n` vs. `n` boxes. The two arguments are symmetric.

    Arguments:
        boxes1 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes2`.
        boxes2 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes1`.
        coords (str, optional): The coordinate format in the input arrays. Can be either 'centroids' for the format
            `(cx, cy, w, h)` or 'minmax' for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.

    Returns:
        A 1D Numpy array of dtype float containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and `boxes2`.
        0 means there is no overlap between two given boxes, 1 means their coordinates are identical.
    '''

    if len(boxes1.shape) > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    if coords == 'centroids':
        # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
        boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2minmax')
        boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2minmax')
    elif coords != 'minmax':
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:,2]))
    union = (boxes1[:,1] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,2]) + (boxes2[:,1] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,2]) - intersection

    return intersection / union


def convert_coordinates(tensor, start_index, conversion='minmax2centroids'):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    two supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (cx, cy, w, h) - the 'centroids' format

    Note that converting from one of the supported formats to another and back is
    an identity operation up to possible rounding errors for integer tensors.

    Arguments:
        tensor (np.array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids'
            or 'centroids2minmax'. Defaults to 'minmax2centroids'.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def convert_coordinates2(tensor, start_index, conversion='minmax2centroids'):
    """
    A pure matrix multiplication implementation of `convert_coordinates()`.

    Although elegant, it turns out to be marginally slower on average than
    `convert_coordinates()`. Note that the two matrices below are each other's
    multiplicative inverse.

    For details please refer to the documentation of `convert_coordinates()`.
    """
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        M = np.array([[0.5, 0. , -1.,  0.],
                      [0.5, 0. ,  1.,  0.],
                      [0. , 0.5,  0., -1.],
                      [0. , 0.5,  0.,  1.]])
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    elif conversion == 'centroids2minmax':
        M = np.array([[ 1. , 1. ,  0. , 0. ],
                      [ 0. , 0. ,  1. , 1. ],
                      [-0.5, 0.5,  0. , 0. ],
                      [ 0. , 0. , -0.5, 0.5]])
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1


def greedy_nms(y_pred_decoded, iou_threshold=0.45, coords='minmax'):
    """
    Perform greedy non-maximum suppression on the input boxes.

    Greedy NMS works by selecting the box with the highest score and
    removing all boxes around it that are too close to it measured by IoU-similarity.
    Out of the boxes that are left over, once again the one with the highest
    score is selected and so on, until no boxes with too much overlap are left.

    This is a basic, straight-forward NMS algorithm that is relatively efficient,
    but it has a number of downsides. One of those downsides is that the box with
    the highest score might not always be the box with the best fit to the object.
    There are more sophisticated NMS techniques like [this one](https://lirias.kuleuven.be/bitstream/123456789/506283/1/3924_postprint.pdf)
    that use a combination of nearby boxes, but in general there will probably
    always be a trade-off between speed and quality for any given NMS technique.

    Arguments:
        y_pred_decoded (list): A batch of decoded predictions. For a given batch size `n` this
            is a list of length `n` where each list element is a 2D Numpy array.
            For a batch item with `k` predicted boxes this 2D Numpy array has
            shape `(k, 6)`, where each row contains the coordinates of the respective
            box in the format `[class_id, score, xmin, xmax, ymin, ymax]`.
            Technically, the number of columns doesn't have to be 6, it can be
            arbitrary as long as the first four elements of each row are
            `xmin`, `xmax`, `ymin`, `ymax` (in this order) and the last element
            is the score assigned to the prediction. Note that this function is
            agnostic to the scale of the score or what it represents.
        iou_threshold (float, optional): All boxes with a Jaccard similarity of
            greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score.
            Defaults to 0.45 following the paper.
        coords (str, optional): The coordinate format of `y_pred_decoded`.
            Can be one of the formats supported by `iou()`. Defaults to 'minmax'.

    Returns:
        The predictions after removing non-maxima. The format is the same as the input format.
    """
    y_pred_decoded_nms = []
    for batch_item in y_pred_decoded: # For the labels of each batch item...
        boxes_left = np.copy(batch_item)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        y_pred_decoded_nms.append(np.array(maxima))

    return y_pred_decoded_nms


def _greedy_nms(predictions, iou_threshold=0.45, coords='minmax'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_y()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)


def _greedy_nms2(predictions, iou_threshold=0.45, coords='minmax'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function in `decode_y2()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)


def decode_y(y_pred,
             confidence_thresh=0.01,
             iou_threshold=0.45,
             top_k=200,
             input_coords='centroids',
             normalize_coords=False,
             img_height=None,
             img_width=None):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_y2()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage. Defaults to 0.01, following the paper.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score. Defaults to 0.45 following the paper.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax'
            for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,-4:-2] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,-2:] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='minmax') # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        pred = np.concatenate(pred, axis=0)
        if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
            top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded


def decode_y2(y_pred,
              confidence_thresh=0.5,
              iou_threshold=0.45,
              top_k='all',
              input_coords='centroids',
              normalize_coords=False,
              img_height=None,
              img_width=None):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Optionally performs confidence thresholding and greedy non-maximum suppression afte the decoding stage.

    Note that the decoding procedure used here is not the same as the procedure used in the original Caffe implementation.
    The procedure used here assigns every box its highest confidence as the class and then removes all boxes fro which
    the highest confidence is the background class. This results in less work for the subsequent non-maximum suppression,
    because the vast majority of the predictions will be filtered out just by the fact that their highest confidence is
    for the background class. It is much more efficient than the procedure of the original implementation, but the
    results may also differ.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in any positive
            class required for a given box to be considered a positive prediction. A lower value will result
            in better recall, while a higher value will result in better precision. Do not use this parameter with the
            goal to combat the inevitably many duplicates that an SSD will produce, the subsequent non-maximum suppression
            stage will take care of those. Defaults to 0.5.
        iou_threshold (float, optional): `None` or a float in [0,1]. If `None`, no non-maximum suppression will be
            performed. If not `None`, greedy NMS will be performed after the confidence thresholding stage, meaning
            all boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score. Defaults to 0.45.
        top_k (int, optional): 'all' or an integer with number of highest scoring predictions to be kept for each batch item
            after the non-maximum suppression stage. Defaults to 'all', in which case all predictions left after the NMS stage
            will be kept.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax'
            for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:,:,-14:-8]) # Slice out the four offset predictions plus two elements whereto we'll write the class IDs and confidences in the next step
    y_pred_converted[:,:,0] = np.argmax(y_pred[:,:,:-12], axis=-1) # The indices of the highest confidence values in the one-hot class vectors are the class ID
    y_pred_converted[:,:,1] = np.amax(y_pred[:,:,:-12], axis=-1) # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        y_pred_converted[:,:,[4,5]] = np.exp(y_pred_converted[:,:,[4,5]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_converted[:,:,[4,5]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_converted[:,:,[2,3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_converted[:,:,[2,3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2minmax')
    elif input_coords == 'minmax':
        y_pred_converted[:,:,2:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_converted[:,:,[2,3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[4,5]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,2:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_converted[:,:,2:4] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:,:,4:] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,1] >= confidence_thresh] # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
        if iou_threshold: # ...if an IoU threshold is set...
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='minmax') # ...perform NMS on the remaining boxes.
        if top_k != 'all' and boxes.shape[0] > top_k: # If we have more than `top_k` results left at this point...
            top_k_indices = np.argpartition(boxes[:,1], kth=boxes.shape[0]-top_k, axis=0)[boxes.shape[0]-top_k:] # ...get the indices of the `top_k` highest-scoring boxes...
            boxes = boxes[top_k_indices] # ...and keep only those boxes...
        y_pred_decoded.append(boxes) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded


class SSDBoxEncoder:
    """
    A class to transform ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD model, and to transform predictions of the SSD model back
    to the original format of the input labels.

    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    """

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=None,
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 variances=None,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.3,
                 coords='centroids',
                 normalize_coords=False):
        """
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of classes including the background class.
            predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Defaults to 0.1. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last predictor layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last predictor layer
                if `two_boxes_for_ar1` is `True`. Defaults to 0.9. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect.
            scales (list, optional): A list of floats containing scaling factors per convolutional predictor layer.
                This list must be one element longer than the number of predictor layers. The first `k` elements are the
                scaling factors for the `k` predictor layers, while the last element is used for the second box
                for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional
                last scaling factor must be passed either way, even if it is not being used.
                Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
                `max_scale`. All scaling factors must be greater than zero. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Defaults to [0.5, 1.0, 2.0]. Note that you should
                set the aspect ratios such that the resulting anchor box shapes very roughly correspond to the shapes of the
                objects you are trying to detect. For many standard detection tasks, the default values will yield good
                results.
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Defaults to `None`. Note that you should
                set the aspect ratios such that the resulting anchor box shapes very roughly correspond to the shapes of the
                objects you are trying to detect. For many standard detection tasks, the default values will yield good
                results.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
                Defaults to `True`.
            variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
                to be precise) for the encoded ground truth (i.e. target) box coordinates. A variance value of 1.0 would apply
                no scaling at all to the targets, while values in (0,1) upscale the encoded targets and values greater than 1.0
                downscale the encoded targets. If you want to reproduce the configuration of the original SSD,
                set this to `[0.1, 0.1, 0.2, 0.2]`, provided the coordinate format is 'centroids'. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box. Defaults to 0.5.
            neg_iou_threshold (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
                of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
                and height) or 'minmax' for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
            normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be within [0,1].
                This way learning becomes independent of the input image size. Defaults to `False`.
        """
        if variances is None:
            variances = [1.0, 1.0, 1.0, 1.0]
        if aspect_ratios_global is None:
            aspect_ratios_global = [0.5, 1.0, 2.0]
        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != len(predictor_sizes)+1): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(len(scales), len(predictor_sizes)+1))

        if aspect_ratios_per_layer:
            if (len(aspect_ratios_per_layer) != len(predictor_sizes)): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (coords == 'minmax' or coords == 'centroids'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if aspect_ratios_per_layer:
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

    def generate_anchor_boxes(self,
                              batch_size,
                              feature_map_size,
                              aspect_ratios,
                              this_scale,
                              next_scale,
                              diagnostics=False):
        '''
        Compute an array of the spatial positions and sizes of the anchor boxes for one particular classification
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            batch_size (int): The batch size.
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, two additional outputs will be returned.
                1) An array containing `(width, height)` for each box aspect ratio.
                2) A tuple `(cell_height, cell_width)` meaning how far apart the box centroids are placed
                   vertically and horizontally.
                This information is useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        n_boxes = len(aspect_ratios)
        for ar in aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular anchor box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w,h))
                # Add 1 to `n_boxes` since we seem to have two boxes for aspect ratio 1
                n_boxes += 1
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width/2, self.img_width-cell_width/2, feature_map_size[1])
        cy = np.linspace(cell_height/2, self.img_height-cell_height/2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.limit_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 1]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 1]] = x_coords
            y_coords = boxes_tensor[:,:,:,[2, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[2, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, :2] /= self.img_width
            boxes_tensor[:, :, :, 2:] /= self.img_height

        if self.coords == 'centroids':
            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
            # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='minmax2centroids')

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 4)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))

        # Now reshape the 5D tensor above into a 3D tensor of shape
        # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
        # order of the tensor content will be identical to the order obtained from the reshaping operation
        # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
        # use the same default index order, which is C-like index ordering)
        boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))

        if diagnostics:
            return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor

    def generate_encode_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the conv net model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 8)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 8` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes.
        '''

        # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
        #    If `scales` is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`
        if self.scales is None:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)

        # 2: For each conv predictor layer (i.e. for each scale factor) get the tensors for
        #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = [] # List to hold the box widths and heights
            cell_sizes = [] # List to hold horizontal and vertical distances between any two boxes
            if self.aspect_ratios_per_layer: # If individual aspect ratios are given per layer, we need to pass them to `generate_anchor_boxes()` accordingly
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
            else: # Use the same global aspect ratio list for all layers
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_global,
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
        else:
            if self.aspect_ratios_per_layer:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))
            else:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_global,
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))

        boxes_tensor = np.concatenate(boxes_tensor, axis=1) # Concatenate the anchor tensors from the individual layers to one

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, wh_list, cell_sizes
        else:
            return y_encode_template

    def encode_y(self, ground_truth_labels):
        '''
        Convert ground truth bounding box data into a suitable format to train an SSD model.

        For each image in the batch, each ground truth bounding box belonging to that image will be compared against each
        anchor box in a template with respect to their jaccard similarity. If the jaccard similarity is greater than
        or equal to the set threshold, the boxes will be matched, meaning that the ground truth box coordinates and class
        will be written to the the specific position of the matched anchor box in the template.

        The class for all anchor boxes for which there was no match with any ground truth box will be set to the
        background class, except for those anchor boxes whose IoU similarity with any ground truth box is higher than
        the set negative threshold (see the `neg_iou_threshold` argument in `__init__()`).

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, xmax, ymin, ymax)`, and `class_id` must be an integer greater than 0 for all boxes
                as class_id 0 is reserved for the background class.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, and the last four elements are just dummy elements.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels), diagnostics=False)
        y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

        class_vector = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]): # For each batch item...
            available_boxes = np.ones((y_encode_template.shape[1])) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
            negative_boxes = np.ones((y_encode_template.shape[1])) # 1 for all negative boxes, 0 otherwise
            for true_box in ground_truth_labels[i]: # For each ground truth box belonging to the current batch item...
                true_box = true_box.astype(np.float)
                if (true_box[2] - true_box[1] == 0) or (true_box[4] - true_box[3] == 0): continue # Protect ourselves against bad ground truth data: boxes with width or height equal to zero
                if self.normalize_coords:
                    true_box[1:3] /= self.img_width # Normalize xmin and xmax to be within [0,1]
                    true_box[3:5] /= self.img_height # Normalize ymin and ymax to be within [0,1]
                if self.coords == 'centroids':
                    true_box = convert_coordinates(true_box, start_index=1, conversion='minmax2centroids')
                similarities = iou(y_encode_template[i,:,-12:-8], true_box[1:], coords=self.coords) # The iou similarities for all anchor boxes
                negative_boxes[similarities >= self.neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:-8] = np.concatenate((class_vector[int(true_box[0])], true_box[1:]), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1

        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        if self.coords == 'centroids':
            y_encoded[:,:,[-12,-11]] -= y_encode_template[:,:,[-12,-11]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-12,-11]] /= y_encode_template[:,:,[-10,-9]] * y_encode_template[:,:,[-4,-3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:,:,[-10,-9]] /= y_encode_template[:,:,[-10,-9]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-10,-9]] = np.log(y_encoded[:,:,[-10,-9]]) / y_encode_template[:,:,[-2,-1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        else:
            y_encoded[:,:,-12:-8] -= y_encode_template[:,:,-12:-8] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-12,-11]] /= np.expand_dims(y_encode_template[:,:,-11] - y_encode_template[:,:,-12], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-10,-9]] /= np.expand_dims(y_encode_template[:,:,-9] - y_encode_template[:,:,-10], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:,:,-12:-8] /= y_encode_template[:,:,-4:] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively

        return y_encoded
