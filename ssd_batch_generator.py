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

from collections import defaultdict
import warnings
import numpy as np
import cv2
import random
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
try:
    import json
except ImportError:
    warnings.warn("'json' module is missing. The JSON-parser will be unavailable.")
try:
    from bs4 import BeautifulSoup
except ImportError:
    warnings.warn("'BeautifulSoup' module is missing. The XML-parser will be unavailable.")
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")

# Image processing functions used by the generator to perform the following image manipulations:
# - Translation
# - Horizontal flip
# - Scaling
# - Brightness change
# - Histogram contrast equalization

def _translate(image, horizontal=(0,40), vertical=(0,10)):
    '''
    Randomly translate the input image horizontally and vertically.

    Arguments:
        image (array-like): The image to be translated.
        horizontal (int tuple, optinal): A 2-tuple `(min, max)` with the minimum
            and maximum horizontal translation. A random translation value will
            be picked from a uniform distribution over [min, max].
        vertical (int tuple, optional): Analog to `horizontal`.

    Returns:
        The translated image and the horzontal and vertical shift values.
    '''
    rows,cols,ch = image.shape

    x = np.random.randint(horizontal[0], horizontal[1]+1)
    y = np.random.randint(vertical[0], vertical[1]+1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    M = np.float32([[1,0,x_shift],[0,1,y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift

def _flip(image, orientation='horizontal'):
    '''
    Flip the input image horizontally or vertically.
    '''
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)

def _scale(image, min=0.9, max=1.1):
    '''
    Scale the input image by a random factor picked from a uniform distribution
    over [min, max].

    Returns:
        The scaled image, the associated warp matrix, and the scaling value.
    '''

    rows,cols,ch = image.shape

    #Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale

def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def histogram_eq(image):
    '''
    Perform histogram equalization on the input image.

    See https://en.wikipedia.org/wiki/Histogram_equalization.
    '''

    image1 = np.copy(image)

    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)

    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])

    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

    return image1

class BatchGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides two methods to parse annotation data: A general-purpose CSV parser
    and an XML parser for the Pascal VOC datasets. If the annotations of your dataset are
    in a format that is not supported by these parsers, you could just add another parser
    method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'],
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None):
        '''
        This class provides parser methods that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the dataset from CSV or XML files. If you already
        have the image filenames and labels in asuitable format (see argument descriptions below), you can pass
        them right here in the constructor, in which case you do not need to call any of the parser methods afterwards.

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            box_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated data. The expected strings are
                'xmin', 'ymin', 'xmax', 'ymax', 'class_id'. If you want to train the model, this
                must be the order that the box encoding class requires as input. Defaults to
                `['class_id', 'xmin', 'ymin', 'xmax', 'ymax']`. Note that even though the parser methods are
                able to produce different output formats, the SSDBoxEncoder currently requires the format
                `['class_id', 'xmin', 'ymin', 'xmax', 'ymax']`. This list only specifies the five box parameters
                that are relevant as training targets, a list of filenames is generated separately.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_dir`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file. Defaults to 'text'.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant. Defaults to `None`.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain the image
                IDs of the images in the dataset.
        '''
        self.box_output_format = box_output_format

        # The variables `self.filenames`, `self.labels`, and `self.image_ids` below store the output from the parsers.
        # This is the input for the `generate()`` method. `self.filenames` is a list containing all file names of the image samples (full paths).
        # Note that it does not contain the actual image files themselves.
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        # Setting `self.labels` is optional, the generator also works if `self.labels` remains `None`.

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
        else:
            self.filenames = []

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  input_format,
                  include_classes='all',
                  random_sample=False,
                  ret=False):
        '''
        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, class ID, xmin, xmax, ymin, ymax.
                The six items do not have to be in a specific order, but they must be the first six columns of
                each line. The order of these items in the CSV file must be specified in `input_format`.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `xmin` and `xmax` are the left-most and right-most absolute horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most absolute vertical coordinates of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located. Defaults to `None`.
            input_format (list): A list of six strings representing the order of the six items
                image file name, class ID, xmin, xmax, ymin, ymax in the input CSV file. The expected strings
                are 'image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'. Defaults to `None`.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes. Defaults to `False`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.

        Returns:
            None by default, optionally the image filenames and labels.
        '''

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.input_format = input_format
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename and an input_format
        if self.labels_filename is None or self.input_format is None:
            raise ValueError("`labels_filename` and/or `input_format` have not been set yet. You need to pass them as arguments.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread) # Skip the header row.
            for row in csvread: # For every line (i.e for every bounding box) in the CSV file...
                if self.include_classes == 'all' or int(row[self.input_format.index('class_id')].strip()) in self.include_classes: # If the class_id is among the classes that are to be included in the dataset...
                    box = [] # Store the box class and coordinates here
                    box.append(row[self.input_format.index('image_name')].strip()) # Select the image name column in the input format and append its content to `box`
                    for element in self.box_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
        current_labels = [] # The list where we collect all ground truth boxes for a given image
        add_to_dataset = False
        for i, box in enumerate(data):

            if box[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
            else: # If this box belongs to a new image file
                if random_sample: # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                current_labels = [] # Reset the labels list because this is a new file.
                current_file = box[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))

        if ret: # In case we want to return these
            return self.filenames, self.labels

    def parse_xml(self,
                  images_dirs,
                  image_set_filenames,
                  annotations_dirs=[],
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data format and XML tags of the Pascal VOC datasets.

        Arguments:
            images_dirs (list): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for Pascal VOC 2007, another that contains
                the images for Pascal VOC 2012, etc.).
            image_set_filenames (list): A list of strings, where each string is the path of the text file with the image
                set to be loaded. Must be one file per image directory given. These text files define what images in the
                respective image directories are to be part of the dataset and simply contains one image ID per line
                and nothing else.
            annotations_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains the annotations (XML files) that belong to the images in the respective image directories given.
                The directories must contain one XML file per image and the name of an XML file must be the image ID
                of the image it belongs to. The content of the XML files must be in the Pascal VOC format.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
        # Set class members.
        self.images_dirs = images_dirs
        self.annotations_dirs = annotations_dirs
        self.image_set_filenames = image_set_filenames
        self.classes = classes
        self.include_classes = include_classes

        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not annotations_dirs:
            self.labels = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

            # Loop over all images in this dataset.
            for image_id in image_ids:

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    # Parse the XML file for this image.
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                    #filename = soup.filename.text

                    boxes = [] # We'll store all boxes for this image here
                    objects = soup.find_all('object') # Get a list of all objects in this image

                    # Parse the data for each object
                    for obj in objects:
                        class_name = obj.find('name').text
                        class_id = self.classes.index(class_name)
                        # Check if this class is supposed to be included in the dataset
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
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
            return self.filenames, self.labels, self.image_ids

    def parse_json(self,
                   images_dirs,
                   annotations_filenames,
                   ground_truth_available=False,
                   include_classes = 'all',
                   ret=False):
        '''
        This is an JSON parser for the MS COCO datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the JSON format of the MS COCO datasets.

        Arguments:
            images_dirs (list, optional): A list of strings, where each string is the path of a directory that
                contains images that are to be part of the dataset. This allows you to aggregate multiple datasets
                into one (e.g. one directory that contains the images for MS COCO Train 2014, another one for MS COCO
                Val 2014, another one for MS COCO Train 2017 etc.).
            annotations_filenames (list): A list of strings, where each string is the path of the JSON file
                that contains the annotations for the images in the respective image directories given, i.e. one
                JSON file per image directory that contains the annotations for all images in that directory.
                The content of the JSON files must be in MS COCO object detection format. Note that these annotations
                files do not necessarily need to contain ground truth information. MS COCO also provides annotations
                files without ground truth information for the test datasets, called `image_info_[...].json`.
            ground_truth_available (bool, optional): Set `True` if the annotations files contain ground truth information.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.

        Returns:
            None by default, optionally the image filenames and labels.
        '''
        self.images_dirs = images_dirs
        self.annotations_filenames = annotations_filenames
        self.include_classes = include_classes
        # Erase data that might have been parsed before.
        self.filenames = []
        self.image_ids = []
        self.labels = []
        if not ground_truth_available:
            self.labels = None

        # Build the dictionaries that map between class names and class IDs.
        with open(annotations_filenames[0], 'r') as f:
            annotations = json.load(f)
        # Unfortunately the 80 MS COCO class IDs are not all consecutive. They go
        # from 1 to 90 and some numbers are skipped. Since the IDs that we feed
        # into a neural network must be consecutive, we'll save both the original
        # (non-consecutive) IDs as well as transformed maps.
        # We'll save both the map between the original
        self.cats_to_names = {} # The map between class names (values) and their original IDs (keys)
        self.classes_to_names = [] # A list of the class names with their indices representing the transformed IDs
        self.classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
        self.cats_to_classes = {} # A dictionary that maps between the original (keys) and the transformed IDs (values)
        self.classes_to_cats = {} # A dictionary that maps between the transformed (keys) and the original IDs (values)
        for i, cat in enumerate(annotations['categories']):
            self.cats_to_names[cat['id']] = cat['name']
            self.classes_to_names.append(cat['name'])
            self.cats_to_classes[cat['id']] = i + 1
            self.classes_to_cats[i + 1] = cat['id']

        # Iterate over all datasets.
        for images_dir, annotations_filename in zip(self.images_dirs, self.annotations_filenames):
            # Load the JSON file.
            with open(annotations_filename, 'r') as f:
                annotations = json.load(f)

            if ground_truth_available:
                # Create the annotations map, a dictionary whose keys are the image IDs
                # and whose values are the annotations for the respective image ID.
                image_ids_to_annotations = defaultdict(list)
                for annotation in annotations['annotations']:
                    image_ids_to_annotations[annotation['image_id']].append(annotation)

            # Iterate over all images in the dataset.
            for img in annotations['images']:

                self.filenames.append(os.path.join(images_dir, img['file_name']))
                self.image_ids.append(img['id'])

                if ground_truth_available:
                    # Get all annotations for this image.
                    annotations = image_ids_to_annotations[img['id']]
                    boxes = []
                    for annotation in annotations:
                        cat_id = annotation['category_id']
                        # Check if this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not cat_id in self.include_classes): continue
                        # Transform the original class ID to fit in the sequence of consecutive IDs.
                        class_id = self.cats_to_classes[cat_id]
                        xmin = annotation['bbox'][0]
                        ymin = annotation['bbox'][1]
                        width = annotation['bbox'][2]
                        height = annotation['bbox'][3]
                        # Compute `xmax` and `ymax`.
                        xmax = xmin + width
                        ymax = ymin + height
                        item_dict = {'image_name': img['file_name'],
                                     'image_id': img['id'],
                                     'class_id': class_id,
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
            return self.filenames, self.labels, self.image_ids

    def save_filenames_and_labels(self, filenames_path='filenames.pkl', labels_path=None, image_ids_path=None):
        '''
        Writes the current `filenames` and `labels` lists to the specified files.
        This is particularly useful for large datasets with annotations that are
        parsed from XML files, which can take quite long. If you'll be using the
        same dataset repeatedly, you don't want to have to parse the XML label
        files every time.

        Arguments:
            filenames_path (str): The path under which to save the filenames pickle.
            labels_path (str): The path under which to save the labels pickle.
            image_ids_path (str, optional): The path under which to save the image IDs pickle.
        '''
        with open(filenames_path, 'wb') as f:
            pickle.dump(self.filenames, f)
        if not labels_path is None:
            with open(labels_path, 'wb') as f:
                pickle.dump(self.labels, f)
        if not image_ids_path is None:
            with open(image_ids_path, 'wb') as f:
                pickle.dump(self.image_ids, f)

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 train=True,
                 ssd_box_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 convert_to_3_channels=True,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 max_crop_and_resize=False,
                 random_pad_and_resize=False,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 gray=False,
                 limit_boxes=True,
                 include_thresh=0.3,
                 subtract_mean=None,
                 divide_by_stddev=None,
                 swap_channels=False,
                 keep_images_without_gt=False):
        '''
        Generate batches of samples and corresponding labels indefinitely from
        lists of filenames and labels.

        Returns two Numpy arrays, one containing the next `batch_size` samples
        from `filenames`, the other containing the corresponding labels from
        `labels`.

        Can shuffle `filenames` and `labels` consistently after each complete pass.

        Can perform image transformations for data conversion and data augmentation.
        Each data augmentation process can set its own independent application probability.
        The transformations are performed in the order of their arguments, i.e. translation
        is performed before scaling. All conversions and transforms default to `False`.

        `prob` works the same way in all arguments in which it appears. It must be a float in [0,1]
        and determines the probability that the respective transform is applied to a given image.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            train (bool, optional): Whether or not the generator is used in training mode. If `True`, then the labels
                will be transformed into the format that the SSD cost function requires. Otherwise,
                the output format of the labels is identical to the input format.
            ssd_box_encoder (SSDBoxEncoder, optional): Only required if `train = True`. An SSDBoxEncoder object
                to encode the ground truth labels to the required format for training an SSD model.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple with the processed images as its first element and, if in training mode, the encoded
                labels as its second element. Apart from that, the output tuple can contain additional outputs according
                to the keywords in `returns`. The possible keyword strings and their respective outputs are:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. This is an array of shape `(batch_size, n_boxes, n_classes + 12)`
                    that is the output of `SSDBoxEncoder.encode_y()`. Will always be in the outputs if in training mode,
                    so it doesn't matter whether or not you include this keyword in the set if in training mode.
                * 'matched_anchors': The same as 'encoded_labels', but containing anchor box coordinates for all matched
                    anchor boxes instead of ground truth coordinates. The can be useful to visualize what anchor boxes
                    are being matched to each ground truth box. Only available in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'inverse_transform': An array of shape `(batch_size, 4, 2)` that contains two coordinate conversion values for
                    each image in the batch and for each of the four coordinates. These these coordinate conversion values makes
                    it possible to convert the box coordinates that were predicted on a transformed image back to what those coordinates
                    would be in the original image. This is mostly relevant for evaluation: If you want to evaluate your model on
                    a dataset with varying image sizes, then you are forced to transform the images somehow (by resizing or cropping)
                    to make them all the same size. Your model will then predict boxes for those transformed images, but for the
                    evaluation you will need the box coordinates to be correct for the original images, not for the transformed
                    images. This means you will have to transform the predicted box coordinates back to the original image sizes.
                    Since the images have varying sizes, the function that transforms the coordinates is different for every image.
                    This array contains the necessary conversion values for every coordinate of every image in the batch.
                    In order to convert coordinates to the original image sizes, first multiply each coordinate by the second
                    conversion value, then add the first conversion value to it. Note that the conversion will only be correct
                    for the `resize`, `random_crop`, `max_crop_and_resize` and `random_pad_and_resize` transformations.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output will simply be skipped and not be part of the yielded tuple.
            equalize (bool, optional): If `True`, performs histogram equalization on the images.
                This can improve contrast and lead the improved model performance.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
                The respective box coordinates are adjusted accordingly.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for the horizontal shift of the image,
                the second tuple for the vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
                The respective box coordinates are adjusted accordingly.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both min and max must be >=0.
            max_crop_and_resize (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`.
                This will crop out the maximal possible image patch with an aspect ratio defined by `height` and `width` from the
                input image and then resize the resulting patch to `(height, width)`. This preserves the aspect ratio of the original
                image, but does not contain the entire original image (unless the aspect ratio of the original image is the same as
                the target aspect ratio) The latter two components of the tuple work identically as in `random_crop`.
                Note the difference to `random_crop`: This operation crops patches of variable size and fixed aspect ratio from the
                input image and then resizes the patch, while `random_crop` crops patches of fixed size and fixed aspect ratio from
                the input image. If this operation is active, it overrides both `random_crop` and `resize`.
            random_pad_and_resize (tuple, optional): `False` or a tuple of four integers and one float,
                `(height, width, min_1_object, max_#_trials, mix_ratio)`. The input image will first be padded with zeros such that
                it has the aspect ratio defined by `height` and `width` and afterwards resized to `(height, width)`. This preserves
                the aspect ratio of the original image an scales it to the maximum possible size that still fits inside a canvas of
                size `(height, width)`. The third and fourth components of the tuple work identically as in `random_crop`.
                `mix_ratio` is only relevant if `max_crop_and_resize` is active, in which case it must be a float in `[0, 1]` that
                decides what ratio of images will be processed using `max_crop_and_resize` and what ratio of images will be processed
                using `random_pad_and_resize`. If `mix_ratio` is 1, all images will be processed using `random_pad_and_resize`.
                Note the difference to `max_crop_and_resize`: While `max_crop_and_resize` will crop out the largest possible patch
                that still lies fully within the input image, the patch generated here will always contain the full input image.
                If this operation is active, it overrides both `random_crop` and `resize`.
            random_crop (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`,
                where `height` and `width` are the height and width of the patch that is to be cropped out at a random
                position in the input image. Note that `height` and `width` can be arbitrary - they are allowed to be larger
                than the image height and width, in which case the original image will be randomly placed on a black background
                canvas of size `(height, width)`. `min_1_object` is either 0 or 1. If 1, there must be at least one detectable
                object remaining in the image for the crop to be valid, and if 0, crops with no detectable objects left in the
                image patch are allowed. `max_#_trials` is only relevant if `min_1_object == 1` and sets the maximum number
                of attempts to get a valid crop. If no valid crop was obtained within this maximum number of attempts,
                the respective image will be removed from the batch without replacement (i.e. for each removed image, the batch
                will be one sample smaller).
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images.
                The targets are adjusted accordingly. Note: Cropping happens before resizing.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected format is `(height, width)`.
                The box coordinates are adjusted accordingly. Note: Resizing happens after cropping.
            gray (bool, optional): If `True`, converts the images to grayscale. Note that the resulting grayscale
                images have shape `(height, width, 1)`.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries
                post any transformation. This should always be set to `True`, even if you set `include_thresh`
                to 0. I don't even know why I made this an option. If this is set to `False`, you could
                end up with some boxes that lie entirely outside the image boundaries after a given transformation
                and such boxes would of course not make any sense and have a strongly adverse effect on the learning.
            include_thresh (float, optional): Only relevant if `limit_boxes` is `True`. Determines the minimum
                fraction of the area of a ground truth box that must be left after limiting in order for the box
                to still be included in the batch data. If set to 0, all boxes are kept except those which lie
                entirely outside of the image bounderies after limiting. If set to 1, only boxes that did not
                need to be limited at all are kept.
            subtract_mean (array-like, optional): `None` or an array-like object of integers or floating point values
                of any shape that is broadcast-compatible with the image shape. The elements of this array will be
                subtracted from the image pixel intensity values. For example, pass a list of three integers
                to perform per-channel mean normalization for color images.
            divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers or
                floating point values of any shape that is broadcast-compatible with the image shape. The image pixel
                intensity values will be divided by the elements of this array. For example, pass a list
                of three integers to perform per-channel standard deviation normalization for color images.
            swap_channels (bool, optional): If `True` the color channel order of the input images will be reversed,
                i.e. if the input color channel order is RGB, the color channels will be swapped to BGR.
            keep_images_without_gt (bool, optional): If `True`, images for which there are no ground truth boxes
                (either because there weren't any to begin with or because random cropping cropped out a patch that
                doesn't contain any objects) will be kept in the batch. If `False`, such images will be removed
                from the batch.
            convert_to_3_channels (bool, optional): If `True`, single-channel images will be converted
                to 3-channel images.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument. By default, this will be
            a 2-tuple containing the processed batch images as its first element and the encoded ground truth boxes
            tensor as its second element if in training mode, or a 1-tuple containing only the processed batch images if
            not in training mode. Any additional outputs must be specified in the `returns` argument.
        '''

        if shuffle: # Shuffle the data before we begin
            if (self.labels is None) and (self.image_ids is None):
                self.filenames = sklearn.utils.shuffle(self.filenames)
            elif (self.labels is None):
                self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
            elif (self.image_ids is None):
                self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
            else:
                self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)
        current = 0

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        ymin = self.box_output_format.index('ymin')
        xmax = self.box_output_format.index('xmax')
        ymax = self.box_output_format.index('ymax')
        ios = np.amin([xmin, ymin, xmax, ymax]) # Index offset, we need this for the inverse coordinate transform indices.

        while True:

            batch_X, batch_y = [], []

            if current >= len(self.filenames):
                current = 0
                if shuffle:
                    # Shuffle the data after each complete pass
                    if (self.labels is None) and (self.image_ids is None):
                        self.filenames = sklearn.utils.shuffle(self.filenames)
                    elif (self.labels is None):
                        self.filenames, self.image_ids = sklearn.utils.shuffle(self.filenames, self.image_ids)
                    elif (self.image_ids is None):
                        self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
                    else:
                        self.filenames, self.labels, self.image_ids = sklearn.utils.shuffle(self.filenames, self.labels, self.image_ids)

            # Get the image filepaths for this batch.
            batch_filenames = self.filenames[current:current+batch_size]

            # Load the images for this batch.
            for filename in batch_filenames:
                with Image.open(filename) as img:
                    batch_X.append(np.array(img))

            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            # Get the image IDs for this batch (if there are any).
            if not self.image_ids is None:
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            # Create the array that is to contain the inverse coordinate transformation values for this batch.
            batch_inverse_coord_transform = np.array([[[0, 1]] * 4] * batch_size, dtype=np.float) # Array of shape `(batch_size, 4, 2)`, where the last axis contains an additive and a multiplicative scalar transformation constant.

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns and not batch_y is None:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            batch_items_to_remove = [] # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list.

            for i in range(len(batch_X)):

                img_height, img_width = batch_X[i].shape[0], batch_X[i].shape[1]

                if not batch_y is None:
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (len(batch_y[i]) == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                    # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks.
                    batch_y[i] = np.array(batch_y[i])

                # From here on, perform some optional image transformations.

                if (batch_X[i].ndim == 2) and convert_to_3_channels:
                    # Convert the 1-channel image into a 3-channel image.
                    batch_X[i] = np.stack([batch_X[i]] * 3, axis=-1)

                if equalize:
                    batch_X[i] = histogram_eq(batch_X[i])

                if brightness:
                    p = np.random.uniform(0,1)
                    if p >= (1-brightness[2]):
                        batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])

                if flip: # Performs flips along the vertical axis only (i.e. horizontal flips).
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        batch_X[i] = _flip(batch_X[i])
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            batch_y[i][:,[xmin,xmax]] = img_width - batch_y[i][:,[xmax,xmin]] # xmin and xmax are swapped when mirrored

                if translate:
                    p = np.random.uniform(0,1)
                    if p >= (1-translate[2]):
                        # Translate the image and return the shift values so that we can adjust the labels
                        batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            # Adjust the box coordinates.
                            batch_y[i][:,[xmin,xmax]] += xshift
                            batch_y[i][:,[ymin,ymax]] += yshift
                            # Limit the box coordinates to lie within the image boundaries
                            if limit_boxes:
                                before_limiting = deepcopy(batch_y[i])
                                x_coords = batch_y[i][:,[xmin,xmax]]
                                x_coords[x_coords >= img_width] = img_width - 1
                                x_coords[x_coords < 0] = 0
                                batch_y[i][:,[xmin,xmax]] = x_coords
                                y_coords = batch_y[i][:,[ymin,ymax]]
                                y_coords[y_coords >= img_height] = img_height - 1
                                y_coords[y_coords < 0] = 0
                                batch_y[i][:,[ymin,ymax]] = y_coords
                                # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                # process that they don't serve as useful training examples anymore, because too little of them is
                                # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                # `include_thresh` of the box area before limiting.
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if scale:
                    p = np.random.uniform(0,1)
                    if p >= (1-scale[2]):
                        # Rescale the image and return the transformation matrix M so we can use it to adjust the box coordinates
                        batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            # Adjust the box coordinates.
                            # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                            toplefts = np.array([batch_y[i][:,xmin], batch_y[i][:,ymin], np.ones(batch_y[i].shape[0])])
                            bottomrights = np.array([batch_y[i][:,xmax], batch_y[i][:,ymax], np.ones(batch_y[i].shape[0])])
                            new_toplefts = (np.dot(M, toplefts)).T
                            new_bottomrights = (np.dot(M, bottomrights)).T
                            batch_y[i][:,[xmin,ymin]] = new_toplefts.astype(np.int)
                            batch_y[i][:,[xmax,ymax]] = new_bottomrights.astype(np.int)
                            # Limit the box coordinates to lie within the image boundaries
                            if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                                before_limiting = deepcopy(batch_y[i])
                                x_coords = batch_y[i][:,[xmin,xmax]]
                                x_coords[x_coords >= img_width] = img_width - 1
                                x_coords[x_coords < 0] = 0
                                batch_y[i][:,[xmin,xmax]] = x_coords
                                y_coords = batch_y[i][:,[ymin,ymax]]
                                y_coords[y_coords >= img_height] = img_height - 1
                                y_coords[y_coords < 0] = 0
                                batch_y[i][:,[ymin,ymax]] = y_coords
                                # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                                # process that they don't serve as useful training examples anymore, because too little of them is
                                # visible. We'll remove all boxes that we had to limit so much that their area is less than
                                # `include_thresh` of the box area before limiting.
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if max_crop_and_resize:
                    # The ratio of the two aspect ratios (source image and target size) determines the maximal possible crop.
                    image_aspect_ratio = img_width / img_height
                    resize_aspect_ratio = max_crop_and_resize[1] / max_crop_and_resize[0]

                    if image_aspect_ratio < resize_aspect_ratio:
                        crop_width = img_width
                        crop_height = int(round(crop_width / resize_aspect_ratio))
                    else:
                        crop_height = img_height
                        crop_width = int(round(crop_height * resize_aspect_ratio))
                    # The actual cropping and resizing will be done by the random crop and resizing operations below.
                    # Here, we only set the parameters for them.
                    random_crop = (crop_height, crop_width, max_crop_and_resize[2], max_crop_and_resize[3])
                    resize = (max_crop_and_resize[0], max_crop_and_resize[1])

                if random_pad_and_resize:

                    resize_aspect_ratio = random_pad_and_resize[1] / random_pad_and_resize[0]

                    if img_width < img_height:
                        crop_height = img_height
                        crop_width = int(round(crop_height * resize_aspect_ratio))
                    else:
                        crop_width = img_width
                        crop_height = int(round(crop_width / resize_aspect_ratio))
                    # The actual cropping and resizing will be done by the random crop and resizing operations below.
                    # Here, we only set the parameters for them.
                    if max_crop_and_resize:
                        p = np.random.uniform(0,1)
                        if p >= (1-random_pad_and_resize[4]):
                            random_crop = (crop_height, crop_width, random_pad_and_resize[2], random_pad_and_resize[3])
                            resize = (random_pad_and_resize[0], random_pad_and_resize[1])
                    else:
                        random_crop = (crop_height, crop_width, random_pad_and_resize[2], random_pad_and_resize[3])
                        resize = (random_pad_and_resize[0], random_pad_and_resize[1])

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
                        if y_range >= 0: crop_ymin = np.random.randint(0, y_range + 1) # There are y_range + 1 possible positions for the crop in the vertical dimension
                        else: crop_ymin = np.random.randint(0, -y_range + 1) # The possible positions for the image on the background canvas in the vertical dimension
                        if x_range >= 0: crop_xmin = np.random.randint(0, x_range + 1) # There are x_range + 1 possible positions for the crop in the horizontal dimension
                        else: crop_xmin = np.random.randint(0, -x_range + 1) # The possible positions for the image on the background canvas in the horizontal dimension
                        # Perform the crop
                        if y_range >= 0 and x_range >= 0: # If the patch to be cropped out is smaller than the original image in both dimenstions, we just perform a regular crop
                            # Crop the image
                            patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0], crop_xmin:crop_xmin+random_crop[1]])
                            # Add the parameters to reverse this transformation.
                            patch_y_inverse_y = crop_ymin
                            patch_y_inverse_x = crop_xmin
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Translate the box coordinates into the new coordinate system: Cropping shifts the origin by `(crop_ymin, crop_xmin)`
                                patch_y = np.copy(batch_y[i])
                                patch_y[:,[ymin,ymax]] -= crop_ymin
                                patch_y[:,[xmin,xmax]] -= crop_xmin
                                # Limit the box coordinates to lie within the new image boundaries
                                if limit_boxes:
                                    # Both the x- and y-coordinates might need to be limited
                                    before_limiting = np.copy(patch_y)
                                    y_coords = patch_y[:,[ymin,ymax]]
                                    y_coords[y_coords < 0] = 0
                                    y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                    patch_y[:,[ymin,ymax]] = y_coords
                                    x_coords = patch_y[:,[xmin,xmax]]
                                    x_coords[x_coords < 0] = 0
                                    x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                    patch_y[:,[xmin,xmax]] = x_coords
                        elif y_range >= 0 and x_range < 0: # If the crop is larger than the original image in the horizontal dimension only,...
                            # Crop the image
                            patch_X = np.copy(batch_X[i][crop_ymin:crop_ymin+random_crop[0]]) # ...crop the vertical dimension just as before,...
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[:, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `crop_xmin` position computed above.
                            patch_X = canvas
                            # Add the parameters to reverse this transformation.
                            patch_y_inverse_y = crop_ymin
                            patch_y_inverse_x = -crop_xmin
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(crop_ymin, -crop_xmin)`
                                patch_y = np.copy(batch_y[i])
                                patch_y[:,[ymin,ymax]] -= crop_ymin
                                patch_y[:,[xmin,xmax]] += crop_xmin
                                # Limit the box coordinates to lie within the new image boundaries
                                if limit_boxes:
                                    # Only the y-coordinates might need to be limited
                                    before_limiting = np.copy(patch_y)
                                    y_coords = patch_y[:,[ymin,ymax]]
                                    y_coords[y_coords < 0] = 0
                                    y_coords[y_coords >= random_crop[0]] = random_crop[0] - 1
                                    patch_y[:,[ymin,ymax]] = y_coords
                        elif y_range < 0 and x_range >= 0: # If the crop is larger than the original image in the vertical dimension only,...
                            # Crop the image
                            patch_X = np.copy(batch_X[i][:,crop_xmin:crop_xmin+random_crop[1]]) # ...crop the horizontal dimension just as in the first case,...
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin+img_height, :] = patch_X # ...and place the patch onto the canvas at the random `crop_ymin` position computed above.
                            patch_X = canvas
                            # Add the parameters to reverse this transformation.
                            patch_y_inverse_y = -crop_ymin
                            patch_y_inverse_x = crop_xmin
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, crop_xmin)`
                                patch_y = np.copy(batch_y[i])
                                patch_y[:,[ymin,ymax]] += crop_ymin
                                patch_y[:,[xmin,xmax]] -= crop_xmin
                                # Limit the box coordinates to lie within the new image boundaries
                                if limit_boxes:
                                    # Only the x-coordinates might need to be limited
                                    before_limiting = np.copy(patch_y)
                                    x_coords = patch_y[:,[xmin,xmax]]
                                    x_coords[x_coords < 0] = 0
                                    x_coords[x_coords >= random_crop[1]] = random_crop[1] - 1
                                    patch_y[:,[xmin,xmax]] = x_coords
                        else:  # If the crop is larger than the original image in both dimensions,...
                            patch_X = np.copy(batch_X[i])
                            canvas = np.zeros((random_crop[0], random_crop[1], patch_X.shape[2]), dtype=np.uint8) # ...generate a blank background image to place the patch onto,...
                            canvas[crop_ymin:crop_ymin+img_height, crop_xmin:crop_xmin+img_width] = patch_X # ...and place the patch onto the canvas at the random `(crop_ymin, crop_xmin)` position computed above.
                            patch_X = canvas
                            # Add the parameters to reverse this transformation.
                            patch_y_inverse_y = -crop_ymin
                            patch_y_inverse_x = -crop_xmin
                            if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                                # Translate the box coordinates into the new coordinate system: In this case, the origin is shifted by `(-crop_ymin, -crop_xmin)`
                                patch_y = np.copy(batch_y[i])
                                patch_y[:,[ymin,ymax]] += crop_ymin
                                patch_y[:,[xmin,xmax]] += crop_xmin
                                # Note that no limiting is necessary in this case
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            if limit_boxes and (y_range >= 0 or x_range >= 0):
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (patch_y[:,xmax] - patch_y[:,xmin]) * (patch_y[:,ymax] - patch_y[:,ymin])
                                if include_thresh == 0: patch_y = patch_y[after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                                else: patch_y = patch_y[after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all
                            trial_counter += 1 # We've just used one of our trials
                            # Check if we have found a valid crop
                            if random_crop[2] == 0: # If `min_1_object == 0`, break out of the while loop after the first loop because we are fine with whatever crop we got
                                batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                                batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                                break
                            elif len(patch_y) > 0: # If we have at least one object left, this crop is valid and we can stop
                                min_1_object_fulfilled = True
                                batch_X[i] = patch_X # The cropped patch becomes our new batch item
                                batch_y[i] = patch_y # The adjusted boxes become our new labels for this batch item
                                batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                                batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                            elif (trial_counter >= random_crop[3]) and (not i in batch_items_to_remove): # If we've reached the trial limit and still not found a valid crop, remove this image from the batch
                                batch_items_to_remove.append(i)
                        else: # If `batch_y` is `None`, i.e. if we don't have ground truth data, any crop is a valid crop.
                            batch_X[i] = patch_X # The cropped patch becomes our new batch item
                            batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],0] += patch_y_inverse_y
                            batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],0] += patch_y_inverse_x
                            break
                    # Update the image size so that subsequent transformations can work correctly.
                    img_height = random_crop[0]
                    img_width = random_crop[1]

                if crop:
                    # Crop the image
                    batch_X[i] = np.copy(batch_X[i][crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])
                    # Update the image size so that subsequent transformations can work correctly
                    img_height -= crop[0] + crop[1]
                    img_width -= crop[2] + crop[3]
                    if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                        # Translate the box coordinates into the new coordinate system if necessary: The origin is shifted by `(crop[0], crop[2])` (i.e. by the top and left crop values)
                        # If nothing was cropped off from the top or left of the image, the coordinate system stays the same as before
                        if crop[0] > 0:
                            batch_y[i][:,[ymin,ymax]] -= crop[0]
                        if crop[2] > 0:
                            batch_y[i][:,[xmin,xmax]] -= crop[2]
                        # Limit the box coordinates to lie within the new image boundaries
                        if limit_boxes:
                            before_limiting = np.copy(batch_y[i])
                            # We only need to check those box coordinates that could possibly have been affected by the cropping
                            # For example, if we only crop off the top and/or bottom of the image, there is no need to check the x-coordinates
                            if crop[0] > 0:
                                y_coords = batch_y[i][:,[ymin,ymax]]
                                y_coords[y_coords < 0] = 0
                                batch_y[i][:,[ymin,ymax]] = y_coords
                            if crop[1] > 0:
                                y_coords = batch_y[i][:,[ymin,ymax]]
                                y_coords[y_coords >= img_height] = img_height - 1
                                batch_y[i][:,[ymin,ymax]] = y_coords
                            if crop[2] > 0:
                                x_coords = batch_y[i][:,[xmin,xmax]]
                                x_coords[x_coords < 0] = 0
                                batch_y[i][:,[xmin,xmax]] = x_coords
                            if crop[3] > 0:
                                x_coords = batch_y[i][:,[xmin,xmax]]
                                x_coords[x_coords >= img_width] = img_width - 1
                                batch_y[i][:,[xmin,xmax]] = x_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * (before_limiting[:,ymax] - before_limiting[:,ymin])
                            after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                            if include_thresh == 0: batch_y[i] = batch_y[i][after_area > include_thresh * before_area] # If `include_thresh == 0`, we want to make sure that boxes with area 0 get thrown out, hence the ">" sign instead of the ">=" sign
                            else: batch_y[i] = batch_y[i][after_area >= include_thresh * before_area] # Especially for the case `include_thresh == 1` we want the ">=" sign, otherwise no boxes would be left at all

                if resize:
                    batch_X[i] = cv2.resize(batch_X[i], dsize=(resize[1], resize[0]))
                    batch_inverse_coord_transform[i,[ymin-ios,ymax-ios],1] *= (img_height / resize[0])
                    batch_inverse_coord_transform[i,[xmin-ios,xmax-ios],1] *= (img_width / resize[1])
                    if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                        batch_y[i][:,[ymin,ymax]] = batch_y[i][:,[ymin,ymax]] * (resize[0] / img_height)
                        batch_y[i][:,[xmin,xmax]] = batch_y[i][:,[xmin,xmax]] * (resize[1] / img_width)
                    img_width, img_height = resize # Updating these at this point is unnecessary, but it's one fewer source of error if this method gets expanded in the future.

                if gray:
                    batch_X[i] = cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY)
                    if convert_to_3_channels:
                        batch_X[i] = np.stack([batch_X[i]] * 3, axis=-1)
                    else:
                        batch_X[i] = np.expand_dims(batch_X[i], axis=-1)

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes.
            #          At this point, all images must have the same size, otherwise you will get an error during training.
            batch_X = np.array(batch_X)

            if not keep_images_without_gt:
                # If any batch items need to be removed because of failed random cropping, remove them now.
                batch_inverse_coord_transform = np.delete(batch_inverse_coord_transform, batch_items_to_remove, axis=0)
                batch_X = np.delete(batch_X, batch_items_to_remove, axis=0)
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully should not need to be done often anyway.
                    batch_filenames.pop(j)
                    if not batch_y is None: batch_y.pop(j)
                    if not batch_imgage_ids is None: batch_image_ids.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not batch_y is None: batch_original_labels.pop(j)

            # Perform image transformations that can be bulk-applied to the whole batch.
            if not (subtract_mean is None):
                batch_X = batch_X.astype(np.int16) - np.array(subtract_mean)
            if not (divide_by_stddev is None):
                batch_X = batch_X.astype(np.int16) / np.array(divide_by_stddev)
            if swap_channels:
                batch_X = batch_X[:,:,:,[2, 1, 0]]

            if train: # During training we need the encoded labels instead of the format that `batch_y` has
                if ssd_box_encoder is None:
                    raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                if 'matched_anchors' in returns:
                    batch_y_true, batch_matched_anchors = ssd_box_encoder.encode_y(batch_y, diagnostics=True) # Encode the labels into the `y_true` tensor that the SSD loss function needs.
                else:
                    batch_y_true = ssd_box_encoder.encode_y(batch_y, diagnostics=False) # Encode the labels into the `y_true` tensor that the SSD loss function needs.

            # Compile the output.
            ret = []
            ret.append(batch_X)
            if train:
                ret.append(batch_y_true)
                if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns and not batch_y is None: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns and not batch_image_ids is None: ret.append(batch_image_ids)
            if 'inverse_transform' in returns: ret.append(batch_inverse_coord_transform)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns and not batch_y is None: ret.append(batch_original_labels)

            yield ret

    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames, the list of labels, and the list of image IDs.
        '''
        return self.filenames, self.labels, self.image_ids

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
                        crop=False,
                        equalize=False,
                        brightness=False,
                        flip=False,
                        translate=False,
                        scale=False,
                        resize=False,
                        gray=False,
                        limit_boxes=True,
                        include_thresh=0.3,
                        diagnostics=False):
        '''
        Perform offline image processing.

        This function has mostly the same image processing capabilities as the generator function above,
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
            with Image.open('{}'.format(os.path.join(self.images_path, filename))) as img:
                image = np.array(img)
            targets = np.copy(self.labels[i])

            if diagnostics:
                original_images.append(image)

            img_height, img_width, ch = image.shape

            if equalize:
                image = histogram_eq(image)

            if brightness:
                p = np.random.uniform(0,1)
                if p >= (1-brightness[2]):
                    image = _brightness(image, min=brightness[0], max=brightness[1])

            # Could easily be extended to also allow vertical flipping, but I'm not convinced of the
            # usefulness of vertical flipping either empirically or theoretically, so I'm going for simplicity.
            # If you want to allow vertical flipping, just change this function to pass the respective argument
            # to `_flip()`.
            if flip:
                p = np.random.uniform(0,1)
                if p >= (1-flip):
                    image = _flip(image)
                    targets[:,[0,1]] = img_width - targets[:,[1,0]] # xmin and xmax are swapped when mirrored

            if translate:
                p = np.random.uniform(0,1)
                if p >= (1-translate[2]):
                    image, xshift, yshift = _translate(image, translate[0], translate[1])
                    targets[:,[0,1]] += xshift
                    targets[:,[2,3]] += yshift
                    if limit_boxes:
                        before_limiting = np.copy(targets)
                        x_coords = targets[:,[0,1]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:,[0,1]] = x_coords
                        y_coords = targets[:,[2,3]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:,[2,3]] = y_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                        after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                        targets = targets[after_area >= include_thresh * before_area]

            if scale:
                p = np.random.uniform(0,1)
                if p >= (1-scale[2]):
                    image, M, scale_factor = _scale(image, scale[0], scale[1])
                    # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                    toplefts = np.array([targets[:,0], targets[:,2], np.ones(targets.shape[0])])
                    bottomrights = np.array([targets[:,1], targets[:,3], np.ones(targets.shape[0])])
                    new_toplefts = (np.dot(M, toplefts)).T
                    new_bottomrights = (np.dot(M, bottomrights)).T
                    targets[:,[0,2]] = new_toplefts.astype(np.int)
                    targets[:,[1,3]] = new_bottomrights.astype(np.int)
                    if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                        before_limiting = np.copy(targets)
                        x_coords = targets[:,[0,1]]
                        x_coords[x_coords >= img_width] = img_width - 1
                        x_coords[x_coords < 0] = 0
                        targets[:,[0,1]] = x_coords
                        y_coords = targets[:,[2,3]]
                        y_coords[y_coords >= img_height] = img_height - 1
                        y_coords[y_coords < 0] = 0
                        targets[:,[2,3]] = y_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                        after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                        targets = targets[after_area >= include_thresh * before_area]

            if crop:
                image = image[crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]]
                if limit_boxes: # Adjust boxes affected by cropping and remove those that will no longer be in the image
                    before_limiting = np.copy(targets)
                    if crop[0] > 0:
                        y_coords = targets[:,[2,3]]
                        y_coords[y_coords < crop[0]] = crop[0]
                        targets[:,[2,3]] = y_coords
                    if crop[1] > 0:
                        y_coords = targets[:,[2,3]]
                        y_coords[y_coords >= (img_height - crop[1])] = img_height - crop[1] - 1
                        targets[:,[2,3]] = y_coords
                    if crop[2] > 0:
                        x_coords = targets[:,[0,1]]
                        x_coords[x_coords < crop[2]] = crop[2]
                        targets[:,[0,1]] = x_coords
                    if crop[3] > 0:
                        x_coords = targets[:,[0,1]]
                        x_coords[x_coords >= (img_width - crop[3])] = img_width - crop[3] - 1
                        targets[:,[0,1]] = x_coords
                    # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                    # process that they don't serve as useful training examples anymore, because too little of them is
                    # visible. We'll remove all boxes that we had to limit so much that their area is less than
                    # `include_thresh` of the box area before limiting.
                    before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                    after_area = (targets[:,1] - targets[:,0]) * (targets[:,3] - targets[:,2])
                    targets = targets[after_area >= include_thresh * before_area]
                # Now adjust the box coordinates for the new image size post cropping
                if crop[0] > 0:
                    targets[:,[2,3]] -= crop[0]
                if crop[2] > 0:
                    targets[:,[0,1]] -= crop[2]
                img_height -= crop[0] - crop[1]
                img_width -= crop[2] - crop[3]

            if resize:
                image = cv2.resize(image, dsize=resize)
                targets[:,[0,1]] = (targets[:,[0,1]] * (resize[0] / img_width)).astype(np.int)
                targets[:,[2,3]] = (targets[:,[2,3]] * (resize[1] / img_height)).astype(np.int)

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
