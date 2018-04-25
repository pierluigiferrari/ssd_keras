'''
A data generator for 2D object detection.

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
import inspect
from collections import defaultdict
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
import sys
from tqdm import tqdm
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

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    Currently provides three methods to parse annotation data: A general-purpose CSV parser,
    an XML parser for the Pascal VOC datasets, and a JSON parser for the MS COCO datasets.
    If the annotations of your dataset are in a format that is not supported by these parsers,
    you could just add another parser method and still use this generator.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 filenames=None,
                 filenames_type='text',
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 eval_neutral=None):
        '''
        This class provides parser methods that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the dataset from CSV or XML files. If you already
        have the image filenames and labels in asuitable format (see argument descriptions below), you can pass
        them right here in the constructor, in which case you do not need to call any of the parser methods afterwards.

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            labels_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, ymin, xmax, ymax in the generated ground truth data (if any). The expected
                strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
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
            eval_neutral (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain for each image
                a list that indicates for each ground truth object in the image whether that object is supposed
                to be treated as neutral during an evaluation.
        '''
        self.labels_output_format = labels_output_format
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} # This dictionary is for internal use.

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

        if not eval_neutral is None:
            if isinstance(eval_neutral, str):
                with open(eval_neutral, 'rb') as f:
                    self.eval_neutral = pickle.load(f)
            elif isinstance(eval_neutral, (list, tuple)):
                self.eval_neutral = eval_neutral
            else:
                raise ValueError("`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.eval_neutral = None

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
            None by default, optionally the image filenames, labels, and image IDs.
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
        self.image_ids = []
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
                    for element in self.labels_output_format: # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                        box.append(int(row[self.input_format.index(element)].strip())) # ...select the respective column in the input format and append it to `box`.
                    data.append(box)

        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = data[0][0] # The current image for which we're collecting the ground truth boxes
        current_image_id = data[0][0].split('.')[0] # The image ID will be the portion of the image name before the first dot.
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
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
            else: # If this box belongs to a new image file
                if random_sample: # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0,1)
                    if p >= (1-random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)
                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)
                current_labels = [] # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data)-1: # If this is the last line of the CSV file
                    if random_sample: # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0,1)
                        if p >= (1-random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        if ret: # In case we want to return these
            return self.filenames, self.labels, self.image_ids

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
            ret (bool, optional): Whether or not to return the outputs.

        Returns:
            None by default, optionally the image filenames, labels, image IDs, and a list indicating which boxes are
            annotated with the label "difficult".
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
        self.eval_neutral = []
        if not annotations_dirs:
            self.labels = None
            self.eval_neutral = None
            annotations_dirs = [None] * len(images_dirs)

        for images_dir, image_set_filename, annotations_dir in zip(images_dirs, image_set_filenames, annotations_dirs):
            # Read the image set file that so that we know all the IDs of all the images to be included in the dataset.
            with open(image_set_filename) as f:
                image_ids = [line.strip() for line in f] # Note: These are strings, not integers.
                self.image_ids += image_ids

            # Loop over all images in this dataset.
            for image_id in tqdm(image_ids, desc=os.path.basename(image_set_filename), file=sys.stdout):

                filename = '{}'.format(image_id) + '.jpg'
                self.filenames.append(os.path.join(images_dir, filename))

                if not annotations_dir is None:
                    # Parse the XML file for this image.
                    with open(os.path.join(annotations_dir, image_id + '.xml')) as f:
                        soup = BeautifulSoup(f, 'xml')

                    folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which dataset an image belongs to.
                    #filename = soup.filename.text

                    boxes = [] # We'll store all boxes for this image here.
                    eval_neutr = [] # We'll store whether a box is annotated as "difficult" here.
                    objects = soup.find_all('object') # Get a list of all objects in this image.

                    # Parse the data for each object.
                    for obj in objects:
                        class_name = obj.find('name', recursive=False).text
                        class_id = self.classes.index(class_name)
                        # Check whether this class is supposed to be included in the dataset.
                        if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                        pose = obj.find('pose', recursive=False).text
                        truncated = int(obj.find('truncated', recursive=False).text)
                        if exclude_truncated and (truncated == 1): continue
                        difficult = int(obj.find('difficult', recursive=False).text)
                        if exclude_difficult and (difficult == 1): continue
                        # Get the bounding box coordinates.
                        bndbox = obj.find('bndbox', recursive=False)
                        xmin = int(bndbox.xmin.text)
                        ymin = int(bndbox.ymin.text)
                        xmax = int(bndbox.xmax.text)
                        ymax = int(bndbox.ymax.text)
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
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                        if difficult: eval_neutr.append(True)
                        else: eval_neutr.append(False)

                    self.labels.append(boxes)
                    self.eval_neutral.append(eval_neutr)

        if ret:
            return self.filenames, self.labels, self.image_ids, self.eval_neutral

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
                        for item in self.labels_output_format:
                            box.append(item_dict[item])
                        boxes.append(box)
                    self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels, self.image_ids

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple that contains the outputs specified in this set and only those. If an output is not available,
                it will be `None`. The output tuple can contain the following outputs according to the specified keyword strings:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'matched_anchors': Only available if `labels_encoder` is an `SSDInputEncoder` object. The same as 'encoded_labels',
                    but containing anchor box coordinates for all matched anchor boxes instead of ground truth coordinates.
                    This can be useful to visualize what anchor boxes are being matched to each ground truth box. Only available
                    in training mode.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'evaluation-neutral': A nested list of lists of booleans. Each list contains `True` or `False` for every ground truth
                    bounding box of the respective image depending on whether that bounding box is supposed to be evaluation-neutral (`True`)
                    or not (`False`). May return `None` if there exists no such concept for a given dataset. An example for
                    evaluation-neutrality are the ground truth boxes annotated as "difficult" in the Pascal VOC datasets, which are
                    usually treated to be neutral in a model evaluation.
                * 'inverse_transform': A nested list that contains a list of "inverter" functions for each item in the batch.
                    These inverter functions take (predicted) labels for an image as input and apply the inverse of the transformations
                    that were applied to the original image to them. This makes it possible to let the model make predictions on a
                    transformed image and then convert these predictions back to the original image. This is mostly relevant for
                    evaluation: If you want to evaluate your model on a dataset with varying image sizes, then you are forced to
                    transform the images somehow (e.g. by resizing or cropping) to make them all the same size. Your model will then
                    predict boxes for those transformed images, but for the evaluation you will need predictions with respect to the
                    original images, not with respect to the transformed images. This means you will have to transform the predicted
                    box coordinates back to the original image sizes. Note that for each image, the inverter functions for that
                    image need to be applied in the order in which they are given in the respective list for that image.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `xmax <= xmin` and/or
                `ymax <= ymin`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument.
        '''

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels', 'matched_anchors', 'evaluation-neutral']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels', 'evaluation-neutral', 'encoded_labels', and 'matched_anchors' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels', 'matched_anchors']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' and 'matched_anchors' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))
        elif not isinstance(label_encoder, SSDInputEncoder):
            if 'matched_anchors' in returns:
                warnings.warn("`label_encoder` is not an `SSDInputEncoder` object, therefore 'matched_anchors' is not a possible return, " +
                              "but you set `returns = {}`. The impossible returns will be `None`.".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.filenames]
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.eval_neutral is None):
                objects_to_shuffle.append(self.eval_neutral)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= len(self.filenames):
                current = 0

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.filenames]
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.eval_neutral is None):
                        objects_to_shuffle.append(self.eval_neutral)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]

            #########################################################################################
            # Get the images, image filenames, (maybe) image IDs, and (maybe) labels for this batch.
            #########################################################################################

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

            if not (self.eval_neutral is None):
                batch_eval_neutral = self.eval_neutral[current:current+batch_size]
            else:
                batch_eval_neutral = None

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []

            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:

                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):
                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None
            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################

            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret

    def save_dataset(self,
                     filenames_path='filenames.pkl',
                     labels_path=None,
                     image_ids_path=None):
        '''
        Writes the current `filenames`, `labels`, and `image_ids` lists to the specified files.
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

    def get_dataset(self):
        '''
        Returns:
            The list of filenames, the list of labels, and the list of image IDs.
        '''
        return self.filenames, self.labels, self.image_ids

    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return len(self.filenames)
