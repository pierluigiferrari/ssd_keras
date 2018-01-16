'''
A few utilities that are useful when working with the MS COCO datasets.

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

import json
from tqdm import trange
from math import ceil
import sys

from ssd_box_encode_decode_utils import decode_y

def get_coco_category_maps(annotations_file):
    '''
    Builds dictionaries that map between MS COCO category IDs, transformed category IDs, and category names.
    The original MS COCO category IDs are not consecutive unfortunately: The 80 category IDs are spread
    across the integers 1 through 90 with some integers skipped. Since we usually use a one-hot
    class representation in neural networks, we need to map these non-consecutive original COCO category
    IDs (let's call them 'cats') to consecutive category IDs (let's call them 'classes').

    Arguments:
        annotations_file (str): The filepath to any MS COCO annotations JSON file.

    Returns:
        1) cats_to_classes: A dictionary that maps between the original (keys) and the transformed category IDs (values).
        2) classes_to_cats: A dictionary that maps between the transformed (keys) and the original category IDs (values).
        3) cats_to_names: A dictionary that maps between original category IDs (keys) and the respective category names (values).
        4) classes_to_names: A list of the category names (values) with their indices representing the transformed IDs.
    '''
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    cats_to_classes = {}
    classes_to_cats = {}
    cats_to_names = {}
    classes_to_names = []
    classes_to_names.append('background') # Need to add the background class first so that the indexing is right.
    for i, cat in enumerate(annotations['categories']):
        cats_to_classes[cat['id']] = i + 1
        classes_to_cats[i + 1] = cat['id']
        cats_to_names[cat['id']] = cat['name']
        classes_to_names.append(cat['name'])

    return cats_to_classes, classes_to_cats, cats_to_names, classes_to_names

def predict_all_to_json(out_file,
                        model,
                        img_height,
                        img_width,
                        classes_to_cats,
                        batch_generator,
                        batch_size,
                        batch_generator_mode='resize',
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        pred_coords='centroids',
                        normalize_coords=True):
    '''
    Runs detection predictions over the whole dataset given a model and saves them in a JSON file
    in the MS COCO detection results format.

    Arguments:
        out_file (str): The file name (full path) under which to save the results JSON file.
        model (Keras model): A Keras SSD model object.
        img_height (int): The input image height for the model.
        img_width (int): The input image width for the model.
        classes_to_cats (dict): A dictionary that maps the consecutive class IDs predicted by the model
            to the non-consecutive original MS COCO category IDs.
        batch_generator (BatchGenerator): A `BatchGenerator` object with the evaluation dataset.
        batch_size (int): The batch size for the evaluation.
        batch_generator_mode (str, optional): Either of 'resize' or 'pad'. If 'resize', the input images will
            be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
            If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
            and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage. Defaults to 200, following the paper.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.

    Returns:
        None.
    '''

    if batch_generator_mode == 'resize':
        random_pad_and_resize=False
        resize=(img_height,img_width)
    elif batch_generator_mode == 'pad':
        random_pad_and_resize=(img_height, img_width, 0, 3, 1.0)
        resize=False
    else:
        raise ValueError("Unexpected argument value: `batch_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(batch_generator_mode))

    # Set the generator parameters.
    generator = batch_generator.generate(batch_size=batch_size,
                                         shuffle=False,
                                         train=False,
                                         returns={'processed_images', 'image_ids', 'inverse_transform'},
                                         convert_to_3_channels=True,
                                         random_pad_and_resize=random_pad_and_resize,
                                         resize=resize,
                                         limit_boxes=False,
                                         keep_images_without_gt=True)
    # Put the results in this list.
    results = []
    # Compute the number of batches to iterate over the entire dataset.
    n_images = batch_generator.get_n_samples()
    print("Number of images in the evaluation dataset: {}".format(n_images))
    n_batches = int(ceil(n_images / batch_size))
    # Loop over all batches.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description('Producing results file')
    for i in tr:
        # Generate batch.
        batch_X, batch_image_ids, batch_inverse_coord_transform = next(generator)
        # Predict.
        y_pred = model.predict(batch_X)
        # Decode.
        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=confidence_thresh,
                                  iou_threshold=iou_threshold,
                                  top_k=top_k,
                                  input_coords=pred_coords,
                                  normalize_coords=normalize_coords,
                                  img_height=img_height,
                                  img_width=img_width)
        # Convert each predicted box into the results format.
        for k, batch_item in enumerate(y_pred_decoded):
            # The box coordinates were predicted for the transformed
            # (resized, cropped, padded, etc.) image. We now have to
            # transform these coordinates back to what they would be
            # in the original images.
            batch_item[:,2:] *= batch_inverse_coord_transform[k,:,1]
            batch_item[:,2:] += batch_inverse_coord_transform[k,:,0]
            for box in batch_item:
                class_id = box[0]
                # Transform the consecutive class IDs back to the original COCO category IDs.
                cat_id = classes_to_cats[class_id]
                # Round the box coordinates to reduce the JSON file size.
                xmin = round(box[2], 1)
                ymin = round(box[3], 1)
                xmax = round(box[4], 1)
                ymax = round(box[5], 1)
                width = xmax - xmin
                height = ymax - ymin
                bbox = [xmin, ymin, width, height]
                result = {}
                result['image_id'] = batch_image_ids[k]
                result['category_id'] = cat_id
                result['score'] = round(box[1], 3)
                result['bbox'] = bbox
                results.append(result)

    with open(out_file, 'w') as f:
        json.dump(results, f)

    print("Prediction results saved in '{}'".format(out_file))
