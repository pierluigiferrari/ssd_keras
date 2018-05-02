'''
A few utilities that are useful when working with the MS COCO datasets.

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

import json
from tqdm import trange
from math import ceil
import sys

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

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
                        data_generator,
                        batch_size,
                        data_generator_mode='resize',
                        model_mode='training',
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
        data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
        batch_size (int): The batch size for the evaluation.
        data_generator_mode (str, optional): Either of 'resize' or 'pad'. If 'resize', the input images will
            be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
            If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
            and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
        model_mode (str, optional): The mode in which the model was created, i.e. 'training', 'inference' or 'inference_fast'.
            This is needed in order to know whether the model output is already decoded or still needs to be decoded. Refer to
            the model documentation for the meaning of the individual modes.
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

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height,width=img_width)
    if data_generator_mode == 'resize':
        transformations = [convert_to_3_channels,
                           resize]
    elif data_generator_mode == 'pad':
        random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width/img_height, clip_boxes=False)
        transformations = [convert_to_3_channels,
                           random_pad,
                           resize]
    else:
        raise ValueError("Unexpected argument value: `data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(data_generator_mode))

    # Set the generator parameters.
    generator = data_generator.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=transformations,
                                        label_encoder=None,
                                        returns={'processed_images',
                                                 'image_ids',
                                                 'inverse_transform'},
                                        keep_images_without_gt=True)
    # Put the results in this list.
    results = []
    # Compute the number of batches to iterate over the entire dataset.
    n_images = data_generator.get_dataset_size()
    print("Number of images in the evaluation dataset: {}".format(n_images))
    n_batches = int(ceil(n_images / batch_size))
    # Loop over all batches.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description('Producing results file')
    for i in tr:
        # Generate batch.
        batch_X, batch_image_ids, batch_inverse_transforms = next(generator)
        # Predict.
        y_pred = model.predict(batch_X)
        # If the model was created in 'training' mode, the raw predictions need to
        # be decoded and filtered, otherwise that's already taken care of.
        if model_mode == 'training':
            # Decode.
            y_pred = decode_detections(y_pred,
                                       confidence_thresh=confidence_thresh,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       input_coords=pred_coords,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
        else:
            # Filter out the all-zeros dummy elements of `y_pred`.
            y_pred_filtered = []
            for i in range(len(y_pred)):
                y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
            y_pred = y_pred_filtered
        # Convert the predicted box coordinates for the original images.
        y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

        # Convert each predicted box into the results format.
        for k, batch_item in enumerate(y_pred):
            for box in batch_item:
                class_id = box[0]
                # Transform the consecutive class IDs back to the original COCO category IDs.
                cat_id = classes_to_cats[class_id]
                # Round the box coordinates to reduce the JSON file size.
                xmin = float(round(box[2], 1))
                ymin = float(round(box[3], 1))
                xmax = float(round(box[4], 1))
                ymax = float(round(box[5], 1))
                width = xmax - xmin
                height = ymax - ymin
                bbox = [xmin, ymin, width, height]
                result = {}
                result['image_id'] = batch_image_ids[k]
                result['category_id'] = cat_id
                result['score'] = float(round(box[1], 3))
                result['bbox'] = bbox
                results.append(result)

    with open(out_file, 'w') as f:
        json.dump(results, f)

    print("Prediction results saved in '{}'".format(out_file))
