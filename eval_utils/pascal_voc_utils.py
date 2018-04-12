'''
Utilities that are useful when evaluating models on the Pascal VOC datasets.

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

from math import ceil
from tqdm import trange
import sys

from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def predict_all_to_txt(model,
                       img_height,
                       img_width,
                       data_generator,
                       batch_size,
                       data_generator_mode='resize',
                       classes=['background',
                                'aeroplane', 'bicycle', 'bird', 'boat',
                                'bottle', 'bus', 'car', 'cat',
                                'chair', 'cow', 'diningtable', 'dog',
                                'horse', 'motorbike', 'person', 'pottedplant',
                                'sheep', 'sofa', 'train', 'tvmonitor'],
                       out_file_prefix='comp3_det_test_',
                       model_mode='training',
                       confidence_thresh=0.01,
                       iou_threshold=0.45,
                       top_k=200,
                       pred_coords='centroids',
                       normalize_coords=True):
    '''
    Runs detection predictions over the whole dataset given a model and saves them in a text file
    in the Pascal VOC detection results format, i.e. the format in which the Pascal VOC test server
    expects results.

    This will result in `n_classes` text files, where each file contains the predictions for one class.

    Arguments:
        model (Keras model): A Keras SSD model object.
        img_height (int): The input image height for the model.
        img_width (int): The input image width for the model.
        data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
        batch_size (int): The batch size for the evaluation.
        data_generator_mode (str, optional): Either of 'resize' or 'pad'. If 'resize', the input images will
            be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
            If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
            and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
        classes (list or dict, optional): A list or dictionary maps the consecutive class IDs predicted by the model
            their respective name strings. The list must contain the background class for class ID zero.
        out_file_prefix (str, optional): A prefix for the output text file names. The suffix to each output text file name will
            be the respective class name followed by the `.txt` file extension. This string is also how you specify the directory
            in which the results are to be saved.
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

    # We have to generate a separate results file for each class.
    results = []
    for i in range(1, len(classes)):
        # Create one text file per class and put it in our results list.
        results.append(open('{}{}.txt'.format(out_file_prefix, classes[i]), 'w'))

    # Compute the number of batches to iterate over the entire dataset.
    n_images = data_generator.get_dataset_size()
    print("Number of images in the evaluation dataset: {}".format(n_images))
    n_batches = int(ceil(n_images / batch_size))
    # Loop over all batches.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description('Producing results files')
    for j in tr:
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
                image_id = batch_image_ids[k]
                class_id = int(box[0])
                # Round the box coordinates to reduce the file size.
                confidence = str(round(box[1], 4))
                xmin = str(round(box[2], 1))
                ymin = str(round(box[3], 1))
                xmax = str(round(box[4], 1))
                ymax = str(round(box[5], 1))
                prediction = [image_id, confidence, xmin, ymin, xmax, ymax]
                prediction_txt = ' '.join(prediction) + '\n'
                # Write the predicted box to the text file for its class.
                results[class_id - 1].write(prediction_txt)

    # Close all the files.
    for results_file in results:
        results_file.close()

    print("All results files saved.")
