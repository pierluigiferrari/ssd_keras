'''
An evaluator to compute the Pascal VOC-style mean average precision (both the pre-2010
and post-2010 algorithm versions) of a given Keras SSD model on a given dataset.

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
from math import ceil
from tqdm import trange
import sys
import warnings

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from bounding_box_utils.bounding_box_utils import iou

class Evaluator:
    '''
    Computes the mean average precision of the given Keras SSD model on the given dataset.

    Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
    and post-2010 (integration) algorithm versions.

    Optionally also returns the average precisions, precisions, and recalls.

    The algorithm is identical to the official Pascal VOC pre-2010 detection evaluation algorithm
    in its default settings, but can be cusomized in a number of ways.
    '''

    def __init__(self,
                 model,
                 n_classes,
                 data_generator,
                 model_mode='inference',
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5},
                 gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            model (Keras model): A Keras SSD model object.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
            model_mode (str, optional): The mode in which the model was created, i.e. 'training', 'inference' or 'inference_fast'.
                This is needed in order to know whether the model output is already decoded or still needs to be decoded. Refer to
                the model documentation for the meaning of the individual modes.
            pred_format (dict, optional): A dictionary that defines which index in the last axis of the model's decoded predictions
                contains which bounding box coordinate. The dictionary must map the keywords 'class_id', 'conf' (for the confidence),
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis.
            gt_format (list, optional): A dictionary that defines which index of a ground truth bounding box contains which of the five
                items class ID, xmin, ymin, xmax, ymax. The expected strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
        '''

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")

        self.model = model
        self.data_generator = data_generator
        self.n_classes = n_classes
        self.model_mode = model_mode
        self.pred_format = pred_format
        self.gt_format = gt_format

        # The following lists all contain per-class data, i.e. all list have the length `n_classes + 1`,
        # where one element is for the background class, i.e. that element is just a dummy entry.
        self.prediction_results = None
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        self.cumulative_precisions = None # "Cumulative" means that the i-th element in each list represents the precision for the first i highest condidence predictions for that class.
        self.cumulative_recalls = None # "Cumulative" means that the i-th element in each list represents the recall for the first i highest condidence predictions for that class.
        self.average_precisions = None
        self.mean_average_precision = None

    def __call__(self,
                 img_height,
                 img_width,
                 batch_size,
                 data_generator_mode='resize',
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='sample',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 return_precisions=False,
                 return_recalls=False,
                 return_average_precisions=False,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):
        '''
        Computes the mean average precision of the given Keras SSD model on the given dataset.

        Optionally also returns the averages precisions, precisions, and recalls.

        All the individual steps of the overall evaluation algorithm can also be called separately
        (check out the other methods of this class), but this runs the overall algorithm all at once.

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            data_generator_mode (str, optional): Either of 'resize' and 'pad'. If 'resize', the input images will
                be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
                If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
                and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
            round_confidences (int, optional): `False` or an integer that is the number of decimals that the prediction
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            average_precision_mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision
                will be computed according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled
                for `num_recall_points` recall values. In the case of 'integrate', the average precision will be computed according to the
                Pascal VOC formula that was used from VOC 2010 onward, where the average precision will be computed by numerically integrating
                over the whole preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just
                the limit case of 'sample' mode as the number of sample points increases.
            num_recall_points (int, optional): The number of points to sample from the precision-recall-curve to compute the average
                precisions. In other words, this is the number of equidistant recall values for which the resulting precision will be
                computed. 11 points is the value used in the official Pascal VOC 2007 detection evaluation algorithm.
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            return_precisions (bool, optional): If `True`, returns a nested list containing the cumulative precisions for each class.
            return_recalls (bool, optional): If `True`, returns a nested list containing the cumulative recalls for each class.
            return_average_precisions (bool, optional): If `True`, returns a list containing the average precision for each class.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            decoding_confidence_thresh (float, optional): Only relevant if the model is in 'training' mode.
                A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered
                for the non-maximum suppression stage for the respective class. A lower value will result in a larger part of the
                selection process being done by the non-maximum suppression stage, while a larger value will result in a larger
                part of the selection process happening in the confidence thresholding stage.
            decoding_iou_threshold (float, optional): Only relevant if the model is in 'training' mode. A float in [0,1].
                All boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
                from the set of predictions for a given class, where 'maximal' refers to the box score.
            decoding_top_k (int, optional): Only relevant if the model is in 'training' mode. The number of highest scoring
                predictions to be kept for each batch item after the non-maximum suppression stage.
            decoding_input_coords (str, optional): Only relevant if the model is in 'training' mode. The box coordinate format
                that the model outputs. Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            decoding_normalize_coords (bool, optional): Only relevant if the model is in 'training' mode. Set to `True` if the model
                outputs relative coordinates. Do not set this to `True` if the model already outputs absolute coordinates,
                as that would result in incorrect coordinates.

        Returns:
            A float, the mean average precision, plus any optional returns specified in the arguments.
        '''

        #############################################################################################
        # Predict on the entire dataset.
        #############################################################################################

        self.predict_on_dataset(img_height=img_height,
                                img_width=img_width,
                                batch_size=batch_size,
                                data_generator_mode=data_generator_mode,
                                decoding_confidence_thresh=decoding_confidence_thresh,
                                decoding_iou_threshold=decoding_iou_threshold,
                                decoding_top_k=decoding_top_k,
                                decoding_pred_coords=decoding_pred_coords,
                                decoding_normalize_coords=decoding_normalize_coords,
                                decoding_border_pixels=border_pixels,
                                round_confidences=round_confidences,
                                verbose=verbose,
                                ret=False)

        #############################################################################################
        # Get the total number of ground truth boxes for each class.
        #############################################################################################

        self.get_num_gt_per_class(ignore_neutral_boxes=ignore_neutral_boxes,
                                  verbose=False,
                                  ret=False)

        #############################################################################################
        # Match predictions to ground truth boxes for all classes.
        #############################################################################################

        self.match_predictions(ignore_neutral_boxes=ignore_neutral_boxes,
                               matching_iou_threshold=matching_iou_threshold,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm,
                               verbose=verbose,
                               ret=False)

        #############################################################################################
        # Compute the cumulative precision and recall for all classes.
        #############################################################################################

        self.compute_precision_recall(verbose=verbose, ret=False)

        #############################################################################################
        # Compute the average precision for this class.
        #############################################################################################

        self.compute_average_precisions(mode=average_precision_mode,
                                        num_recall_points=num_recall_points,
                                        verbose=verbose,
                                        ret=False)

        #############################################################################################
        # Compute the mean average precision.
        #############################################################################################

        mean_average_precision = self.compute_mean_average_precision(ret=True)

        #############################################################################################

        # Compile the returns.
        if return_precisions or return_recalls or return_average_precisions:
            ret = [mean_average_precision]
            if return_average_precisions:
                ret.append(self.average_precisions)
            if return_precisions:
                ret.append(self.cumulative_precisions)
            if return_recalls:
                ret.append(self.cumulative_recalls)
            return ret
        else:
            return mean_average_precision

    def predict_on_dataset(self,
                           img_height,
                           img_width,
                           batch_size,
                           data_generator_mode='resize',
                           decoding_confidence_thresh=0.01,
                           decoding_iou_threshold=0.45,
                           decoding_top_k=200,
                           decoding_pred_coords='centroids',
                           decoding_normalize_coords=True,
                           decoding_border_pixels='include',
                           round_confidences=False,
                           verbose=True,
                           ret=False):
        '''
        Runs predictions for the given model over the entire dataset given by `data_generator`.

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            data_generator_mode (str, optional): Either of 'resize' and 'pad'. If 'resize', the input images will
                be resized (i.e. warped) to `(img_height, img_width)`. This mode does not preserve the aspect ratios of the images.
                If 'pad', the input images will be first padded so that they have the aspect ratio defined by `img_height`
                and `img_width` and then resized to `(img_height, img_width)`. This mode preserves the aspect ratios of the images.
            decoding_confidence_thresh (float, optional): Only relevant if the model is in 'training' mode.
                A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered
                for the non-maximum suppression stage for the respective class. A lower value will result in a larger part of the
                selection process being done by the non-maximum suppression stage, while a larger value will result in a larger
                part of the selection process happening in the confidence thresholding stage.
            decoding_iou_threshold (float, optional): Only relevant if the model is in 'training' mode. A float in [0,1].
                All boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
                from the set of predictions for a given class, where 'maximal' refers to the box score.
            decoding_top_k (int, optional): Only relevant if the model is in 'training' mode. The number of highest scoring
                predictions to be kept for each batch item after the non-maximum suppression stage.
            decoding_input_coords (str, optional): Only relevant if the model is in 'training' mode. The box coordinate format
                that the model outputs. Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            decoding_normalize_coords (bool, optional): Only relevant if the model is in 'training' mode. Set to `True` if the model
                outputs relative coordinates. Do not set this to `True` if the model already outputs absolute coordinates,
                as that would result in incorrect coordinates.
            round_confidences (int, optional): `False` or an integer that is the number of decimals that the prediction
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the predictions.

        Returns:
            None by default. Optionally, a nested list containing the predictions for each class.
        '''

        class_id_pred = self.pred_format['class_id']
        conf_pred     = self.pred_format['conf']
        xmin_pred     = self.pred_format['xmin']
        ymin_pred     = self.pred_format['ymin']
        xmax_pred     = self.pred_format['xmax']
        ymax_pred     = self.pred_format['ymax']

        #############################################################################################
        # Configure the data generator for the evaluation.
        #############################################################################################

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height,width=img_width, labels_format=self.gt_format)
        if data_generator_mode == 'resize':
            transformations = [convert_to_3_channels,
                               resize]
        elif data_generator_mode == 'pad':
            random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width/img_height, labels_format=self.gt_format)
            transformations = [convert_to_3_channels,
                               random_pad,
                               resize]
        else:
            raise ValueError("`data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(data_generator_mode))

        # Set the generator parameters.
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 transformations=transformations,
                                                 label_encoder=None,
                                                 returns={'processed_images',
                                                          'image_ids',
                                                          'evaluation-neutral',
                                                          'inverse_transform',
                                                          'original_labels'},
                                                 keep_images_without_gt=True,
                                                 degenerate_box_handling='remove')

        # If we don't have any real image IDs, generate pseudo-image IDs.
        # This is just to make the evaluator compatible both with datasets that do and don't
        # have image IDs.
        if self.data_generator.image_ids is None:
            self.data_generator.image_ids = list(range(self.data_generator.get_dataset_size()))

        #############################################################################################
        # Predict over all batches of the dataset and store the predictions.
        #############################################################################################

        # We have to generate a separate results list for each class.
        results = [list() for _ in range(self.n_classes + 1)]

        # Create a dictionary that maps image IDs to ground truth annotations.
        # We'll need it below.
        image_ids_to_labels = {}

        # Compute the number of batches to iterate over the entire dataset.
        n_images = self.data_generator.get_dataset_size()
        n_batches = int(ceil(n_images / batch_size))
        if verbose:
            print("Number of images in the evaluation dataset: {}".format(n_images))
            print()
            tr = trange(n_batches, file=sys.stdout)
            tr.set_description('Producing predictions batch-wise')
        else:
            tr = range(n_batches)

        # Loop over all batches.
        for j in tr:
            # Generate batch.
            batch_X, batch_image_ids, batch_eval_neutral, batch_inverse_transforms, batch_orig_labels = next(generator)
            # Predict.
            y_pred = self.model.predict(batch_X)
            # If the model was created in 'training' mode, the raw predictions need to
            # be decoded and filtered, otherwise that's already taken care of.
            if self.model_mode == 'training':
                # Decode.
                y_pred = decode_detections(y_pred,
                                           confidence_thresh=decoding_confidence_thresh,
                                           iou_threshold=decoding_iou_threshold,
                                           top_k=decoding_top_k,
                                           input_coords=decoding_pred_coords,
                                           normalize_coords=decoding_normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           border_pixels=decoding_border_pixels)
            else:
                # Filter out the all-zeros dummy elements of `y_pred`.
                y_pred_filtered = []
                for i in range(len(y_pred)):
                    y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
                y_pred = y_pred_filtered
            # Convert the predicted box coordinates for the original images.
            y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

            # Iterate over all batch items.
            for k, batch_item in enumerate(y_pred):

                image_id = batch_image_ids[k]

                for box in batch_item:
                    class_id = int(box[class_id_pred])
                    # Round the box coordinates to reduce the required memory.
                    if round_confidences:
                        confidence = round(box[conf_pred], round_confidences)
                    else:
                        confidence = box[conf_pred]
                    xmin = round(box[xmin_pred], 1)
                    ymin = round(box[ymin_pred], 1)
                    xmax = round(box[xmax_pred], 1)
                    ymax = round(box[ymax_pred], 1)
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    # Append the predicted box to the results list for its class.
                    results[class_id].append(prediction)

        self.prediction_results = results

        if ret:
            return results

    def write_predictions_to_txt(self,
                                 classes=None,
                                 out_file_prefix='comp3_det_test_',
                                 verbose=True):
        '''
        Writes the predictions for all classes to separate text files according to the Pascal VOC results format.

        Arguments:
            classes (list, optional): `None` or a list of strings containing the class names of all classes in the dataset,
                including some arbitrary name for the background class. This list will be used to name the output text files.
                The ordering of the names in the list represents the ordering of the classes as they are predicted by the model,
                i.e. the element with index 3 in this list should correspond to the class with class ID 3 in the model's predictions.
                If `None`, the output text files will be named by their class IDs.
            out_file_prefix (str, optional): A prefix for the output text file names. The suffix to each output text file name will
                be the respective class name followed by the `.txt` file extension. This string is also how you specify the directory
                in which the results are to be saved.
            verbose (bool, optional): If `True`, will print out the progress during runtime.

        Returns:
            None.
        '''

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        # We generate a separate results file for each class.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Writing results file for class {}/{}.".format(class_id, self.n_classes))

            if classes is None:
                class_suffix = '{:04d}'.format(class_id)
            else:
                class_suffix = classes[class_id]

            results_file = open('{}{}.txt'.format(out_file_prefix, class_suffix), 'w')

            for prediction in self.prediction_results[class_id]:

                prediction_list = list(prediction)
                prediction_list[0] = '{:06d}'.format(int(prediction_list[0]))
                prediction_list[1] = round(prediction_list[1], 4)
                prediction_txt = ' '.join(map(str, prediction_list)) + '\n'
                results_file.write(prediction_txt)

            results_file.close()

        if verbose:
            print("All results files saved.")

    def get_num_gt_per_class(self,
                             ignore_neutral_boxes=True,
                             verbose=True,
                             ret=False):
        '''
        Counts the number of ground truth boxes for each class across the dataset.

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `True`, only non-neutral ground truth boxes will be counted, otherwise all ground truth boxes will
                be counted.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the list of counts.

        Returns:
            None by default. Optionally, a list containing a count of the number of ground truth boxes for each class across the
            entire dataset.
        '''

        if self.data_generator.labels is None:
            raise ValueError("Computing the number of ground truth boxes per class not possible, no ground truth given.")

        num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)

        class_id_index = self.gt_format['class_id']

        ground_truth = self.data_generator.labels

        if verbose:
            print('Computing the number of positive ground truth boxes per class.')
            tr = trange(len(ground_truth), file=sys.stdout)
        else:
            tr = range(len(ground_truth))

        # Iterate over the ground truth for all images in the dataset.
        for i in tr:

            boxes = np.asarray(ground_truth[i])

            # Iterate over all ground truth boxes for the current image.
            for j in range(boxes.shape[0]):

                if ignore_neutral_boxes and not (self.data_generator.eval_neutral is None):
                    if not self.data_generator.eval_neutral[i][j]:
                        # If this box is not supposed to be evaluation-neutral,
                        # increment the counter for the respective class ID.
                        class_id = boxes[j, class_id_index]
                        num_gt_per_class[class_id] += 1
                else:
                    # If there is no such thing as evaluation-neutral boxes for
                    # our dataset, always increment the counter for the respective
                    # class ID.
                    class_id = boxes[j, class_id_index]
                    num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class

        if ret:
            return num_gt_per_class

    def match_predictions(self,
                          ignore_neutral_boxes=True,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True,
                          ret=False):
        '''
        Matches predictions to ground truth boxes.

        Note that `predict_on_dataset()` must be called before calling this method.

        Arguments:
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the true and false positives.

        Returns:
            None by default. Optionally, four nested lists containing the true positives, false positives, cumulative true positives,
            and cumulative false positives for each class.
        '''

        if self.data_generator.labels is None:
            raise ValueError("Matching predictions to ground truth boxes not possible, no ground truth given.")

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']

        # Convert the ground truth to a more efficient format for what we need
        # to do, which is access ground truth by image ID repeatedly.
        ground_truth = {}
        eval_neutral_available = not (self.data_generator.eval_neutral is None) # Whether or not we have annotations to decide whether ground truth boxes should be neutral or not.
        for i in range(len(self.data_generator.image_ids)):
            image_id = str(self.data_generator.image_ids[i])
            labels = self.data_generator.labels[i]
            if ignore_neutral_boxes and eval_neutral_available:
                ground_truth[image_id] = (np.asarray(labels), np.asarray(self.data_generator.eval_neutral[i]))
            else:
                ground_truth[image_id] = np.asarray(labels)

        true_positives = [[]] # The false positives for each class, sorted by descending confidence.
        false_positives = [[]] # The true positives for each class, sorted by descending confidence.
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            predictions = self.prediction_results[class_id]

            # Store the matching results in these lists:
            true_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a true positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a false positive, 0 otherwise

            # In case there are no predictions at all for this class, we're done here.
            if len(predictions) == 0:
                print("No predictions for class {}/{}".format(class_id, self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                continue

            # Convert the predictions list for this class into a structured array so that we can sort it by confidence.

            # Get the number of characters needed to store the image ID strings in the structured array.
            num_chars_per_image_id = len(str(predictions[0][0])) + 6 # Keep a few characters buffer in case some image IDs are longer than others.
            # Create the data type for the structured array.
            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            # Create the structured array
            predictions = np.array(predictions, dtype=preds_data_type)

            # Sort the detections by decreasing confidence.
            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)
            predictions_sorted = predictions[descending_indices]

            if verbose:
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description("Matching predictions to ground truth, class {}/{}.".format(class_id, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            # Keep track of which ground truth boxes were already matched to a detection.
            gt_matched = {}

            # Iterate over all predictions.
            for i in tr:

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']])) # Convert the structured array element to a regular array.

                # Get the relevant ground truth boxes for this prediction,
                # i.e. all ground truth boxes that match the prediction's
                # image ID and class ID.

                # The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
                # or only `ground_truth_boxes`.
                if ignore_neutral_boxes and eval_neutral_available:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)
                class_mask = gt[:,class_id_gt] == class_id
                gt = gt[class_mask]
                if ignore_neutral_boxes and eval_neutral_available:
                    eval_neutral = eval_neutral[class_mask]

                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue

                # Compute the IoU of this prediction with all ground truth boxes of the same class.
                overlaps = iou(boxes1=gt[:,[xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)

                # For each detection, match the ground truth box with the highest overlap.
                # It's possible that the same ground truth box will be matched to multiple
                # detections.
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    # False positive, IoU threshold violated:
                    # Those predictions whose matched overlap is below the threshold become
                    # false positives.
                    false_pos[i] = 1
                else:
                    if not (ignore_neutral_boxes and eval_neutral_available) or (eval_neutral[gt_match_index] == False):
                        # If this is not a ground truth that is supposed to be evaluation-neutral
                        # (i.e. should be skipped for the evaluation) or if we don't even have the
                        # concept of neutral boxes.
                        if not (image_id in gt_matched):
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                        elif not gt_matched[image_id][gt_match_index]:
                            # True positive:
                            # If the matched ground truth box for this prediction hasn't been matched to a
                            # different prediction already, we have a true positive.
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                        else:
                            # False positive, duplicate detection:
                            # If the matched ground truth box for this prediction has already been matched
                            # to a different prediction previously, it is a duplicate detection for an
                            # already detected object, which counts as a false positive.
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

        if ret:
            return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

    def compute_precision_recall(self, verbose=True, ret=False):
        '''
        Computes the precisions and recalls for all classes.

        Note that `match_predictions()` must be called before calling this method.

        Arguments:
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the precisions and recalls.

        Returns:
            None by default. Optionally, two nested lists containing the cumulative precisions and recalls for each class.
        '''

        if (self.cumulative_true_positives is None) or (self.cumulative_false_positives is None):
            raise ValueError("True and false positives not available. You must run `match_predictions()` before you call this method.")

        if (self.num_gt_per_class is None):
            raise ValueError("Number of ground truth boxes per class not available. You must run `get_num_gt_per_class()` before you call this method.")

        cumulative_precisions = [[]]
        cumulative_recalls = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Computing precisions and recalls, class {}/{}".format(class_id, self.n_classes))

            tp = self.cumulative_true_positives[class_id]
            fp = self.cumulative_false_positives[class_id]


            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0) # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / self.num_gt_per_class[class_id] # 1D array with shape `(num_predictions,)`

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls

        if ret:
            return cumulative_precisions, cumulative_recalls

    def compute_average_precisions(self, mode='sample', num_recall_points=11, verbose=True, ret=False):
        '''
        Computes the average precision for each class.

        Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
        and post-2010 (integration) algorithm versions.

        Note that `compute_precision_recall()` must be called before calling this method.

        Arguments:
            mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision will be computed
                according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled for `num_recall_points`
                recall values. In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula that
                was used from VOC 2010 onward, where the average precision will be computed by numerically integrating over the whole
                preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just the limit case
                of 'sample' mode as the number of sample points increases. For details, see the references below.
            num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of points to sample from the precision-recall-curve
                to compute the average precisions. In other words, this is the number of equidistant recall values for which the resulting
                precision will be computed. 11 points is the value used in the official Pascal VOC pre-2010 detection evaluation algorithm.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the average precisions.

        Returns:
            None by default. Optionally, a list containing average precision for each class.

        References:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
        '''

        if (self.cumulative_precisions is None) or (self.cumulative_recalls is None):
            raise ValueError("Precisions and recalls not available. You must run `compute_precision_recall()` before you call this method.")

        if not (mode in {'sample', 'integrate'}):
            raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'".format(mode))

        average_precisions = [0.0]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Computing average precision, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall, return_index=True, return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                # Iterate over all unique recall values in reverse order. This saves a lot of computation:
                # For each unique recall value `r`, we want to get the maximal precision value obtained
                # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
                # values after a given iteration, then in the next iteration, in order compute the maximal
                # precisions for the last `l > k` recall values, we only need to compute the maximal precision
                # for `l - k` recall values and then take the maximum between that and the previously computed
                # maximum instead of computing the maximum over all `l` values.
                # We skip the very last recall value, since the precision after between the last recall value
                # recall 1.0 is defined to be zero.
                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end   = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous iteration to
                    # avoid unnecessary repeated computation over the same precision values.
                    # The maximal precisions are the heights of the rectangle areas of our integral under
                    # the precision-recall curve.
                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]), maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        self.average_precisions = average_precisions

        if ret:
            return average_precisions

    def compute_mean_average_precision(self, ret=True):
        '''
        Computes the mean average precision over all classes.

        Note that `compute_average_precisions()` must be called before calling this method.

        Arguments:
            ret (bool, optional): If `True`, returns the mean average precision.

        Returns:
            A float, the mean average precision, by default. Optionally, None.
        '''

        if self.average_precisions is None:
            raise ValueError("Average precisions not available. You must run `compute_average_precisions()` before you call this method.")

        mean_average_precision = np.average(self.average_precisions[1:]) # The first element is for the background class, so skip it.
        self.mean_average_precision = mean_average_precision

        if ret:
            return mean_average_precision
