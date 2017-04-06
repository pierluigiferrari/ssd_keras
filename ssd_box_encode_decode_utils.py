'''
Includes:
* Function to compute IoU similarity
* Function to decode SSD model output
* Class to encode targets for SSD model training
'''

import numpy as np

def iou(boxes1, boxes2):
    '''
    Compute the intersection-over-union similarity (also known as Jaccard similarity)
    of two rectangular boxes or of multiple rectangular boxes contained in two arrays with
    broadcast-compatible shapes.

    Three common use cases would be to compute the similarities for 1 vs. 1, 1 vs. `n`,
    or `n` vs. `n` boxes. The two arguments are symmetric.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format `[xmin, xmax, ymin, ymax]` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format `[xmin, xmax, ymin, ymax]` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes1`.

    Returns:
        A 1D Numpy array of dtype float containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and `boxes2`.
        0 means there is no overlap between two given boxes, 1 means their coordinates are identical.
    '''

    if len(boxes1.shape) > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(len(boxes1.shape)))
    if len(boxes2.shape) > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(len(boxes2.shape)))

    if len(boxes1.shape) == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if len(boxes2.shape) == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4): raise ValueError("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.".format(boxes1.shape[1], boxes2.shape[1]))

    intersection = np.maximum(0, np.minimum(boxes1[:,1], boxes2[:,1]) - np.maximum(boxes1[:,0], boxes2[:,0])) * np.maximum(0, np.minimum(boxes1[:,3], boxes2[:,3]) - np.maximum(boxes1[:,2], boxes2[:,2]))
    union = (boxes1[:,1] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,2]) + (boxes2[:,1] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,2]) - intersection

    return intersection / union

def decode_y(y_pred):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last dimension contains
            `[one-hot vector for the classes, xmin, xmax, ymin, ymax]`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 5)` where each row is a box prediction for
        a non-background class for the respective image in the format `[xmin, xmax, ymin, ymax, class_id]`.
    '''
    # First, decode the classes from 1-hot-encoding to their class ID
    y_pred_classes_decoded = (np.copy(y_pred[:,:,-9:-4])).astype(np.int32)
    y_pred_classes_decoded[:,:,0] = np.argmax(y_pred[:,:,:-8], axis=-1)

    y_pred_decoded = []
    for batch_item in y_pred_classes_decoded: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class...
        boxes = np.roll(boxes, -1, axis=-1) # ...and change the order from [class_id, xmin, xmax, ymin, ymax] to [xmin, xmax, ymin, ymax, class_id], like in the ground truth data
        y_pred_decoded.append(boxes)

    return y_pred_decoded

class SSDBoxEncoder:
    '''
    A class to transform ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD model, and to transform predictions of the SSD model back
    to the original format of the input labels.

    In the process of encoding ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 classifier_sizes,
                 min_scale=0.1,
                 max_scale=0.8,
                 scales=None,
                 aspect_ratios=[0.5, 1, 2],
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 iou_threshold=0.5):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of classes including the background class.
            classifier_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional classifier layers.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Defaults to 0.1.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Defaults to 0.8.
            scales (list, optional): A list containing one scaling factor per convolutional classifier layer.
                Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
                `max_scale`. All scaling factors must be in [0,1].
            aspect_ratios (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. Defaults to [0.5, 1, 2].
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
                Defaults to `True`.
            iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box. Defaults to 0.5.
        '''

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if len(scales) != len(classifier_sizes):
                raise ValueError("It must be either scales is None or len(scales) == len(classifier_sizes), but len(scales) == {} and len(classifier_sizes) == {}".format(len(scales), len(classifier_sizes)))

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.classifier_sizes = classifier_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.iou_threshold = iou_threshold

    def generate_anchor_boxes(self,
                              batch_size,
                              feature_map_size,
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

        # Compute the number of boxes per cell
        if (1 in self.aspect_ratios) & self.two_boxes_for_ar1:
            n_boxes = len(self.aspect_ratios) + 1
        else:
            n_boxes = len(self.aspect_ratios)

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        self.aspect_ratios = np.sort(self.aspect_ratios)
        size = min(self.img_height, self.img_width)
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular anchor box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w,h))
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w,h))
        wh_list = np.array(wh_list, dtype=np.int32)

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = (np.linspace(cell_width/2, self.img_width-cell_width/2, feature_map_size[1])).astype(np.int32)
        cy = (np.linspace(cell_height/2, self.img_height-cell_height/2, feature_map_size[0])).astype(np.int32)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1).astype(np.int32) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1).astype(np.int32) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4), dtype=np.int32)

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Now we convert `(cx, cy, w, h)` into `(xmin, xmax, ymin, ymax)` in order to be able
        # to limit the boxes to lie entirely within the image boundaries
        temp = np.copy(boxes_tensor)
        temp[:, :, :, 0] = boxes_tensor[:, :, :, 0] - (boxes_tensor[:, :, :, 2] / 2).astype(np.int32) # Set xmin
        temp[:, :, :, 1] = boxes_tensor[:, :, :, 0] + (boxes_tensor[:, :, :, 2] / 2).astype(np.int32) # Set xmax
        temp[:, :, :, 2] = boxes_tensor[:, :, :, 1] - (boxes_tensor[:, :, :, 3] / 2).astype(np.int32) # Set ymin
        temp[:, :, :, 3] = boxes_tensor[:, :, :, 1] + (boxes_tensor[:, :, :, 3] / 2).astype(np.int32) # Set ymax
        boxes_tensor = temp

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
                here is similar, just for all classifier conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 8)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 8` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes.
        '''

        # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
        #    If `scales` is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`
        if not self.scales:
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.classifier_sizes))

        # 2: For each conv classifier layer (i.e. for each scale factor) get the tensors for
        #    the box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = [] # List to hold the box widths and heights
            cell_sizes = [] # List to hold horizontal and vertical distances between any two boxes
            for i in range(len(self.scales)-1):
                boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                              feature_map_size=self.classifier_sizes[i],
                                                              this_scale=self.scales[i],
                                                              next_scale=self.scales[i+1],
                                                              diagnostics=True)
                boxes_tensor.append(boxes)
                wh_list.append(wh)
                cell_sizes.append(cells)
            # For the last scale value, set `next_scale = 1`
            boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                          feature_map_size=self.classifier_sizes[-1],
                                                          this_scale=self.scales[-1],
                                                          next_scale=1.0,
                                                          diagnostics=True)
            boxes_tensor.append(boxes)
            wh_list.append(wh)
            cell_sizes.append(cells)
        else:
            for i in range(len(self.scales)-1):
                boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                               feature_map_size=self.classifier_sizes[i],
                                                               this_scale=self.scales[i],
                                                               next_scale=self.scales[i+1],
                                                               diagnostics=False))
            # For the the last scale value, set `next_scale = 1`
            boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                           feature_map_size=self.classifier_sizes[-1],
                                                           this_scale=self.scales[-1],
                                                           next_scale=1.0,
                                                           diagnostics=False))

        boxes_tensor = np.concatenate(boxes_tensor, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes), dtype=np.int32)

        # 4: Concatenate the classes and boxes tensors to get our final template for y_encoded. We also need
        #    to append a dummy tensor of the shape of `boxes_tensor` so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this dummy tensor is irrelevant, it won't be
        #    used, so we'll just use `boxes_tensor` a second time.
        y_encode_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor), axis=2)

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
        background class.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(xmin, xmax, ymin, ymax, class_id)`, and `class_id` must be an integer greater than 0 for all boxes
                as class_id 0 is reserved for the background class.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(batch_size=len(ground_truth_labels),
                                                          diagnostics=False)

        y_encoded = np.copy(y_encode_template) # We'll write the ground truth box data to this array

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

        class_vector = np.eye(self.n_classes, dtype=np.int32) # An identity matrix that we'll use as one-hot class vectors

        for i in range(y_encode_template.shape[0]):
            available_boxes = np.ones((y_encode_template.shape[1]), dtype=np.int32) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
            for true_box in ground_truth_labels[i]:
                similarities = iou(y_encode_template[i,:,-8:-4], true_box[:4]) # The iou similarities for all anchor boxes
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met < self.iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:-4] = np.concatenate((class_vector[true_box[4]], true_box[:4].astype(np.int32)), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:-4] = np.concatenate((class_vector[true_box[4]], true_box[:4].astype(np.int32)), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(available_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1

        return y_encoded
