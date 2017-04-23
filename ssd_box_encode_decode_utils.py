'''
Includes:
* Function to compute IoU similarity
* Function to decode SSD model output
* Class to encode targets for SSD model training
'''

import numpy as np

def iou(boxes1, boxes2, coords='centroids'):
    '''
    Compute the intersection-over-union similarity (also known as Jaccard similarity)
    of two 2D rectangular boxes or of multiple 2D rectangular boxes contained in two
    arrays with broadcast-compatible shapes.

    Three common use cases would be to compute the similarities for 1 vs. 1, 1 vs. `n`,
    or `n` vs. `n` boxes. The two arguments are symmetric.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            format specified by `coords` or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            Shape must be broadcast-compatible to `boxes2`.
        boxes2 (array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
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
        tensor (array): A Numpy nD array containing the four consecutive coordinates
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
    tensor1 = np.copy(tensor)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+1]) / 2 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] + tensor[..., ind+3]) / 2 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind] # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+2] # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + tensor[..., ind+2] / 2 # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - tensor[..., ind+3] / 2 # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2 # Set ymax
    else:
        raise ValueError("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")

    return tensor1

def convert_coordinates2(tensor, start_index, conversion='minmax2centroids'):
    '''
    A pure matrix multiplication implementation of `convert_coordinates()`.

    Although elegant, it turns out to be marginally slower on average than
    `convert_coordinates()`. Note that the two matrices below are each other's
    multiplicative inverse.

    For details please refer to the documentation of `convert_coordinates()`.
    '''
    ind = start_index
    tensor1 = np.copy(tensor)
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

def decode_y(y_pred, confidence_thresh=0.9, coords='centroids'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence
            required for a given box to be considered a positive prediction. Defaults to 0.9.
        coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax'
            for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[xmin, xmax, ymin, ymax, class_id, confidence]`.
    '''
    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:,:,-10:-4]) # Slice out the four offset predictions plus one element where we'll write the class IDs in the next step
    y_pred_converted[:,:,0] = np.argmax(y_pred[:,:,:-8], axis=-1) # The indices of the highest confidence values in the one-hot class vectors are the class ID
    y_pred_converted[:,:,1] = np.amax(y_pred[:,:,:-8], axis=-1) # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if coords == 'centroids':
        y_pred_converted[:,:,[-2,-1]] = np.exp(y_pred_converted[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor))) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor))) == h(pred) / h(anchor)
        y_pred_converted[:,:,[-2,-1]] *= y_pred[:,:,[-2,-1]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_converted[:,:,[-4,-3]] *= y_pred[:,:,[-2,-1]] # (delta_cx(pred) / w(anchor)) * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor)) * h(anchor) == delta_cy(pred)
        y_pred_converted[:,:,[-4,-3]] += y_pred[:,:,[-4,-3]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2minmax')
    elif coords == 'minmax':
        y_pred_converted[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-3] - y_pred[:,:,-4], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-1] - y_pred[:,:,-2], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,-4:] += y_pred[:,:,-4:] # delta_pred + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    # 3: Decode our huge `(batch, #boxes, 5)` into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,1] >= confidence_thresh] # ...then filter out those positive boxes for which the prediction confidence is too low...
        # TODO: Change the indexing in step 1 above so that this inefficient roll operation becomes obsolete
        boxes = np.roll(boxes, -2, axis=-1) # ...and then change the order from [class_id, confidence, xmin, xmax, ymin, ymax] to [xmin, xmax, ymin, ymax, class_id, confidence], so that the first 5 columns are ordered like in the ground truth data
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
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0],
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 limit_boxes=True,
                 pos_iou_threshold=0.5,
                 neg_iou_threshold=0.3,
                 coords='centroids'):
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
                largest will be linearly interpolated. Note that the second to last of the linearly interpolated
                scaling factors will actually be the scaling factor for the last classifier layer, while the last
                scaling factor is used for the second box for aspect ratio 1 in the last classifier layer
                if `two_boxes_for_ar1` is `True`. Defaults to 0.9.
            scales (list, optional): A list of floats containing scaling factors per convolutional classifier layer.
                This list must be one element longer than the number of classifier layers. The first `k` elements are the
                scaling factors for the `k` classifier layers, while the last element is used for the second box
                for aspect ratio 1 in the last classifier layer if `two_boxes_for_ar1` is `True`. This additional
                last scaling factor must be passed either way, even if it is not being used.
                Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
                `max_scale`. All scaling factors must be greater than zero.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Defaults to [0.5, 1.0, 2.0].
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Defaults to `None`.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
                If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
                Defaults to `True`.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box. Defaults to 0.5.
            neg_iou_threshold (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            coords (str, optional): The box coordinate format to be used. Can be either 'centroids' for the format
                `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the format
                `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        '''
        classifier_sizes = np.array(classifier_sizes)
        if len(classifier_sizes.shape) == 1:
            classifier_sizes = np.expand_dims(classifier_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != len(classifier_sizes)+1): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(classifier_sizes)+1, but len(scales) == {} and len(classifier_sizes)+1 == {}".format(len(scales), len(classifier_sizes)+1))

        if aspect_ratios_per_layer:
            if (len(aspect_ratios_per_layer) != len(classifier_sizes)): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(classifier_sizes), but len(aspect_ratios_per_layer) == {} and len(classifier_sizes) == {}".format(len(aspect_ratios_per_layer), len(classifier_sizes)))

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError("It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (coords == 'minmax' or coords == 'centroids'):
            raise ValueError("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.classifier_sizes = classifier_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords

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
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.classifier_sizes)+1)

        # 2: For each conv classifier layer (i.e. for each scale factor) get the tensors for
        #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = [] # List to hold the box widths and heights
            cell_sizes = [] # List to hold horizontal and vertical distances between any two boxes
            if self.aspect_ratios_per_layer: # If individual aspect ratios are given per layer, we need to pass them to `generate_anchor_boxes()` accordingly
                for i in range(len(self.classifier_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.classifier_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
            else: # Use the same global aspect ratio list for all layers
                for i in range(len(self.classifier_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.classifier_sizes[i],
                                                                  aspect_ratios=self.aspect_ratios_global,
                                                                  this_scale=self.scales[i],
                                                                  next_scale=self.scales[i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
        else:
            if self.aspect_ratios_per_layer:
                for i in range(len(self.classifier_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.classifier_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_per_layer[i],
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))
            else:
                for i in range(len(self.classifier_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.classifier_sizes[i],
                                                                   aspect_ratios=self.aspect_ratios_global,
                                                                   this_scale=self.scales[i],
                                                                   next_scale=self.scales[i+1],
                                                                   diagnostics=False))

        boxes_tensor = np.concatenate(boxes_tensor, axis=1) # Concatenate the anchor tensors from the individual layers to one

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

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
                if self.coords == 'centroids':
                    true_box = convert_coordinates(true_box, start_index=0, conversion='minmax2centroids')
                if np.any(true_box[2:] == 0): continue # We have to protect ourselves against bad ground truth data: The width and height of a ground truth box cannot be zero, otherwise we'll get an error when we take the natural log below (and of course it doesn't make much sense to have such a box)
                similarities = iou(y_encode_template[i,:,-8:-4], true_box[:4], coords=self.coords) # The iou similarities for all anchor boxes
                negative_boxes[similarities >= self.neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                similarities *= available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                available_and_thresh_met = np.copy(similarities)
                available_and_thresh_met[available_and_thresh_met < self.pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                assign_indices = np.nonzero(available_and_thresh_met)[0] # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                if len(assign_indices) > 0: # If we have any matches
                    y_encoded[i,assign_indices,:-4] = np.concatenate((class_vector[true_box[4]], true_box[:4].astype(np.int32)), axis=0) # Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                    available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                else: # If we don't have any matches
                    best_match_index = np.argmax(similarities) # Get the index of the best iou match out of all available boxes
                    y_encoded[i,best_match_index,:-4] = np.concatenate((class_vector[true_box[4]], true_box[:4].astype(np.int32)), axis=0) # Write the ground truth box coordinates and class to the best match anchor box position
                    available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                    negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
            # Set the classes of all remaining available anchor boxes to class zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i,background_class_indices,0] = 1

        # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
        if self.coords == 'centroids':
            y_encoded[:,:,[-8,-7]] -= y_encode_template[:,:,[-8,-7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:,:,[-8,-7]] /= y_encode_template[:,:,[-6,-5]] # (cx(gt) - cx(anchor)) / w(anchor), (cy(gt) - cy(anchor)) / h(anchor)
            y_encoded[:,:,[-6,-5]] /= y_encode_template[:,:,[-6,-5]] # w(gt) / w(anchor), h(gt) / h(anchor)
            y_encoded[:,:,[-6,-5]] = np.log(y_encoded[:,:,[-6,-5]]) # ln(w(gt) / w(anchor)), ln(h(gt) / h(anchor)) (natural logarithm)
        else:
            y_encoded[:,:,-8:-4] -= y_encode_template[:,:,-8:-4] # (gt - anchor) for all four coordinates
            y_encoded[:,:,[-8,-7]] /= np.expand_dims(y_encode_template[:,:,-7] - y_encode_template[:,:,-8], axis=-1) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:,:,[-6,-5]] /= np.expand_dims(y_encode_template[:,:,-5] - y_encode_template[:,:,-6], axis=-1) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)

        return y_encoded
