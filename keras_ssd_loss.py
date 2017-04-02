'''The Keras-compatible loss function for the SSD model. Currently supports TensorFlow only.'''

import tensorflow as tf

class SSD_Loss:
    '''
    The SSD loss. This implementation has an important difference to the loss
    function used in https://arxiv.org/abs/1512.02325: The paper regresses to
    `(cx, cy, w, h)`, i.e. to the box center x and y coordinates and the box width
    and height, while this implementation regresses to `(xmin, xmax, ymin, ymax)`,
    i.e. to the horizontal and vertical min and max box coordinates. This is relevant
    for the normalization performed in `smooth_L1_loss()`. If it weren't for this
    normalization, the format of the four box coordinates wouldn't matter for this
    loss function as long as it would be consistent between `y_true` and `y_pred`.
    '''

    def __init__(self,
                 loc_norm,
                 neg_pos_ratio=3,
                 alpha=1.0):
        '''
        Arguments:
            loc_norm (array): A Numpy array with shape `(batch_size, #boxes, 4)`,
                where the last dimension contains the default box widths and heights
                in the format `(width, width, height, height)`. This is used for
                normalization in `smooth_L1_loss`.
            neg_pos_ratio (int): The maximum number of negative (i.e. background)
                ground truth boxes to include in the loss computation. There are no
                actual background ground truth boxes of course, but `y_true`
                contains default boxes labeled with the background class. Since
                the number of background boxes in `y_true` will ususally exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            alpha (float): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.loc_norm = loc_norm
        self.neg_pos_ratio = tf.constant(neg_pos_ratio)
        self.alpha = tf.constant(alpha)

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.

        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).

        References:
            https://arxiv.org/abs/1504.08083
        '''
        # In order to normalize the localization loss, we perform element-wise division by the default box widths and heights.
        # Deviations in xmin and xmax are divided by their respective default box widths, deviations in ymin and ymax are divided
        # by their respective default box heights.
        absolute_loss = tf.abs(y_true - y_pred) / self.loc_norm
        square_loss = 0.5 * (y_true - y_pred)**2 / self.loc_norm
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.

        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.

        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.

        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 4)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last dimension must contain
                `[classes 1-hot encoded, 4 box coordinates]` in this order,
                including the background class.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`.

        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        batch_size = tf.shape(y_pred)[0] # tf.int32
        n_boxes = tf.shape(y_pred)[1] # tf.int32
        depth = tf.shape(y_pred)[2] # tf.int32

        # 1: Compute the losses for class and box predictions for each default box

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-4], y_pred[:,:,:-4])) # Tensor of shape (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-4:], y_pred[:,:,-4:])) # Tensor of shape (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets

        # Count the number of positive (classes [1:]) and negative (class 0) boxes in y_true across the whole batch
        n_boxes_batch = batch_size * n_boxes # tf.int32
        n_negative = tf.to_int32(tf.reduce_sum(y_true[:,:,0]))
        n_positive = n_boxes_batch - n_negative

        # Create masks for the positive and negative ground truth classes
        negatives = y_true[:,:,0] # Tensor of shape (batch_size, n_boxes)
        positives = 1 - negatives # Tensor of shape (batch_size, n_boxes)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any)

        # First, compute the classification loss for all negative boxes
        neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) # The number of non-zero loss entries in `neg_class_loss_all`

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
        def f1():
            return tf.zeros([batch_size])
        # Otherwise compute the negative loss
        def f2():
            # Compute the number of negative examples we want to account for in the loss
            # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`
            n_negative_keep = tf.to_int32(tf.minimum(self.neg_pos_ratio * n_positive, n_neg_losses))

            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_negative_keep, False) # We don't need sorting
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(tf.expand_dims(indices, axis=1), updates=tf.ones_like(indices, dtype=tf.int32), shape=tf.shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1) # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss

        # 3: Compute the localization loss for the positive targets
        #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # Tensor of shape (batch_size,)

        # 4: Compute the total loss

        total_loss = (class_loss + self.alpha * loc_loss) / tf.to_float(n_positive)

        return total_loss
