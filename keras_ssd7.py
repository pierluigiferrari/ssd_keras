import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation

def build_model(image_size,
                n_classes,
                n_boxes):
    '''
    Build a Keras model with SSD architecture, see references.

    The model consists of convolutional feature layers and a number of convolutional
    classifier layers that take their input from different feature layers.
    The model is fully convolutional.

    The implementation found here is a smaller version of the original architecture
    used in the paper (where the base network consists of a modified VGG-16 extended
    by a few convolutional feature layers), but of course it could easily be changed to
    an arbitrarily large SSD architecture by following the general design pattern used here.
    This implementation has 7 convolutional layers and 4 convolutional classifier
    layers that take their input from layers 4, 5, 6, and 7, respectively.

    Note: Requires Keras v2.0 or later. Training currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(height, width, channels)`.
        n_classes (int): The number of categories for classification including
            the background class (i.e. the number of positive classes +1 for
            the background calss).
        n_boxes (int): The number of boxes the model will generate per cell,
            where a 2D convolutional classifier layer with output shape
            `(batch, height, width, depth)` has `height * width` cells. The purpose
            of multiple boxes per cell is for the different feature maps of a
            classifier layer to learn to recognize objects of specific shapes
            (i.e. specific aspect ratios) within the same spatial location in
            an image.

    Returns:
        model: The Keras SSD model.
        classifier_sizes: A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional classifier. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    # Input image format
    in_row, in_col, ch = image_size[0], image_size[1], image_size[2]

    # Input layer and normalization
    x = Input(shape=(in_row, in_col, ch))
    normed = Lambda(lambda z: z/127.5 - 1., # Convert input feature range to [-1,1]
                    output_shape=(in_row, in_col, ch),
                    name='lambda1')(x)

    # Build the network
    conv1 = Convolution2D(32, (5, 5), name='conv1', strides=(1, 1), padding="same")(normed)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Convolution2D(48, (3, 3), name='conv2', strides=(1, 1), padding="same")(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Convolution2D(64, (3, 3), name='conv3', strides=(1, 1), padding="same")(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Convolution2D(64, (3, 3), name='conv4', strides=(1, 1), padding="same")(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Convolution2D(48, (3, 3), name='conv5', strides=(1, 1), padding="same")(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Convolution2D(48, (3, 3), name='conv6', strides=(1, 1), padding="same")(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Convolution2D(32, (3, 3), name='conv7', strides=(1, 1), padding="same")(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)

    # Build the convolutional classifiers on top of conv layers 4, 5, 6, and 7
    # We build two classifiers on top of each of these layers: One for classes, one for boxes
    # We precidt a class for each box, hence the classes classifiers have depth n_classes*n_boxes
    # We predict 4 box offset values delta(cx,cy,w,h) for each box, hence the boxes classifiers have depth n_boxes*4
    # Output shape of classes: (batch, height, width, n_boxes * n_classes)
    classes4 = Convolution2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes4')(conv4)
    classes5 = Convolution2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes5')(conv5)
    classes6 = Convolution2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes6')(conv6)
    classes7 = Convolution2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="valid", name='classes7')(conv7)
    # Output shape of boxes: (batch, height, width, n_boxes * 4)
    boxes4 = Convolution2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes4')(conv4)
    boxes5 = Convolution2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes5')(conv5)
    boxes6 = Convolution2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes6')(conv6)
    boxes7 = Convolution2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="valid", name='boxes7')(conv7)

    # Reshape the class predictions, yielding 3D tensors of shape (batch, height * width * n_boxes, n_classes)
    # We want the classes in an isolated last axis to perform softmax on
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box predictions, yielding 3D tensors of shape (batch, height * width * n_boxes, 4)
    # We want `(cx,cy,w,h)` in an isolated last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)

    # Concatenate the predictions from the different layers
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_merged`: (batch, n_boxes_total, n_classes)
    classes_merged = Concatenate(axis=1, name='classes_concatenate')([classes4_reshaped,
                                                                      classes5_reshaped,
                                                                      classes6_reshaped,
                                                                      classes7_reshaped])

    # Output shape of `boxes_final`: (batch, n_boxes_total, 4)
    boxes_final = Concatenate(axis=1, name='boxes_final')([boxes4_reshaped,
                                                           boxes5_reshaped,
                                                           boxes6_reshaped,
                                                           boxes7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_final = Activation('softmax', name='classes_final')(classes_merged)

    # Concatenate the class and box predictions to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + n_boxes)
    predictions = Concatenate(axis=2, name='predictions')([classes_final, boxes_final])

    model = Model(inputs=x, outputs=predictions)

    # Get the spatial dimensions (height, width) of the classifier conv layers, we need them to generate the default boxes
    # The spatial dimensions are the same for the classes and boxes classifiers
    classifier_sizes = np.array([classes4._keras_shape[1:3],
                                 classes5._keras_shape[1:3],
                                 classes6._keras_shape[1:3],
                                 classes7._keras_shape[1:3]])

    return model, classifier_sizes
