'''
Includes:
* A batch generator for SSD model training and inference which can perform online data agumentation
* An offline image processor that saves processed images and adjusted labels to disk
'''

import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from copy import deepcopy
from PIL import Image
import csv

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

    image1[:,:,0] = cv2.equalizeHist(image1[:,:,0])
    image1[:,:,1] = cv2.equalizeHist(image1[:,:,1])
    image1[:,:,2] = cv2.equalizeHist(image1[:,:,2])

    return image1

class Batch_Generator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Shuffles the dataset consistently after each complete pass.

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    def __init__(self,
                 images_path='./data/',
                 labels_path='./data/labels.csv',
                 n_classes=None):
        '''
        Arguments:
            images_path (str): The filepath to the image samples, ending on a slash.
            labels_path (str): The filepath to a CSV file that contains lines with
                `(image file name, xmin, xmax, ymin, ymax, class_id)` for each ground truth bounding box.
                `xmin` and `xmax` are the left-most and right-most horizontal coordinates of the box,
                `ymin` and `ymax` are the top-most and bottom-most vertical coordinates of the box.
                `class_id` is an integer greater than zero that is the class number associated with a given
                box. The class ID 0 is reserved for the background class. The image file name is expected
                to be just the name of the image file without the directory path at which the image is located.
            n_classes (int, optional): If set, limits the number of classes included in the generated dataset
                to the first `n_classes` classes. For example, if a label CSV file contains 10 different
                positive classes and `n_classes` is set to be 4, then only the first three positive classes
                will be included in the generated dataset (the fourth class being the background class, 0),
                i.e. the dataset would contain the classes 0, 1, 2, and 3. This can be useful for experimental
                purposes to limit the problem complexity by reducing the number of classes the model needs to
                learn to distinguish. Defaults to `None`, in which case all classes found in the labels CSV
                file will be included in the dataset.
        '''
        self.images_path = images_path
        self.labels_path = labels_path
        self.n_classes = n_classes
        # `self.filenames` is a list containing all file names of the image samples. Note that it does not contain the actual image files themselves.
        self.filenames = [] # All unique image filenames will go here
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        self.labels = [] # Each entry here will contain a 2D Numpy array with all the ground truth boxes for a given image

        ### Parse the file names and labels from the labels CSV file.

        data = []

        # First, just read in the CSV file lines and sort them.

        with open(self.labels_path, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            k = 0
            if self.n_classes:
                for i in csvread:
                    if k == 0: # Skip the header row
                        k += 1
                        continue
                    else:
                        if int(i[5].strip()) >= self.n_classes:
                            continue
                        else:
                            data.append([i[0].strip(),
                                         int(i[1].strip()),
                                         int(i[2].strip()),
                                         int(i[3].strip()),
                                         int(i[4].strip()),
                                         int(i[5].strip())])
            else:
                for i in csvread:
                    if k == 0: # Skip the header row
                        k += 1
                        continue
                    else:
                        data.append([i[0].strip(),
                                     int(i[1].strip()),
                                     int(i[2].strip()),
                                     int(i[3].strip()),
                                     int(i[4].strip()),
                                     int(i[5].strip())])

        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        current_file = '' # The current image for which we're collecting the ground truth boxes
        current_labels = [] # The list where we collect all ground truth boxes for a given image
        for idx, i in enumerate(data):
            if current_file == '': # If this is the first image file
                current_file = i[0]
                current_labels.append(i[1:])
                if len(data) == 1: # If there is only one box in the CVS file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
            else:
                if i[0] == current_file: # If this box (i.e. this line of the CSV file) belongs to the current image file
                    current_labels.append(i[1:])
                    if idx == len(data)-1: # If this is the last line of the CSV file
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(current_file)
                else: # If this box belongs to a new image file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(current_file)
                    current_labels = []
                    current_file = i[0]
                    current_labels.append(i[1:])

    def generate(self,
                 batch_size=32,
                 train=True,
                 ssd_box_encoder=None,
                 crop=False,
                 resize=False,
                 gray=False,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 limit_boxes=True,
                 include_thresh=0.3,
                 diagnostics=False):
        '''
        Generate batches of samples and corresponding labels indefinitely from
        lists of filenames and labels.

        Returns two numpy arrays, one containing the next `batch_size` samples
        from `filenames`, the other containing the corresponding labels from
        `labels`.

        Shuffles `filenames` and `labels` consistently after each complete pass.

        Can perform image transformations for data conversion and data augmentation.
        `resize`, `gray`, and `equalize` are image conversion tools and should be
        used consistently during training and inference. The remaining transformations
        serve for data augmentation. Each data augmentation process can set its own
        independent application probability.

        `prob` works the same way in all arguments in which it appears. It must be a float in [0,1]
        and determines the probability that the respective transform is applied to any given image.

        All conversions and transforms default to `False`.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated. Defaults to 32.
            train (bool, optional): Whether or not the generator is used in training mode. If `True`, then the labels
                will be transformed into the format that the SSD cost function requires. Otherwise,
                the output format of the labels is identical to the input format. Defaults to `True`.
            ssd_box_encoder (SSDBoxEncoder, optional): Only required if `train = True`. An SSDBoxEncoder object
                to encode the ground truth labels to the required format for training an SSD model.
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images.
                The targets are adjusted accordingly. Note: Cropping happens before resizing.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected format is `(width, height)`.
                The box coordinates are adjusted accordingly. Note: Resizing happens after cropping.
            gray (bool, optional): If `True`, converts the images to grayscale.
            equalize (bool, optional): If `True`, performs histogram equalization on the images.
                This can improve contrast and lead the improved model performance.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
                The respective box coordinates are adjusted accordingly.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for horizontal shift of the image,
                the second tuple for vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
                The respective box coordinates are adjusted accordingly.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both min and max must be >=0.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries
                post any transformation.
            include_thresh (float, optional): Determines the minimum fraction of the area of a ground truth box
                that must be left after clipping in order for the box to still be included in the batch data.
                Only relevant if `limit_boxes` is `True`. Defaults to 0.3.
            diagnostics (bool, optional): If `True`, yields three additional output items:
                1) A list of the image file names in the batch.
                2) An array with the original, unaltered images.
                3) A list with the original, unaltered labels.
                This can be useful for diagnostic purposes. Defaults to `False`. Only works if `train = True`.

        Yields:
            The next batch as a tuple containing a numpy array that contains the images and a python list
            that contains the corresponding labels for each image as 2D numpy arrays.
        '''

        self.filenames, self.labels = shuffle(self.filenames, self.labels) # Shuffle the data before we begin
        current = 0

        while True:

            batch_X, batch_y = [], []

            #Shuffle the data after each complete pass
            if current >= len(self.filenames):
                self.filenames, self.labels = shuffle(self.filenames, self.labels)
                current = 0

            for filename in self.filenames[current:current+batch_size]:
                with Image.open('{}{}'.format(self.images_path, filename)) as img:
                    batch_X.append(np.array(img))
            batch_y = deepcopy(self.labels[current:current+batch_size])

            if diagnostics:
                ret1 = self.filenames[current:current+batch_size] # The filenames of the files in the batch
                ret2 = np.copy(batch_X) # The original, unaltered images
                ret3 = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            # At this point we're done producing the batch. Now perform some
            # optional image transformations:

            for i in range(len(batch_X)):

                img_height, img_width, ch = batch_X[i].shape

                if equalize:
                    batch_X[i] = histogram_eq(batch_X[i])

                if brightness:
                    p = np.random.uniform(0,1)
                    if p >= (1-brightness[2]):
                        batch_X[i] = _brightness(batch_X[i], min=brightness[0], max=brightness[1])

                # Could easily be extended to also allow vertical flipping, but I'm not convinced of the
                # usefulness of vertical flipping either empirically or theoretically, so I'm going for simplicity.
                # If you want to allow vertical flipping, just change this function to pass the respective argument
                # to `_flip()`.
                if flip:
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        batch_X[i] = _flip(batch_X[i])
                        batch_y[i][:,[0,1]] = img_width - batch_y[i][:,[1,0]] # xmin and xmax are swapped when mirrored

                if translate:
                    p = np.random.uniform(0,1)
                    if p >= (1-translate[2]):
                        batch_X[i], xshift, yshift = _translate(batch_X[i], translate[0], translate[1])
                        batch_y[i][:,[0,1]] += xshift
                        batch_y[i][:,[2,3]] += yshift
                        if limit_boxes:
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:,[0,1]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:,[0,1]] = x_coords
                            y_coords = batch_y[i][:,[2,3]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:,[2,3]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                            after_area = (batch_y[i][:,1] - batch_y[i][:,0]) * (batch_y[i][:,3] - batch_y[i][:,2])
                            batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]

                if scale:
                    p = np.random.uniform(0,1)
                    if p >= (1-scale[2]):
                        batch_X[i], M, scale_factor = _scale(batch_X[i], scale[0], scale[1])
                        # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                        toplefts = np.array([batch_y[i][:,0], batch_y[i][:,2], np.ones(batch_y[i].shape[0])])
                        bottomrights = np.array([batch_y[i][:,1], batch_y[i][:,3], np.ones(batch_y[i].shape[0])])
                        new_toplefts = (np.dot(M, toplefts)).T
                        new_bottomrights = (np.dot(M, bottomrights)).T
                        batch_y[i][:,[0,2]] = new_toplefts.astype(np.int)
                        batch_y[i][:,[1,3]] = new_bottomrights.astype(np.int)
                        if limit_boxes and (scale_factor > 1): # We don't need to do any limiting in case we shrunk the image
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:,[0,1]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:,[0,1]] = x_coords
                            y_coords = batch_y[i][:,[2,3]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:,[2,3]] = y_coords
                            # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                            # process that they don't serve as useful training examples anymore, because too little of them is
                            # visible. We'll remove all boxes that we had to limit so much that their area is less than
                            # `include_thresh` of the box area before limiting.
                            before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                            after_area = (batch_y[i][:,1] - batch_y[i][:,0]) * (batch_y[i][:,3] - batch_y[i][:,2])
                            batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]

                if crop:
                    batch_X[i] = np.copy(batch_X[i][crop[0]:img_height-crop[1], crop[2]:img_width-crop[3]])
                    if limit_boxes:
                        before_limiting = deepcopy(batch_y[i])
                        if crop[0] > 0:
                            y_coords = batch_y[i][:,[2,3]]
                            y_coords[y_coords < crop[0]] = crop[0]
                            batch_y[i][:,[2,3]] = y_coords
                        if crop[1] > 0:
                            y_coords = batch_y[i][:,[2,3]]
                            y_coords[y_coords >= (img_height - crop[1])] = img_height - crop[1] - 1
                            batch_y[i][:,[2,3]] = y_coords
                        if crop[2] > 0:
                            x_coords = batch_y[i][:,[0,1]]
                            x_coords[x_coords < crop[2]] = crop[2]
                            batch_y[i][:,[0,1]] = x_coords
                        if crop[3] > 0:
                            x_coords = batch_y[i][:,[0,1]]
                            x_coords[x_coords >= (img_width - crop[3])] = img_width - crop[3] - 1
                            batch_y[i][:,[0,1]] = x_coords
                        # Some objects might have gotten pushed so far outside the image boundaries in the transformation
                        # process that they don't serve as useful training examples anymore, because too little of them is
                        # visible. We'll remove all boxes that we had to limit so much that their area is less than
                        # `include_thresh` of the box area before limiting.
                        before_area = (before_limiting[:,1] - before_limiting[:,0]) * (before_limiting[:,3] - before_limiting[:,2])
                        after_area = (batch_y[i][:,1] - batch_y[i][:,0]) * (batch_y[i][:,3] - batch_y[i][:,2])
                        batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]
                    if crop[0] > 0:
                        batch_y[i][:,[2,3]] -= crop[0]
                    if crop[2] > 0:
                        batch_y[i][:,[0,1]] -= crop[2]
                    img_height -= crop[0] - crop[1]
                    img_width -= crop[2] - crop[3]

                if resize:
                    batch_X[i] = cv2.resize(batch_X[i], dsize=resize)
                    batch_y[i][:,[0,1]] = (batch_y[i][:,[0,1]] * (resize[0] / img_width)).astype(np.int)
                    batch_y[i][:,[2,3]] = (batch_y[i][:,[2,3]] * (resize[1] / img_height)).astype(np.int)

                if gray:
                    batch_X[i] = np.expand_dims(cv2.cvtColor(batch_X[i], cv2.COLOR_RGB2GRAY), 3)

            if train:
                if ssd_box_encoder is None:
                    raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                y_true = ssd_box_encoder.encode_y(batch_y)

            if train:
                if diagnostics:
                    yield (np.array(batch_X), y_true, batch_y, ret1, ret2, ret3)
                else:
                    yield (np.array(batch_X), y_true)
            else:
                yield (np.array(batch_X), batch_y)

    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames and the list of labels.
        '''
        return self.filenames, self.labels

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
                        resize=False,
                        gray=False,
                        equalize=False,
                        brightness=False,
                        flip=False,
                        translate=False,
                        scale=False,
                        limit_boxes=True,
                        include_thresh=0.3,
                        diagnostics=False):
        '''
        Perform offline image processing.

        This function the same image processing capabilities as the generator function above,
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

        for k, filename in enumerate(self.filenames[start:stop]):
            i = k + start
            with Image.open('{}{}'.format(self.images_path, filename)) as img:
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
