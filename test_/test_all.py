from models.keras_ssd7 import build_model
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation


from keras_loss_function.keras_ssd_loss import SSDLoss

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.models import Model

import numpy as np
from matplotlib import pyplot

GlobalParameters = namedtuple("Parameters",
                              ["img_height", "img_width", "img_channels",
                               "intensity_mean", "intensity_range", "n_classes",
                               "scales", "aspect_ratios", "two_boxes_for_ar1",
                               "steps", "offsets", "clip_boxes", "variances",
                               "normalize_coords"])
p = GlobalParameters
p.img_height = 300
p.img_width = 480
p.img_channels = 3
p.intensity_mean = 127.5
p.intensity_range = 127.5
p.n_classes = 5
p.scales = [0.08, 0.16, 0.32, 0.64, 0.96]
p.aspect_ratios = [0.5, 1.0, 2.0]
p.two_boxes_for_ar1 = True
p.steps = None
p.offsets = None
p.clip_boxes = False
p.variances = [1.0, 1.0, 1.0, 1.0]
p.normalize_coords = True

EncodingParameters = namedtuple("EncodingParameters",
                                ["pos_iou_threshold", "neg_iou_limit",
                                 "matching_type"])
pe = EncodingParameters
pe.pos_iou_threshold = 0.5
pe.neg_iou_limit = 0.3
pe.matching_type = "multi"


class test_Model:
    def __init__(self):
        self.model: Model = self.set_model()

    def set_model(self):
        self.model = build_model(
            image_size=(p.img_height, p.img_width, p.img_channels),
            n_classes=p.n_classes,
            mode="training",
            l2_regularization=0.0005,
            scales=p.scales,
            aspect_ratios_global=p.aspect_ratios,
            aspect_ratios_per_layer=None,
            two_boxes_for_ar1=p.two_boxes_for_ar1,
            steps=p.steps,
            offsets=p.offsets,
            clip_boxes=p.clip_boxes,
            variances=p.variances,
            normalize_coords=p.normalize_coords,
            subtract_mean=p.intensity_mean,
            divide_by_stddev=p.intensity_range)
        return self.model

    def test_summary(self):
        print(self.model.summary())

    def test_forward_pass(self):
        BATCH_SIZE = 32
        X_shape = [BATCH_SIZE] + list(p.img_height, p.img_width, p.img_channels)
        X = np.random.randint(0, 255, size=X_shape).astype(np.float32) / 255.0
        X_ = tf.constant(X)
        model = self.set_model()
        y_pred = model(X_)
        print(y_pred.shape)


class test_Data_Generator:
    def __init__(self):
        self.images_dir = "K:/datasets/udacity_driving_datasets"
        self.train_labels_filename = "K:/datasets/udacity_driving_datasets/labels_train.csv"
        self.val_labels_filename = "K:/datasets/udacity_driving_datasets/labels_val.csv"

        self.train_dataset, self.val_dataset = self.test_init_datasets(log=False)

    def test_init_datasets(self, log=True):
        from data_generator.object_detection_2d_data_generator import DataGenerator
        train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        if log:
            print(f"{train_dataset}\n{val_dataset}")
        return train_dataset, val_dataset

    def test_parse_csv(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset

        dataset.parse_csv(images_dir=self.images_dir,
                          labels_filename=self.train_labels_filename,
                          input_format=["image_name", "xmin", "xmax", "ymin", "ymax", "class_id"],
                          include_classes="all")

        print(dataset.filenames[0])
        print(dataset.labels[0])
        print(dataset.image_ids[0])

    @staticmethod
    def __set_transformations():
        from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
        data_augmentation_chain = DataAugmentationConstantInputSize(
            random_brightness=(-48, 48, 0.5),
            random_contrast=(0.5, 1.8, 0.5),
            random_saturation=(0.5, 1.8, 0.5),
            random_hue=(18, 0.5),
            random_flip=0.5,
            random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
            random_scale=(0.5, 2.0, 0.5),
            n_trials_max=3,
            clip_boxes=True,
            overlap_criterion='area',
            bounds_box_filter=(0.3, 1.0),
            bounds_validator=(0.5, 1.0),
            n_boxes_min=1,
            background=(0, 0, 0))
        return data_augmentation_chain

    @staticmethod
    def __set_ssd_input_encoder():
        input_encoder = Set_input_encoder()
        return input_encoder()

    def test_generate_train_dataset(self, log=True):
        self.train_dataset.parse_csv(images_dir=self.images_dir,
                                     labels_filename=self.train_labels_filename,
                                     input_format=["image_name", "xmin", "xmax", "ymin", "ymax", "class_id"],
                                     include_classes="all")
        train_gen = self.train_dataset.generate(
            batch_size=16,
            shuffle=True,
            transformations=[self.__set_transformations()],
            label_encoder=self.__set_ssd_input_encoder(),
            returns={"processed_images", "encoded_labels"},
            keep_images_without_gt=False
        )
        if log:
            print(train_gen)
        return train_gen

    def test_generate_valid_dataset(self, log=True):
        self.val_dataset.parse_csv(images_dir=self.images_dir,
                                   labels_filename=self.val_labels_filename,
                                   input_format=["image_name", "xmin", "xmax", "ymin", "ymax", "class_id"],
                                   include_classes="all")
        val_gen = self.val_dataset.generate(
            batch_size=16,
            shuffle=False,
            transformations=[],
            label_encoder=self.__set_ssd_input_encoder(),
            returns={"processed_images", "encoded_labels"},
            keep_images_without_gt=False
        )
        if log:
            print(val_gen)
        return val_gen


class test_Data_Augmentation:
    def __init__(self):
        image_file = "fish_bike.jpg"
        path = "images"
        image_file_path = os.path.join("E:/10.Repos/vct-ml-ssd_keras", path, image_file)
        self.image = cv2.imread(image_file_path)
        print("image shape: ", self.image.shape)

    def test_DataAugmentationConstantInputSize(self):
        from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
        data_augmentation_chain = DataAugmentationConstantInputSize(
            random_brightness=(-48, 48, 0.5),
            random_contrast=(0.5, 1.8, 0.5),
            random_saturation=(0.5, 1.8, 0.5),
            random_hue=(18, 0.5),
            random_flip=0.5,
            random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
            random_scale=(0.5, 2.0, 0.5),
            n_trials_max=3,
            clip_boxes=True,
            overlap_criterion='area',
            bounds_box_filter=(0.3, 1.0),
            bounds_validator=(0.5, 1.0),
            n_boxes_min=1,
            background=(0, 0, 0))
        image_augmented = data_augmentation_chain(self.image)
        print(image_augmented)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(self.image)
        axes[1].imshow(image_augmented)
        plt.show()


class test_train:
    def __init__(self):
        self.model = test_Model().set_model()
        self.train_generator = test_Data_Generator().test_generate_train_dataset(log=False)
        self.val_generator = test_Data_Generator().test_generate_valid_dataset(log=False)
        self.initial_epoch = 0
        self.final_Epoch = 3
        self.steps_per_epoch = 10

    @staticmethod
    def __set_callbacks():
        callbacks = Set_callbacks()
        return callbacks()

    def __set_compile(self):
        self.model.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, decay=0.0),
            loss=SSDLoss(neg_pos_ratio=3, alpha=1.0).compute_loss
        )

    def train(self, show=True):
        self.__set_compile()
        history = self.model.fit_generator(
            generator=self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            initial_epoch=self.initial_epoch,
            epochs=self.final_Epoch,
            callbacks=self.__set_callbacks(),
            validation_data=self.val_generator,
            validation_steps=10,  # TODO set ceil(val dataset size/batch size)
        )
        if show:
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(history.history["loss"], label="loss")
            axes[0].legend(loc="upper right", prop={"size": 24})
            axes[1].plot(history.history["val_loss"], label="val_loss")
            axes[1].legend(loc="upper right", prop={"size": 24})
            plt.show()
        return history


class Set_input_encoder:
    def __init__(self):
        self.model: Model = test_Model().set_model()

    def __get_predictor_sizes(self):
        predictor_sizes = [
            self.model.get_layer('classes4').output_shape[1:3],
            self.model.get_layer('classes5').output_shape[1:3],
            self.model.get_layer('classes6').output_shape[1:3],
            self.model.get_layer('classes7').output_shape[1:3]]
        return predictor_sizes

    def __call__(self, *args, **kwargs):
        ssd_input_encoder = SSDInputEncoder(
            img_height=p.img_height,
            img_width=p.img_width,
            n_classes=p.n_classes,
            predictor_sizes=self.__get_predictor_sizes(),
            scales=p.scales,
            aspect_ratios_global=p.aspect_ratios,
            two_boxes_for_ar1=p.two_boxes_for_ar1,
            steps=p.steps,
            offsets=p.offsets,
            clip_boxes=p.clip_boxes,
            variances=p.variances,
            matching_type=pe.matching_type,
            pos_iou_threshold=pe.pos_iou_threshold,
            neg_iou_limit=pe.neg_iou_limit,
            normalize_coords=p.normalize_coords)
        return ssd_input_encoder


class Set_callbacks:
    def __init__(self):
        self.checkpoint_dir = "checkpoints"
        self.cvs_logger_dir = "csv_log_dir"
        self.root = os.path.dirname(os.getcwd())

    def __call__(self, *args, **kwargs):
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(
                self.root,
                self.checkpoint_dir,
                "ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1)
        cvs_Logger = CSVLogger(
            filename=os.path.join(
                self.root,
                self.cvs_logger_dir,
                "ssd7_training_log.csv"),
            separator=",",
            append=True)
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=10,
            verbose=1)
        reduce_learning_rate = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=8,
            verbose=1,
            epsilon=0.001,
            cooldown=0,
            min_lr=0.00001)
        callbacks = [model_checkpoint, cvs_Logger, early_stopping, reduce_learning_rate]
        return callbacks


if __name__ == "__main__":
    # t = test_Model()
    # t.test_summary()
    # t.test_forward_pass()

    # t = test_Data_Generator()
    # t.test_init()
    # t.test_parse_csv()
    # t.test_generate()

    # t = test_Data_Augmentation()

    t = test_train()
    t.train()
