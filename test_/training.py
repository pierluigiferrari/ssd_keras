from models.keras_ssd7 import build_model
import numpy as np
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot


class test_Model:
    def __init__(self):
        self.image_size = (300, 300, 3)
        self.n_classes = 5
        self.mode = "training"
        self.min_scale = 0.1
        self.max_scale = 0.9
        NUM_PRED_LAYERS = 5
        self.scales = np.linspace(self.min_scale, self.max_scale, NUM_PRED_LAYERS)

    def set_model(self):
        return build_model(image_size=self.image_size,
                           n_classes=self.n_classes,
                           mode=self.mode)

    def test_summary(self):
        model = self.set_model()
        print(model.summary())

    def test_forward_pass(self):
        BATCH_SIZE = 32
        X_shape = [BATCH_SIZE] + list(self.image_size)
        X = np.random.randint(0, 255, size=X_shape).astype(np.float32)/255.0
        X_ = tf.constant(X)
        model = self.set_model()
        y_pred = model(X_)
        print(y_pred.shape)


if __name__ == "__main__":
    t = test_Model()
    # t.test_summary()
    t.test_forward_pass()
