
# %%
import os

import cv2

PATH_TRAIN = "../../datasets/tires-data/train/"
PATH_VALID = "../../datasets/tires-data/valid/"

TARGET_SIZE = (300, 300)

train_files = [i for i in os.listdir(PATH_TRAIN) if i.endswith(".jpg")]
valid_files = [i for i in os.listdir(PATH_VALID) if i.endswith(".jpg")]

# %%
for train_file in train_files:
    img = cv2.imread(PATH_TRAIN + train_file)
    img_resized = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))
    cv2.imwrite(PATH_TRAIN + train_file, img_resized)

# %%
for valid_file in valid_files:
    img = cv2.imread(PATH_VALID + valid_file)
    img_resized = cv2.resize(img, (TARGET_SIZE[1], TARGET_SIZE[0]))
    cv2.imwrite(PATH_VALID + valid_file, img_resized)

# %%
