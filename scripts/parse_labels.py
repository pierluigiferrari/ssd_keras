
# %%
import os
from collections import namedtuple

import pandas as pd
from bs4 import BeautifulSoup

Box = namedtuple("Box", ["file_name", "xmin", "xmax", "ymin", "ymax", "class_id"])

PATH_TRAIN = "../../datasets/tires-data/train/"
PATH_VALID = "../../datasets/tires-data/valid/"
PATH_LABEL = "../../datasets/tires-data/"
CLASS_ID = 0

# label files to collect boxes from
train_files = [i for i in os.listdir(PATH_TRAIN) if i.endswith(".xml")]
valid_files = [i for i in os.listdir(PATH_VALID) if i.endswith(".xml")]


# %%
def box_from_object(obj):
    xmin_ = obj.find("xmin").text
    ymin_ = obj.find("ymin").text
    xmax_ = obj.find("xmax").text
    ymax_ = obj.find("ymax").text
    return xmin_, ymin_, xmax_, ymax_


# %% TRAIN IMAGES
df_train = pd.DataFrame(columns=["file_name", "xmin", "xmax", "ymin", "ymax", "class_id"])
for voc_file in train_files:
    # read xml
    with open(PATH_TRAIN + voc_file) as f:
        xml = f.read()
    soup = BeautifulSoup(xml)
    # get filename
    file_name = soup.find("filename").text
    # get bounding box
    objects = soup.find_all("object")
    # write each box to dataframe:
    for obj in objects:
        # parse corners
        xmin, ymin, xmax, ymax = box_from_object(obj)
        # data to dictionary
        box = Box(file_name, xmin, xmax, ymin, ymax, CLASS_ID)
        box_dict = dict(box._asdict())
        # append to dataframe
        df_train = df_train.append(box_dict, ignore_index=True)


# %% TEST IMAGES
df_valid = pd.DataFrame(columns=["file_name", "xmin", "xmax", "ymin", "ymax", "class_id"])
for voc_file in valid_files:
    # read xml
    with open(PATH_VALID + voc_file) as f:
        xml = f.read()
    soup = BeautifulSoup(xml)
    # get filename
    file_name = soup.find("filename").text
    # get bounding box
    objects = soup.find_all("object")
    # write each box to dataframe:
    for obj in objects:
        # parse corners
        xmin, ymin, xmax, ymax = box_from_object(obj)
        # data to dictionary
        box = Box(file_name, xmin, xmax, ymin, ymax, CLASS_ID)
        box_dict = dict(box._asdict())
        # append to dataframe
        df_valid = df_valid.append(box_dict, ignore_index=True)

# %% save results to csv
df_train.to_csv(PATH_LABEL + "labels_train.csv", index=False)
df_valid.to_csv(PATH_LABEL + "labels_valid.csv", index=False)

# %%
