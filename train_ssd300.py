from argparse import ArgumentParser
from math import ceil

import os
import rasterio
import pandas
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_batch_generator import BatchGenerator
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y


parser = ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--name', default='ssd')
parser.add_argument('--scale', type=float)
parser.add_argument('--classes', type=lambda ss: [int(s) for s in ss.split()])
parser.add_argument('--min_scale', type=float, default=.05)
parser.add_argument('--max_scale', type=float, default=.25)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--outcsv', default='ssd_results.csv')
parser.add_argument('csv', default='/osn/share/rail.csv')
args = parser.parse_args()

print(args.min_scale, args.max_scale)
img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
n_classes = len(args.classes)+1 if args.classes else 2
# Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [args.scale] * 7 if args.scale else None
aspect_ratios = [[1.0]] * 6
two_boxes_for_ar1 = True
limit_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'minmax'
normalize_coords = False

K.clear_session()
model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=args.min_scale,
                                 max_scale=args.max_scale,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)
if args.model:
    model.load_weights(args.model, by_name=True)


model.compile(optimizer=(Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)),
              loss=SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1).compute_loss)


ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=args.min_scale,
                                max_scale=args.max_scale,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.4,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)


train_dataset = BatchGenerator(include_classes=args.classes)

train_dataset.parse_csv(labels_path=args.csv,
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

train_generator = train_dataset.generate(batch_size=args.batch_size,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         limit_boxes=True,  # While the anchor boxes are not being clipped,
                                         include_thresh=0.4,
                                         diagnostics=False)


val_dataset = BatchGenerator(include_classes=args.classes)

val_dataset.parse_csv(labels_path=args.csv,
                      input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

val_generator = val_dataset.generate(batch_size=args.batch_size,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     diagnostics=False)


def lr_schedule(epoch):
    if epoch <= 500:
        return 0.001
    else:
        return 0.0001


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(train_dataset.count / args.batch_size),
                              epochs=args.epochs,
                              callbacks=[ModelCheckpoint('./{}_epoch{epoch:04d}_loss{loss:.4f}.h5',
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=False,
                                                         save_weights_only=False,
                                                         mode='auto',
                                                         period=1),
                                         LearningRateScheduler(lr_schedule),
                                         ],
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset.count / args.batch_size))

model.save('./{}.h5'.format(args.name))
model.save_weights('./{}_weights.h5'.format(args.name))

print("Model and weights saved as {}[_weights].h5".format(args.name))

predict_generator = val_dataset.generate(batch_size=1,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False)


val_dir = '/osn/SpaceNet-MOD/testing/rgb-ps-dra/300/'

results = []
for r, d, filenames in os.walk(val_dir):
    for filename in filenames:
        if filename.endswith('png'):
            with rasterio.open(os.path.join(r, filename)) as f:
                x = f.read().transpose([1, 2, 0])[np.newaxis, :]
                p = model.predict(x)
                try:
                    y = decode_y(p,
                                 confidence_thresh=0.15,
                                 iou_threshold=0.35,
                                 top_k=200,
                                 input_coords='minmax',
                                 normalize_coords=normalize_coords,
                                 img_height=img_height,
                                 img_width=img_width)
                    for row in y[0]:
                        results.append([filename] + row.tolist())
                except ValueError as e:
                    pass
df = pandas.DataFrame(results, columns=['file_name', 'class_id', 'conf', 'xmin', 'xmax', 'ymin', 'ymax'])
df['class_id'] = df['class_id'].apply(lambda xx: train_dataset.class_map[xx])
df.to_csv(args.outcsv)
