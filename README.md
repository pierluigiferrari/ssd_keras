###README is under construction and will be expanded soon

## SSD implementation in Keras
---

This is a Keras implementation of the SSD model architecture proposed in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

This project is still in development. Parts that will be added or updated soon are, among others:
* Currently there is only a smaller 7-layer architecture (keras_ssd7.py) in the repository. I will add the original VGG-16-based architecture with trained weights soon.
* The NMS (non-maximum suppression) stage has not yet been implemented. I will add it soon.

A note on dependencies:
This code requires Keras 2.0 or later, which in turn requires TensorFlow 1.0 or later. Both Keras and TensorFlow underwent some syntax changes in those releases, so the code won't run on older versions.

In this repository:
* keras_ssd7.py contains the Keras SSD7 model, a smaller version of the original VGG-16 model from the paper.
* keras_layer_L2Normalization.py contains a custom L2 normalization layer. SSD7 does not implement this normalization layer, but the original VGG-16 implementation needs it to adjust for the pretrained weights.
* keras_ssd_loss.py contains the custom loss function for the SSD model.
* ssd_box_encode_decode_utils.py contains utilities to encode ground truth labels into the format required by the loss function to train the SSD model, and also a function to decode the output from the model for inference.
* ssd_batch_generator.py contains a generator to generate mini-batches for training or inference. Note the label format it requires, see the documentation.
* train_ssd7.ipynb contains the training setup as an example.
