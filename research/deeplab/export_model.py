# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

# Converted export script for DeepLab to run under TensorFlow 2.x (compat.v1).
# Save as export_tf2_compat.py and run with a TF2.x environment that has tf-slim.
#
# Usage example:
# python export_tf2_compat.py --checkpoint_path=/path/to/model.ckpt --export_path=/path/to/out.pb --dataset_dir=... --model_variant=xception_65

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import contextlib

# Use TF2 but run TF1-style code via compat.v1
import tensorflow as _tf_orig
_tf_orig.compat.v1.disable_v2_behavior()
tf = _tf_orig.compat.v1

# Try to import contrib quantize (may be missing); fallback to None.
try:
  from tensorflow.contrib import quantize as contrib_quantize  # type: ignore
except Exception:
  contrib_quantize = None

# freeze_graph tool (compat)
try:
  from tensorflow.python.tools import freeze_graph
except Exception:
  freeze_graph = None

from deeplab import common
from deeplab import input_preprocess
from deeplab import model

# Attempt to import slim from tf_slim package first.
try:
  import tf_slim as slim  # type: ignore
except Exception:
  try:
    slim = tf.contrib.slim  # type: ignore
  except Exception:
    slim = None

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')
flags.DEFINE_string('export_path', None, 'Path to output Tensorflow frozen graph.')
flags.DEFINE_integer('num_classes', 21, 'Number of classes.')
flags.DEFINE_multi_integer('crop_size', [513, 513], 'Crop size [height, width].')
flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 8, 'The ratio of input to output spatial resolution.')
flags.DEFINE_multi_float('inference_scales', [1.0], 'The scales to resize images for inference.')
flags.DEFINE_bool('add_flipped_images', False, 'Add flipped images during inference or not.')
flags.DEFINE_integer('quantize_delay_step', -1, 'Steps to start quantized training. If < 0, will not quantize model.')
flags.DEFINE_bool('save_inference_graph', False, 'Save inference graph in text proto.')

# Input / output names
_INPUT_NAME = 'ImageTensor'
_OUTPUT_NAME = 'SemanticPredictions'
_RAW_OUTPUT_NAME = 'RawSemanticPredictions'
_OUTPUT_PROB_NAME = 'SemanticProbabilities'
_RAW_OUTPUT_PROB_NAME = 'RawSemanticProbabilities'


def _create_input_tensors():
  """Creates and prepares input tensors for DeepLab model."""
  input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=_INPUT_NAME)
  original_image_size = tf.shape(input_image)[1:3]

  # Squeeze to 3-D for preprocessing (matches original code)
  image = tf.squeeze(input_image, axis=0)
  resized_image, image, _ = input_preprocess.preprocess_image_and_label(
      image,
      label=None,
      crop_height=int(FLAGS.crop_size[0]),
      crop_width=int(FLAGS.crop_size[1]),
      min_resize_value=FLAGS.min_resize_value if hasattr(FLAGS, 'min_resize_value') else None,
      max_resize_value=FLAGS.max_resize_value if hasattr(FLAGS, 'max_resize_value') else None,
      resize_factor=FLAGS.resize_factor if hasattr(FLAGS, 'resize_factor') else None,
      is_training=False,
      model_variant=FLAGS.model_variant)
  resized_image_size = tf.shape(resized_image)[:2]

  image = tf.expand_dims(image, 0)
  return image, original_image_size, resized_image_size


def _resize_label(label, label_size):
  # Expand dimension of label to [1, height, width, 1] for resize operation.
  label = tf.expand_dims(label, 3)
  # Use TF1 resize op (available under compat.v1)
  resized_label = tf.image.resize_images(
      label,
      label_size,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(resized_label, 3), tf.int32)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

  with tf.Graph().as_default():
    image, image_size, resized_image_size = _create_input_tensors()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
        crop_size=FLAGS.crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.inference_scales) == (1.0,):
      tf.logging.info('Exported model performs single-scale inference.')
      predictions = model.predict_labels(
          image,
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Exported model performs multi-scale inference.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError('Quantize mode is not supported with multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          image,
          model_options=model_options,
          eval_scales=FLAGS.inference_scales,
          add_flipped_images=FLAGS.add_flipped_images)

    raw_predictions = tf.identity(
        tf.cast(predictions[common.OUTPUT_TYPE], tf.float32),
        _RAW_OUTPUT_NAME)
    raw_probabilities = tf.identity(
        predictions[common.OUTPUT_TYPE + model.PROB_SUFFIX],
        _RAW_OUTPUT_PROB_NAME)

    # Crop the valid regions from the predictions.
    semantic_predictions = raw_predictions[:, :resized_image_size[0], :resized_image_size[1]]
    semantic_probabilities = raw_probabilities[:, :resized_image_size[0], :resized_image_size[1]]

    # Resize back the prediction to the original image size.
    semantic_predictions = _resize_label(semantic_predictions, image_size)
    semantic_predictions = tf.identity(semantic_predictions, name=_OUTPUT_NAME)

    semantic_probabilities = tf.image.resize_bilinear(
        semantic_probabilities, image_size, align_corners=True, name=_OUTPUT_PROB_NAME)

    if FLAGS.quantize_delay_step >= 0 and contrib_quantize is not None:
      contrib_quantize.create_eval_graph()

    # Saver and freeze
    saver = tf.train.Saver(tf.global_variables())

    dirname = os.path.dirname(FLAGS.export_path)
    tf.gfile.MakeDirs(dirname)

    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

    if freeze_graph is None:
      raise RuntimeError('freeze_graph tool not available in this TF build.')

    # Use freeze_graph_with_def_protos for compatibility with TF1 API
    freeze_graph.freeze_graph_with_def_protos(
        graph_def,
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)

    if FLAGS.save_inference_graph:
      tf.train.write_graph(graph_def, dirname, 'inference_graph.pbtxt')


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()
