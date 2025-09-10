from __future__ import absolute_import, division, print_function
import os
import contextlib
import tensorflow as _tf_orig
_tf_orig.compat.v1.disable_v2_behavior()
tf = _tf_orig.compat.v1

# Optional contrib modules
try:
    from tensorflow.contrib import quantize as contrib_quantize
except Exception:
    contrib_quantize = None

# Freeze graph compat
try:
    from tensorflow.python.tools import freeze_graph
except Exception:
    freeze_graph = None

from deeplab import common, input_preprocess, model

# tf_slim import
try:
    import tf_slim as slim
except Exception:
    try:
        slim = tf.contrib.slim
    except Exception:
        slim = None

# Use absl for flags and logging
from absl import app, flags, logging

FLAGS = flags.FLAGS

# --- Define flags ---
flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')
flags.DEFINE_string('export_path', None, 'Path to output TensorFlow frozen graph.')
flags.DEFINE_integer('num_classes', 21, 'Number of classes.')
flags.DEFINE_multi_integer('crop_size', [513, 513], 'Crop size [height, width].')
flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for ASPP.')
flags.DEFINE_integer('output_stride', 8, 'Input/output resolution ratio.')
flags.DEFINE_multi_float('inference_scales', [1.0], 'Scales for inference.')
flags.DEFINE_bool('add_flipped_images', False, 'Add flipped images during inference.')
flags.DEFINE_integer('quantize_delay_step', -1, 'Quantize delay steps; <0 disables quantize.')
flags.DEFINE_bool('save_inference_graph', False, 'Save inference graph as pbtxt.')

# Input/output tensor names
_INPUT_NAME = 'ImageTensor'
_OUTPUT_NAME = 'SemanticPredictions'
_RAW_OUTPUT_NAME = 'RawSemanticPredictions'
_OUTPUT_PROB_NAME = 'SemanticProbabilities'
_RAW_OUTPUT_PROB_NAME = 'RawSemanticProbabilities'


def _create_input_tensors():
    """Creates input tensor for DeepLab model."""
    input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=_INPUT_NAME)
    original_image_size = tf.shape(input_image)[1:3]

    # Preprocess
    image = tf.squeeze(input_image, axis=0)
    resized_image, image, _ = input_preprocess.preprocess_image_and_label(
        image,
        label=None,
        crop_height=int(FLAGS.crop_size[0]),
        crop_width=int(FLAGS.crop_size[1]),
        min_resize_value=getattr(FLAGS, 'min_resize_value', None),
        max_resize_value=getattr(FLAGS, 'max_resize_value', None),
        resize_factor=getattr(FLAGS, 'resize_factor', None),
        is_training=False,
        model_variant=getattr(FLAGS, 'model_variant', None)
    )
    image = tf.expand_dims(image, 0)
    resized_image_size = tf.shape(resized_image)[:2]
    return image, original_image_size, resized_image_size


def _resize_label(label, label_size):
    label = tf.expand_dims(label, 3)
    resized_label = tf.image.resize(
        label, label_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False
    )
    return tf.cast(tf.squeeze(resized_label, 3), tf.int32)


def main(_argv):
    logging.info('Exporting model to: %s', FLAGS.export_path)
    with tf.Graph().as_default():
        image, image_size, resized_image_size = _create_input_tensors()

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=FLAGS.crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride
        )

        if tuple(FLAGS.inference_scales) == (1.0,):
            logging.info('Single-scale inference.')
            predictions = model.predict_labels(image, model_options=model_options, image_pyramid=FLAGS.image_pyramid)
        else:
            logging.info('Multi-scale inference.')
            if FLAGS.quantize_delay_step >= 0:
                raise ValueError('Quantize mode not supported for multi-scale inference.')
            predictions = model.predict_labels_multi_scale(
                image, model_options=model_options,
                eval_scales=FLAGS.inference_scales,
                add_flipped_images=FLAGS.add_flipped_images
            )

        raw_predictions = tf.identity(tf.cast(predictions[common.OUTPUT_TYPE], tf.float32), _RAW_OUTPUT_NAME)
        raw_probabilities = tf.identity(predictions[common.OUTPUT_TYPE + model.PROB_SUFFIX], _RAW_OUTPUT_PROB_NAME)

        # Crop valid region
        semantic_predictions = raw_predictions[:, :resized_image_size[0], :resized_image_size[1]]
        semantic_probabilities = raw_probabilities[:, :resized_image_size[0], :resized_image_size[1]]

        # Resize to original size
        semantic_predictions = _resize_label(semantic_predictions, image_size)
        semantic_predictions = tf.identity(semantic_predictions, name=_OUTPUT_NAME)
        semantic_probabilities = tf.image.resize(semantic_probabilities, image_size, method=tf.image.ResizeMethod.BILINEAR, name=_OUTPUT_PROB_NAME)

        if FLAGS.quantize_delay_step >= 0 and contrib_quantize is not None:
            contrib_quantize.create_eval_graph()

        saver = tf.train.Saver(tf.global_variables())
        os.makedirs(os.path.dirname(FLAGS.export_path), exist_ok=True)

        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        if freeze_graph is None:
            raise RuntimeError('freeze_graph not available in this TF build.')

        freeze_graph.freeze_graph_with_def_protos(
            graph_def,
            saver.as_saver_def(),
            FLAGS.checkpoint_path,
            _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.export_path,
            clear_devices=True,
            initializer_nodes=None
        )

        if FLAGS.save_inference_graph:
            tf.train.write_graph(graph_def, os.path.dirname(FLAGS.export_path), 'inference_graph.pbtxt')


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('export_path')
    app.run(main)
