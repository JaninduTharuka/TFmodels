# Copyright 2018 The TensorFlow Authors
# Licensed under the Apache License, Version 2.0
# ==============================================================================

"""Provides flags that are common to scripts (TF2 version)."""

import collections
import copy
import json
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

# ------------------ Flags for input preprocessing ------------------
flags.DEFINE_integer('min_resize_value', None, 'Desired size of the smaller image side.')
flags.DEFINE_integer('max_resize_value', None, 'Maximum allowed size of the larger image side.')
flags.DEFINE_integer('resize_factor', None, 'Resized dimensions are multiple of factor plus one.')
flags.DEFINE_boolean('keep_aspect_ratio', True, 'Keep aspect ratio after resizing or not.')

# ------------------ Model dependent flags ------------------
flags.DEFINE_integer('logits_kernel_size', 1, 'Kernel size for logits conv.')

flags.DEFINE_string('model_variant', 'mobilenet_v2', 'DeepLab model variant.')
flags.DEFINE_multi_float('image_pyramid', None, 'Input scales for multi-scale feature extraction.')
flags.DEFINE_boolean('add_image_level_feature', True, 'Add image level feature.')

flags.DEFINE_list('image_pooling_crop_size', None,
                  'Image pooling crop size [height, width] for ASPP.')
flags.DEFINE_list('image_pooling_stride', '1,1',
                  'Image pooling stride [height, width] for ASPP image pooling.')

flags.DEFINE_boolean('aspp_with_batch_norm', True, 'Use batch norm in ASPP.')
flags.DEFINE_boolean('aspp_with_separable_conv', True, 'Use separable conv for ASPP.')
flags.DEFINE_multi_integer('multi_grid', None, 'Atrous rates hierarchy for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for MobileNet.')
flags.DEFINE_integer('divisible_by', None, 'Ensure layer channels divisible by this value.')

flags.DEFINE_list('decoder_output_stride', None,
                  'Output stride of low-level features at each network level.')
flags.DEFINE_boolean('decoder_use_separable_conv', True, 'Use separable conv for decoder.')
flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'], 'Scheme to merge multi scale features.')
flags.DEFINE_boolean('prediction_with_upsampled_logits', True,
                     'Upsample logits before softmax or not.')

flags.DEFINE_string('dense_prediction_cell_json', '', 'JSON file for dense prediction cell.')

flags.DEFINE_integer('nas_stem_output_num_conv_filters', 20,
                     'Number of filters of NAS stem output tensor.')
flags.DEFINE_bool('nas_use_classification_head', False, 'Use classification head for NAS.')
flags.DEFINE_bool('nas_remove_os32_stride', False, 'Remove stride in OS32 branch.')

flags.DEFINE_bool('use_bounded_activation', False, 'Use bounded activations for quantization.')
flags.DEFINE_boolean('aspp_with_concat_projection', True, 'ASPP with concat projection.')
flags.DEFINE_boolean('aspp_with_squeeze_and_excitation', False, 'ASPP with squeeze and excitation.')
flags.DEFINE_integer('aspp_convs_filters', 256, 'ASPP convolution filters.')
flags.DEFINE_boolean('decoder_use_sum_merge', False, 'Decoder uses sum merge.')
flags.DEFINE_integer('decoder_filters', 256, 'Decoder filters.')
flags.DEFINE_boolean('decoder_output_is_logits', False, 'Use decoder output as logits.')
flags.DEFINE_boolean('image_se_uses_qsigmoid', False, 'Use q-sigmoid.')

flags.DEFINE_multi_float('label_weights', None, 'List of label weights.')
flags.DEFINE_float('batch_norm_decay', 0.9997, 'Batchnorm decay.')

# ------------------ Constants ------------------
OUTPUT_TYPE = 'semantic'
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'
TEST_SET = 'test'

# ------------------ Model Options ------------------
class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_architecture_options',
        'use_bounded_activation',
        'aspp_with_concat_projection',
        'aspp_with_squeeze_and_excitation',
        'aspp_convs_filters',
        'decoder_use_sum_merge',
        'decoder_filters',
        'decoder_output_is_logits',
        'image_se_uses_qsigmoid',
        'label_weights',
        'sync_batch_norm_method',
        'batch_norm_decay',
    ])):
    """Immutable class to hold model options."""

    __slots__ = ()

    def __new__(cls,
                outputs_to_num_classes,
                crop_size=None,
                atrous_rates=None,
                output_stride=8,
                preprocessed_images_dtype=tf.float32):
        """Constructor to set default values."""

        dense_prediction_cell_config = None
        if FLAGS.dense_prediction_cell_json:
            with tf.io.gfile.GFile(FLAGS.dense_prediction_cell_json, 'r') as f:
                dense_prediction_cell_config = json.load(f)

        decoder_output_stride = None
        if FLAGS.decoder_output_stride:
            decoder_output_stride = [int(x) for x in FLAGS.decoder_output_stride]
            if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
                raise ValueError('Decoder output stride must be sorted descending.')

        image_pooling_crop_size = None
        if FLAGS.image_pooling_crop_size:
            image_pooling_crop_size = [int(x) for x in FLAGS.image_pooling_crop_size]

        image_pooling_stride = [1, 1]
        if FLAGS.image_pooling_stride:
            image_pooling_stride = [int(x) for x in FLAGS.image_pooling_stride]

        label_weights = FLAGS.label_weights
        if label_weights is None:
            label_weights = 1.0

        nas_architecture_options = {
            'nas_stem_output_num_conv_filters': FLAGS.nas_stem_output_num_conv_filters,
            'nas_use_classification_head': FLAGS.nas_use_classification_head,
            'nas_remove_os32_stride': FLAGS.nas_remove_os32_stride,
        }

        return super(ModelOptions, cls).__new__(
            cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
            preprocessed_images_dtype, FLAGS.merge_method, FLAGS.add_image_level_feature,
            image_pooling_crop_size, image_pooling_stride, FLAGS.aspp_with_batch_norm,
            FLAGS.aspp_with_separable_conv, FLAGS.multi_grid, decoder_output_stride,
            FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size, FLAGS.model_variant,
            FLAGS.depth_multiplier, FLAGS.divisible_by, FLAGS.prediction_with_upsampled_logits,
            dense_prediction_cell_config, nas_architecture_options, FLAGS.use_bounded_activation,
            FLAGS.aspp_with_concat_projection, FLAGS.aspp_with_squeeze_and_excitation,
            FLAGS.aspp_convs_filters, FLAGS.decoder_use_sum_merge, FLAGS.decoder_filters,
            FLAGS.decoder_output_is_logits, FLAGS.image_se_uses_qsigmoid, label_weights,
            'None', FLAGS.batch_norm_decay)

    def __deepcopy__(self, memo):
        return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                            self.crop_size,
                            self.atrous_rates,
                            self.output_stride,
                            self.preprocessed_images_dtype)
