# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================

r"""Provides DeepLab model definition and helper functions."""

import tensorflow as tf
import tf_slim as slim
from deeplab.core import dense_prediction_cell
from deeplab.core import feature_extractor
from deeplab.core import utils

# Disable eager execution for TF1-style graph
tf.compat.v1.disable_eager_execution()

LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'
META_ARCHITECTURE_SCOPE = 'meta_architecture'

PROB_SUFFIX = '_prob'

_resize_bilinear = utils.resize_bilinear
scale_dimension = utils.scale_dimension
split_separable_conv2d = utils.split_separable_conv2d


def get_extra_layer_scopes(last_layers_contain_logits_only=False):
    if last_layers_contain_logits_only:
        return [LOGITS_SCOPE_NAME]
    else:
        return [
            LOGITS_SCOPE_NAME,
            IMAGE_POOLING_SCOPE,
            ASPP_SCOPE,
            CONCAT_PROJECTION_SCOPE,
            DECODER_SCOPE,
            META_ARCHITECTURE_SCOPE,
        ]


def predict_labels_multi_scale(images, model_options,
                               eval_scales=(1.0,),
                               add_flipped_images=False):
    outputs_to_predictions = {output: [] for output in model_options.outputs_to_num_classes}

    for i, image_scale in enumerate(eval_scales):
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
            outputs_to_scales_to_logits = multi_scale_logits(
                images,
                model_options=model_options,
                image_pyramid=[image_scale],
                is_training=False,
                fine_tune_batch_norm=False)

        if add_flipped_images:
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
                outputs_to_scales_to_logits_reversed = multi_scale_logits(
                    tf.reverse(images, [2]),
                    model_options=model_options,
                    image_pyramid=[image_scale],
                    is_training=False,
                    fine_tune_batch_norm=False)

        for output in sorted(outputs_to_scales_to_logits):
            scales_to_logits = outputs_to_scales_to_logits[output]
            logits = _resize_bilinear(scales_to_logits[MERGED_LOGITS_SCOPE],
                                      tf.shape(images)[1:3],
                                      scales_to_logits[MERGED_LOGITS_SCOPE].dtype)
            outputs_to_predictions[output].append(tf.expand_dims(tf.nn.softmax(logits), 4))

            if add_flipped_images:
                scales_to_logits_reversed = outputs_to_scales_to_logits_reversed[output]
                logits_reversed = _resize_bilinear(
                    tf.reverse(scales_to_logits_reversed[MERGED_LOGITS_SCOPE], [2]),
                    tf.shape(images)[1:3],
                    scales_to_logits_reversed[MERGED_LOGITS_SCOPE].dtype)
                outputs_to_predictions[output].append(tf.expand_dims(tf.nn.softmax(logits_reversed), 4))

    for output in sorted(outputs_to_predictions):
        predictions = outputs_to_predictions[output]
        predictions = tf.reduce_mean(tf.concat(predictions, 4), axis=4)
        outputs_to_predictions[output] = tf.argmax(predictions, 3)
        outputs_to_predictions[output + PROB_SUFFIX] = tf.nn.softmax(predictions)

    return outputs_to_predictions


def predict_labels(images, model_options, image_pyramid=None):
    outputs_to_scales_to_logits = multi_scale_logits(
        images,
        model_options=model_options,
        image_pyramid=image_pyramid,
        is_training=False,
        fine_tune_batch_norm=False)

    predictions = {}
    for output in sorted(outputs_to_scales_to_logits):
        scales_to_logits = outputs_to_scales_to_logits[output]
        logits = scales_to_logits[MERGED_LOGITS_SCOPE]
        if model_options.prediction_with_upsampled_logits:
            logits = _resize_bilinear(logits, tf.shape(images)[1:3], logits.dtype)
            predictions[output] = tf.argmax(logits, 3)
            predictions[output + PROB_SUFFIX] = tf.nn.softmax(logits)
        else:
            argmax_results = tf.argmax(logits, 3)
            argmax_results = tf.image.resize_nearest_neighbor(
                tf.expand_dims(argmax_results, 3),
                tf.shape(images)[1:3],
                align_corners=True,
                name='resize_prediction')
            predictions[output] = tf.squeeze(argmax_results, 3)
            predictions[output + PROB_SUFFIX] = tf.image.resize_bilinear(
                tf.nn.softmax(logits),
                tf.shape(images)[1:3],
                align_corners=True,
                name='resize_prob')
    return predictions


def multi_scale_logits(images, model_options, image_pyramid, weight_decay=0.0001,
                       is_training=False, fine_tune_batch_norm=False,
                       nas_training_hyper_parameters=None):
    if not image_pyramid:
        image_pyramid = [1.0]

    crop_height = model_options.crop_size[0] if model_options.crop_size else tf.shape(images)[1]
    crop_width = model_options.crop_size[1] if model_options.crop_size else tf.shape(images)[2]
    if model_options.image_pooling_crop_size:
        image_pooling_crop_height = model_options.image_pooling_crop_size[0]
        image_pooling_crop_width = model_options.image_pooling_crop_size[1]

    logits_output_stride = min(model_options.decoder_output_stride) if model_options.decoder_output_stride else model_options.output_stride
    logits_height = scale_dimension(crop_height, max(1.0, max(image_pyramid)) / logits_output_stride)
    logits_width = scale_dimension(crop_width, max(1.0, max(image_pyramid)) / logits_output_stride)

    outputs_to_scales_to_logits = {k: {} for k in model_options.outputs_to_num_classes}
    num_channels = images.get_shape().as_list()[-1]

    for image_scale in image_pyramid:
        if image_scale != 1.0:
            scaled_height = scale_dimension(crop_height, image_scale)
            scaled_width = scale_dimension(crop_width, image_scale)
            scaled_crop_size = [scaled_height, scaled_width]
            scaled_images = _resize_bilinear(images, scaled_crop_size, images.dtype)
            if model_options.crop_size:
                scaled_images.set_shape([None, scaled_height, scaled_width, num_channels])
            scaled_image_pooling_crop_size = [scale_dimension(image_pooling_crop_height, image_scale),
                                              scale_dimension(image_pooling_crop_width, image_scale)] if model_options.image_pooling_crop_size else None
        else:
            scaled_crop_size = model_options.crop_size
            scaled_images = images
            scaled_image_pooling_crop_size = model_options.image_pooling_crop_size

        updated_options = model_options._replace(crop_size=scaled_crop_size,
                                                 image_pooling_crop_size=scaled_image_pooling_crop_size)
        outputs_to_logits = _get_logits(
            scaled_images,
            updated_options,
            weight_decay=weight_decay,
            reuse=tf.compat.v1.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            nas_training_hyper_parameters=nas_training_hyper_parameters)

        for output in sorted(outputs_to_logits):
            outputs_to_logits[output] = _resize_bilinear(outputs_to_logits[output],
                                                         [logits_height, logits_width],
                                                         outputs_to_logits[output].dtype)

        if len(image_pyramid) == 1:
            for output in sorted(model_options.outputs_to_num_classes):
                outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
            return outputs_to_scales_to_logits

        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_scales_to_logits[output]['logits_%.2f' % image_scale] = outputs_to_logits[output]

    for output in sorted(model_options.outputs_to_num_classes):
        all_logits = [tf.expand_dims(logits, axis=4) for logits in outputs_to_scales_to_logits[output].values()]
        all_logits = tf.concat(all_logits, 4)
        merge_fn = tf.reduce_max if model_options.merge_method == 'max' else tf.reduce_mean
        outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(all_logits, axis=4)

    return outputs_to_scales_to_logits


def extract_features(images, model_options, weight_decay=0.0001, reuse=None,
                     is_training=False, fine_tune_batch_norm=False,
                     nas_training_hyper_parameters=None):
    features, end_points = feature_extractor.extract_features(
        images,
        output_stride=model_options.output_stride,
        multi_grid=model_options.multi_grid,
        model_variant=model_options.model_variant,
        depth_multiplier=model_options.depth_multiplier,
        divisible_by=model_options.divisible_by,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        preprocessed_images_dtype=model_options.preprocessed_images_dtype,
        fine_tune_batch_norm=fine_tune_batch_norm,
        nas_architecture_options=model_options.nas_architecture_options,
        nas_training_hyper_parameters=nas_training_hyper_parameters,
        use_bounded_activation=model_options.use_bounded_activation
    )

    if not model_options.aspp_with_batch_norm:
        return features, end_points

    if model_options.dense_prediction_cell_config is not None:
        dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
            config=model_options.dense_prediction_cell_config,
            hparams={'conv_rate_multiplier': 16 // model_options.output_stride})
        concat_logits = dense_prediction_layer.build_cell(
            features,
            output_stride=model_options.output_stride,
            crop_size=model_options.crop_size,
            image_pooling_crop_size=model_options.image_pooling_crop_size,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm
        )
        return concat_logits, end_points

    batch_norm_params = utils.get_batch_norm_params(
        decay=0.9997,
        epsilon=1e-5,
        scale=True,
        is_training=(is_training and fine_tune_batch_norm),
        sync_batch_norm_method=model_options.sync_batch_norm_method
    )
    batch_norm = utils.get_batch_norm_fn(model_options.sync_batch_norm_method)
    activation_fn = tf.nn.relu6 if model_options.use_bounded_activation else tf.nn.relu

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=activation_fn,
                        normalizer_fn=batch_norm,
                        padding='SAME',
                        stride=1,
                        reuse=reuse):
        with slim.arg_scope([batch_norm], **batch_norm_params):
            depth = model_options.aspp_convs_filters
            branch_logits = []

            if model_options.add_image_level_feature:
                if model_options.crop_size:
                    image_pooling_crop_size = model_options.image_pooling_crop_size or model_options.crop_size
                    pool_height = scale_dimension(image_pooling_crop_size[0], 1. / model_options.output_stride)
                    pool_width = scale_dimension(image_pooling_crop_size[1], 1. / model_options.output_stride)
                    image_feature = slim.avg_pool2d(features, [pool_height, pool_width],
                                                    model_options.image_pooling_stride, padding='VALID')
                    resize_height = scale_dimension(model_options.crop_size[0], 1. / model_options.output_stride)
                    resize_width = scale_dimension(model_options.crop_size[1], 1. / model_options.output_stride)
                else:
                    pool_height = tf.shape(features)[1]
                    pool_width = tf.shape(features)[2]
                    image_feature = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
                    resize_height = pool_height
                    resize_width = pool_width

                image_feature_activation_fn = tf.nn.relu
                image_feature_normalizer_fn = batch_norm
                if model_options.aspp_with_squeeze_and_excitation:
                    image_feature_activation_fn = tf.nn.sigmoid
                    if model_options.image_se_uses_qsigmoid:
                        image_feature_activation_fn = utils.q_sigmoid
                    image_feature_normalizer_fn = None

                image_feature = slim.conv2d(
                    image_feature, depth, 1,
                    activation_fn=image_feature_activation_fn,
                    normalizer_fn=image_feature_normalizer_fn,
                    scope=IMAGE_POOLING_SCOPE
                )
                image_feature = _resize_bilinear(image_feature, [resize_height, resize_width], image_feature.dtype)
                if not model_options.aspp_with_squeeze_and_excitation:
                    branch_logits.append(image_feature)

            branch_logits.append(slim.conv2d(features, depth, 1, scope=ASPP_SCOPE + str(0)))

            if model_options.atrous_rates:
                for i, rate in enumerate(model_options.atrous_rates, 1):
                    scope = ASPP_SCOPE + str(i)
                    if model_options.aspp_with_separable_conv:
                        aspp_features = split_separable_conv2d(features, filters=depth, rate=rate,
                                                               weight_decay=weight_decay, scope=scope)
                    else:
                        aspp_features = slim.conv2d(features, depth, 3, rate=rate, scope=scope)
                    branch_logits.append(aspp_features)

            concat_logits = tf.concat(branch_logits, 3)
            if model_options.aspp_with_concat_projection:
                concat_logits = slim.conv2d(concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
                concat_logits = slim.dropout(concat_logits, keep_prob=0.9, is_training=is_training)

    return concat_logits, end_points


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    """TF2-compatible get_branch_logits using tf.compat.v1."""
    
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                             'using aspp_with_batch_norm. Gets %d.' % kernel_size)
        atrous_rates = [1]

    with tf.compat.v1.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features], reuse=tf.compat.v1.AUTO_REUSE):
        branch_logits = []
        for i, rate in enumerate(atrous_rates):
            scope = scope_suffix
            if i:
                scope += '_%d' % i
            branch_logits.append(
                slim.conv2d(
                    features,
                    num_classes,
                    kernel_size=kernel_size,
                    rate=rate,
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    scope=scope
                )
            )
        return tf.add_n(branch_logits)


def refine_by_decoder(features,
                      end_points,
                      crop_size,
                      decoder_output_stride,
                      decoder_use_separable_conv=False,
                      decoder_use_sum_merge=False,
                      decoder_filters=256,
                      decoder_output_is_logits=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      use_bounded_activation=False,
                      sync_batch_norm_method='None'):
    """TF2-compatible refine_by_decoder using tf.compat.v1."""
    
    if crop_size is None:
        raise ValueError('crop_size must be provided when using decoder.')

    batch_norm_params = utils.get_batch_norm_params(
        decay=0.9997,
        epsilon=1e-5,
        scale=True,
        is_training=(is_training and fine_tune_batch_norm),
        sync_batch_norm_method=sync_batch_norm_method)
    
    batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
    decoder_depth = decoder_filters
    projected_filters = 48 if not decoder_use_sum_merge else decoder_filters

    if decoder_output_is_logits:
        activation_fn = None
        normalizer_fn = None
        conv2d_kernel = 1
        decoder_use_separable_conv = False
    else:
        activation_fn = tf.nn.relu6 if use_bounded_activation else tf.nn.relu
        normalizer_fn = batch_norm
        conv2d_kernel = 3

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        padding='SAME',
        stride=1,
        reuse=reuse):
        with slim.arg_scope([batch_norm], **batch_norm_params):
            with tf.compat.v1.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features], reuse=tf.compat.v1.AUTO_REUSE):
                decoder_features = features
                decoder_stage = 0
                scope_suffix = ''
                
                for output_stride in decoder_output_stride:
                    feature_list = feature_extractor.networks_to_feature_maps[model_variant][
                        feature_extractor.DECODER_END_POINTS][output_stride]
                    
                    if decoder_stage:
                        scope_suffix = '_{}'.format(decoder_stage)
                    
                    for i, name in enumerate(feature_list):
                        decoder_features_list = [decoder_features]
                        if ('mobilenet' in model_variant or
                            model_variant.startswith('mnas') or
                            model_variant.startswith('nas')):
                            feature_name = name
                        else:
                            feature_name = '{}/{}'.format(feature_extractor.name_scope[model_variant], name)

                        decoder_features_list.append(
                            slim.conv2d(
                                end_points[feature_name],
                                projected_filters,
                                1,
                                scope='feature_projection' + str(i) + scope_suffix
                            )
                        )

                        decoder_height = scale_dimension(crop_size[0], 1.0 / output_stride)
                        decoder_width = scale_dimension(crop_size[1], 1.0 / output_stride)

                        for j, feature in enumerate(decoder_features_list):
                            decoder_features_list[j] = _resize_bilinear(
                                feature, [decoder_height, decoder_width], feature.dtype)
                            h = None if isinstance(decoder_height, tf.Tensor) else decoder_height
                            w = None if isinstance(decoder_width, tf.Tensor) else decoder_width
                            decoder_features_list[j].set_shape([None, h, w, None])

                        if decoder_use_sum_merge:
                            decoder_features = _decoder_with_sum_merge(
                                decoder_features_list,
                                decoder_depth,
                                conv2d_kernel=conv2d_kernel,
                                decoder_use_separable_conv=decoder_use_separable_conv,
                                weight_decay=weight_decay,
                                scope_suffix=scope_suffix
                            )
                        else:
                            if not decoder_use_separable_conv:
                                scope_suffix = str(i) + scope_suffix
                            decoder_features = _decoder_with_concat_merge(
                                decoder_features_list,
                                decoder_depth,
                                decoder_use_separable_conv=decoder_use_separable_conv,
                                weight_decay=weight_decay,
                                scope_suffix=scope_suffix
                            )
                    decoder_stage += 1
                return decoder_features


def _decoder_with_sum_merge(decoder_features_list,
                            decoder_depth,
                            conv2d_kernel=3,
                            decoder_use_separable_conv=True,
                            weight_decay=0.0001,
                            scope_suffix=''):
    if len(decoder_features_list) != 2:
        raise RuntimeError('Expect decoder_features has length 2.')
    if decoder_use_separable_conv:
        decoder_features = split_separable_conv2d(
            decoder_features_list[0],
            filters=decoder_depth,
            rate=1,
            weight_decay=weight_decay,
            scope='decoder_split_sep_conv0'+scope_suffix
        ) + decoder_features_list[1]
    else:
        decoder_features = slim.conv2d(
            decoder_features_list[0],
            decoder_depth,
            conv2d_kernel,
            scope='decoder_conv0'+scope_suffix
        ) + decoder_features_list[1]
    return decoder_features


def _decoder_with_concat_merge(decoder_features_list,
                               decoder_depth,
                               decoder_use_separable_conv=True,
                               weight_decay=0.0001,
                               scope_suffix=''):
    if decoder_use_separable_conv:
        decoder_features = split_separable_conv2d(
            tf.concat(decoder_features_list, 3),
            filters=decoder_depth,
            rate=1,
            weight_decay=weight_decay,
            scope='decoder_conv0'+scope_suffix
        )
        decoder_features = split_separable_conv2d(
            decoder_features,
            filters=decoder_depth,
            rate=1,
            weight_decay=weight_decay,
            scope='decoder_conv1'+scope_suffix
        )
    else:
        decoder_features = slim.repeat(
            tf.concat(decoder_features_list, 3),
            2,
            slim.conv2d,
            decoder_depth,
            3,
            scope='decoder_conv'+scope_suffix
        )
    return decoder_features


def get_branch_logits(features,
                      num_classes,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    if aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            raise ValueError('Kernel size must be 1 when atrous_rates is None or using aspp_with_batch_norm.')
        atrous_rates = [1]

    with tf.compat.v1.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features], reuse=tf.compat.v1.AUTO_REUSE):
        branch_logits = []
        for i, rate in enumerate(atrous_rates):
            scope = scope_suffix + (f'_{i}' if i else '')
            branch_logits.append(
                slim.conv2d(
                    features,
                    num_classes,
                    kernel_size=kernel_size,
                    rate=rate,
                    activation_fn=None,
                    normalizer_fn=None,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    scope=scope
                )
            )
        return tf.add_n(branch_logits)

