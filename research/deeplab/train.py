from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import contextlib
import tensorflow as tf

# Disable TF2 behavior for TF1 code
tf.compat.v1.disable_v2_behavior()
tf = tf.compat.v1

# Optional contrib imports
try:
    from tensorflow.contrib import quantize as contrib_quantize
except Exception:
    contrib_quantize = None

try:
    from tensorflow.contrib import tfprof as contrib_tfprof
except Exception:
    contrib_tfprof = None

try:
    import tf_slim as slim
except Exception:
    try:
        slim = tf.contrib.slim
    except Exception:
        slim = None

from deeplab import common, model
from deeplab.datasets import data_generator
from deeplab.utils import train_utils
from deployment import model_deploy

import absl.logging as logging
from absl import app, flags

# Dummy ProfileContext if tfprof not available
@contextlib.contextmanager
def _dummy_profile_context(enabled=False, profile_dir=None):
    yield

ProfileContext = contrib_tfprof.ProfileContext if contrib_tfprof else _dummy_profile_context

FLAGS = flags.FLAGS

# --- Define flags ---
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')
flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')
flags.DEFINE_integer('startup_delay_steps', 15, 'Steps between replicas startup.')
flags.DEFINE_integer('num_ps_tasks', 0, 'Number of parameter servers.')
flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')
flags.DEFINE_integer('task', 0, 'Task ID.')
flags.DEFINE_string('train_logdir', None, 'Checkpoint and log directory.')
flags.DEFINE_integer('log_steps', 10, 'Display logging every n steps.')
flags.DEFINE_integer('save_interval_secs', 1200, 'Save model interval (secs).')
flags.DEFINE_integer('save_summaries_secs', 600, 'Summary interval (secs).')
flags.DEFINE_boolean('save_summaries_images', False, 'Save images to summary.')
flags.DEFINE_string('profile_logdir', None, 'Profile directory.')
flags.DEFINE_enum('optimizer', 'momentum', ['momentum', 'adam'], 'Optimizer type.')
flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'], 'Learning rate policy.')
flags.DEFINE_float('base_learning_rate', .0001, 'Base learning rate.')
flags.DEFINE_integer('training_number_of_steps', 30000, 'Number of training steps.')
flags.DEFINE_integer('train_batch_size', 8, 'Batch size.')
flags.DEFINE_string('dataset', 'pascal_voc_seg', 'Dataset name.')
flags.DEFINE_string('train_split', 'train', 'Dataset split for training.')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory.')
flags.DEFINE_string('tf_initial_checkpoint', None, 'Initial checkpoint.')
flags.DEFINE_boolean('initialize_last_layer', True, 'Initialize last layer.')

# --- Build model function ---
def _build_deeplab(iterator, outputs_to_num_classes, ignore_label):
    samples = iterator.get_next()
    samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=[int(sz) for sz in FLAGS.train_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride
    )

    outputs_to_scales_to_logits = model.multi_scale_logits(
        samples[common.IMAGE],
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm
    )

    output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
    output_type_dict[model.MERGED_LOGITS_SCOPE] = tf.identity(
        output_type_dict[model.MERGED_LOGITS_SCOPE], name=common.OUTPUT_TYPE
    )

    for output, num_classes in six.iteritems(outputs_to_num_classes):
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples[common.LABEL],
            num_classes,
            ignore_label,
            loss_weight=model_options.label_weights,
            upsample_logits=FLAGS.upsample_logits
        )

# --- Main training loop ---
def main(_argv):
    logging.set_verbosity(logging.INFO)
    config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.num_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks
    )

    assert FLAGS.train_batch_size % config.num_clones == 0
    clone_batch_size = FLAGS.train_batch_size // config.num_clones

    tf.io.gfile.makedirs(FLAGS.train_logdir)
    logging.info('Training on %s set', FLAGS.train_split)

    with tf.Graph().as_default() as graph:
        with tf.device(config.inputs_device()):
            dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=FLAGS.train_split,
                dataset_dir=FLAGS.dataset_dir,
                batch_size=clone_batch_size,
                crop_size=[int(sz) for sz in FLAGS.train_crop_size],
                is_training=True
            )

        with tf.device(config.variables_device()):
            global_step = tf.train.get_or_create_global_step()
            model_fn = _build_deeplab
            model_args = (dataset.get_one_shot_iterator(), {common.OUTPUT_TYPE: dataset.num_of_classes}, dataset.ignore_label)
            clones = model_deploy.create_clones(config, model_fn, args=model_args)
            first_clone_scope = config.clone_scope(0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        total_loss, grads_and_vars = model_deploy.optimize_clones(clones, tf.train.MomentumOptimizer(0.001, 0.9))
        grad_updates = tf.train.Optimizer(0.001).apply_gradients(grads_and_vars, global_step=global_step)
        update_ops.append(grad_updates)

        train_tensor = tf.identity(total_loss, name='train_op')

        session_config = tf.ConfigProto(allow_soft_placement=True)
        slim.learning.train(train_tensor, logdir=FLAGS.train_logdir, number_of_steps=FLAGS.training_number_of_steps, session_config=session_config)

if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('dataset_dir')
    app.run(main)
