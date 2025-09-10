# Lint as: python3
"""
Converts UW-IS / VOC-style data to TFRecord files.
"""

from __future__ import absolute_import, division, print_function
import math
import sys
import os
from pathlib import Path
import build_data
from six.moves import range
import tensorflow as tf
from absl import app, flags, logging

FLAGS = flags.FLAGS

# --- Flags ---
flags.DEFINE_string('image_folder', './data/JPEGImages_livingroom', 'Folder containing images.')
flags.DEFINE_string('semantic_segmentation_folder', './data/foreground_livingroom',
                    'Folder containing segmentation annotations.')
flags.DEFINE_string('list_folder', './data/ImageSets_livingroom', 'Folder containing train/val split text files.')
flags.DEFINE_string('output_dir', './uwis/tfrecord', 'Where to save the generated TFRecord files.')
flags.DEFINE_string('image_format', 'jpg', 'Image file extension (e.g., jpg, png).')
flags.DEFINE_string('label_format', 'png', 'Segmentation label file extension.')
flags.DEFINE_integer('num_shards', 4, 'Number of shards per dataset split.')


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format."""
    dataset_name = Path(dataset_split).stem
    logging.info('Processing %s', dataset_name)

    filenames = [x.strip() for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / FLAGS.num_shards))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_id in range(FLAGS.num_shards):
        output_filename = output_dir / f'{dataset_name}-{shard_id:05d}-of-{FLAGS.num_shards:05d}.tfrecord'
        with tf.io.TFRecordWriter(str(output_filename)) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write(f'\r>> Converting image {i + 1}/{num_images} shard {shard_id}')
                sys.stdout.flush()

                # Read image
                image_path = Path(FLAGS.image_folder) / f'{filenames[i]}.{FLAGS.image_format}'
                image_data = image_path.read_bytes()
                height, width = image_reader.read_image_dims(image_data)

                # Read label
                label_path = Path(FLAGS.semantic_segmentation_folder) / f'{filenames[i]}.{FLAGS.label_format}'
                seg_data = label_path.read_bytes()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)

                if height != seg_height or width != seg_width:
                    raise RuntimeError(f'Shape mismatch between image and label for {filenames[i]}.')

                # Convert to TF example
                example = build_data.image_seg_to_tfexample(image_data, filenames[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(_argv):
    dataset_splits = list(Path(FLAGS.list_folder).glob('*.txt'))
    for dataset_split in dataset_splits:
        _convert_dataset(str(dataset_split))


if __name__ == '__main__':
    flags.mark_flag_as_required('image_folder')
    flags.mark_flag_as_required('semantic_segmentation_folder')
    flags.mark_flag_as_required('list_folder')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
