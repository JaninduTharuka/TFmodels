# Lint as: python3
"""
Common utilities for converting images and segmentation labels to TFRecord.
"""

import collections
import six
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'], 'Image format.')
flags.DEFINE_enum('label_format', 'png', ['png'], 'Segmentation label format.')

# Map from image format to expected TF decode function
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}


class ImageReader:
    """Helper class for reading and decoding images."""

    def __init__(self, image_format='jpeg', channels=3):
        self._image_format = image_format
        self._channels = channels

    def read_image_dims(self, image_data):
        """Returns height and width of the image."""
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes image bytes to a TF Tensor."""
        if self._image_format in ('jpeg', 'jpg'):
            image = tf.io.decode_jpeg(image_data, channels=self._channels)
        elif self._image_format == 'png':
            image = tf.io.decode_png(image_data, channels=self._channels)
        else:
            raise ValueError(f"Unsupported image format: {self._image_format}")

        # Convert to numpy array
        image = tf.convert_to_tensor(image)
        image = image.numpy() if tf.executing_eagerly() else image
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError("Image channels not supported.")
        return image


def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list."""
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes."""
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, seg_data):
    """Creates a tf.train.Example from image and segmentation data."""
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_list_feature(image_data),
        'image/filename': _bytes_list_feature(filename),
        'image/format': _bytes_list_feature(_IMAGE_FORMAT_MAP[FLAGS.image_format]),
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/channels': _int64_list_feature(3),
        'image/segmentation/class/encoded': _bytes_list_feature(seg_data),
        'image/segmentation/class/format': _bytes_list_feature(FLAGS.label_format),
    }))
