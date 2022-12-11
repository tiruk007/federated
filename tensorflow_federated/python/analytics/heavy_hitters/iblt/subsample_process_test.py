# Copyright 2022, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for subsample_process.py."""
import collections
from typing import Union
import numpy as np

import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import subsample_process

DATA = [
    (['seattle', 'hello', 'world'], [[4], [5], [10]]),
    (['hi', 'seattle'], [[2], [-5]]),
    (['good', 'morning', 'hi', 'bye'], [[3], [1], [2], [3]]),
    (['new', 'york'], [[0], [0]]),
]

THRESHOLD = 4


def _get_count_from_dataset(dataset: tf.data.Dataset, key: str) -> int:
  for x in dataset:
    if x[iblt_factory.DATASET_KEY] == key:
      return x[iblt_factory.DATASET_VALUE][0].numpy()
  return 0  # Return 0 if x is not in the dataset.


def _generate_client_local_data(
    client_data: tuple[list[Union[str, bytes]], list[list[int]]]
) -> tf.data.Dataset:

  input_strings, string_values = client_data
  client_dict = collections.OrderedDict([
      (iblt_factory.DATASET_KEY, tf.constant(input_strings, dtype=tf.string)),
      (iblt_factory.DATASET_VALUE, tf.constant(string_values, dtype=tf.int64))
  ])
  return tf.data.Dataset.from_tensor_slices(client_dict)


DATA_ALL_ABOVE_THRESHOLD = _generate_client_local_data(DATA[0])
DATA_WITH_NEGATIVE = _generate_client_local_data(DATA[1])
DATA_ALL_IN_BETWEEN = _generate_client_local_data(DATA[2])
DATA_ALL_ZERO = _generate_client_local_data(DATA[3])


class SubsampleProcessTest(tf.test.TestCase):

  def test_threshold_at_least_one(self):
    with self.assertRaisesRegex(ValueError, 'Threshold must be at least 1!'):
      subsample_process.ThresholdSamplingProcess(init_param=0)
    with self.assertRaisesRegex(ValueError, 'Threshold must be at least 1!'):
      subsample_process.ThresholdSamplingProcess(init_param=-1)
    # Should not raise
    subsample_process.ThresholdSamplingProcess(init_param=THRESHOLD)

  def test_if_adaptive(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=2)
    self.assertEqual(threshold_sampling.if_adaptive(), False)

  def test_tuning_not_implemented(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=2)
    with self.assertRaisesRegex(NotImplementedError,
                                'Not an adaptive process!'):
      threshold_sampling.update(sample_param_old=2)

  def test_get_init(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD)
    self.assertEqual(threshold_sampling.get_init(), THRESHOLD)

  def test_negative_value(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD)
    with self.assertRaisesRegex(
        ValueError, 'Current implementation only supports positive values!'):
      threshold_sampling.subsample_fn(DATA_WITH_NEGATIVE)

  def test_all_above_threshold(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD)
    self.assertEqual(
        list(threshold_sampling.subsample_fn(DATA_ALL_ABOVE_THRESHOLD)),
        list(DATA_ALL_ABOVE_THRESHOLD))

  def test_all_zero(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD)
    self.assertEmpty(list(threshold_sampling.subsample_fn(DATA_ALL_ZERO)))

  def test_sampling_in_between(self):
    threshold_sampling = subsample_process.ThresholdSamplingProcess(
        init_param=THRESHOLD)
    rep = 100
    strings = ['good', 'morning', 'hi', 'bye']
    freqs = np.array([3, 1, 2, 3])
    counts = np.zeros(len(strings))
    for i in range(rep):
      tf.random.set_seed(i)
      sampled_dataset = threshold_sampling.subsample_fn(DATA_ALL_IN_BETWEEN)
      for j, _ in enumerate(strings):
        counts[j] += _get_count_from_dataset(sampled_dataset, strings[j])
    self.assertAllClose(counts / rep, freqs, atol=0.5)


if __name__ == '__main__':
  tf.test.main()
