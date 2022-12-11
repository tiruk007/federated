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
"""Subsampling and tuning logics for adaptive SecAggIBLT."""

import abc
import collections
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory


class SubsampleProcess(abc.ABC):
  """Describes the API for subsample tuning."""

  @abc.abstractmethod
  def if_adaptive(self):
    """Return whether the process is adative."""

  @abc.abstractmethod
  def get_init(self):
    """Returns the initial sampling parameter."""

  @abc.abstractmethod
  def update(self, sample_param_old: float, measurements: tf.Tensor):
    """Update the sample parameter based on the return of an IBLT round."""

  @abc.abstractmethod
  def subsample_fn(self, client_data: tf.data.Dataset, sample_param: float):
    """Performs subsampling at clients.

    Args:
      client_data: a `tf.data.Dataset` that represents client dataset.
      sample_param: sampling parameter to use.

    Returns:
      subsampled client dataset with the same format as client_data.
    """


class ThresholdSamplingProcess(SubsampleProcess):
  """Implements threshold sampling without tuning."""

  def __init__(self, init_param: float):
    """Initialize the internal parameters.

    Args:
      init_param: initial sampling prameter.
    """
    if init_param < 1:
      raise ValueError('Threshold must be at least 1!')
    else:
      self.init_param = init_param

  def if_adaptive(self):
    return False

  def get_init(self):
    return self.init_param

  def update(self,
             sample_param_old: float,
             measurements: Optional[tf.Tensor] = None):
    raise NotImplementedError('Not an adaptive process!')

  def subsample_fn(self,
                   client_data: tf.data.Dataset,
                   sample_param: Optional[float] = None):
    """Performs subsampling at clients.

    Args:
      client_data: a `tf.data.Dataset` that yields `OrderedDict`. In each
        `OrderedDict` there are two key, value pairs: `DATASET_KEY`: A
        `tf.string` representing a string in the dataset. `DATASET_VALUE`: A
        rank 1 `tf.Tensor` with `dtype` `tf.int64` representing the value
        associate with the string.
      sample_param: subsampling parameter (threshold) to use. If set to None,
        use self.init_param instead.

    Returns:
      A `tf.data.Dataset` that yields the subsampled client data (same format as
      client_data).

    Raises:
      ValueError if client data has negative counts.
    """

    if sample_param is None:
      sample_param = self.init_param

    for element in client_data:
      if element[iblt_factory.DATASET_VALUE] < 0:
        raise ValueError(
            'Current implementation only supports positive values!')

    @tf.function
    def threshold_sampling(element):
      count = element[iblt_factory.DATASET_VALUE]
      if count >= sample_param:
        return element
      else:
        temp = tf.random.uniform(
            shape=(), minval=0, maxval=sample_param, dtype=tf.int64)
        if count > temp:
          return collections.OrderedDict(
              zip([iblt_factory.DATASET_KEY, iblt_factory.DATASET_VALUE], [
                  element[iblt_factory.DATASET_KEY],
                  tf.cast([sample_param], dtype=tf.int64)
              ]))
        else:
          return collections.OrderedDict(
              zip([iblt_factory.DATASET_KEY, iblt_factory.DATASET_VALUE], [
                  element[iblt_factory.DATASET_KEY],
                  tf.constant([0], dtype=tf.int64)
              ]))

    subsampled_client_data = client_data.map(threshold_sampling)
    return subsampled_client_data.filter(
        lambda x: x[iblt_factory.DATASET_VALUE][0] > 0)
