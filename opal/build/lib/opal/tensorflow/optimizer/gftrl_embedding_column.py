
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math

import numpy as np
import six


from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine.base_layer import Layer
# TODO(b/118385027): Dependency on keras can be problematic if Keras moves out
# of the main repo.
from tensorflow.python.keras import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.feature_column import feature_column_v2 as fcv2

def embedding_with_map_column(categorical_column,
                              dimension,
                              combiner='mean',
                              initializer=None,
                              ckpt_to_load_from=None,
                              tensor_name_in_ckpt=None,
                              max_norm=None,
                              trainable=True,
                              non_zero_embedding_size=None):
  """`DenseColumn` that converts from sparse, categorical input.

  Use this when your inputs are sparse, but you want to convert them to a dense
  representation (e.g., to feed to a DNN).

  Inputs must be a `CategoricalColumn` created by any of the
  `categorical_column_*` function. Here is an example of using
  `embedding_column` with `DNNClassifier`:

  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [embedding_column(video_id, 9),...]

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `embedding_column` with model_fn:

  ```python
  def model_fn(features, ...):
    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    columns = [embedding_column(video_id, 9),...]
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_column: A `CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    `DenseColumn` that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: If eager execution is enabled.
  """
  if (dimension is None) or (dimension < 1):
    raise ValueError('Invalid dimension {}.'.format(dimension))
  if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
    raise ValueError('Must specify both `ckpt_to_load_from` and '
                     '`tensor_name_in_ckpt` or none of them.')

  if (initializer is not None) and (not callable(initializer)):
    raise ValueError('initializer must be callable if specified. '
                     'Embedding of column_name: {}'.format(
                         categorical_column.name))
  if initializer is None:
    initializer = init_ops.truncated_normal_initializer(
        mean=0.0, stddev=1 / math.sqrt(dimension))
  
  if non_zero_embedding_size:
    embedding_column = EmbeddingWithMapColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable)
    embedding_column.set_new_embedding_size(non_zero_embedding_size)
  else:
    embedding_column = fcv2.EmbeddingColumn(
      categorical_column=categorical_column,
      dimension=dimension,
      combiner=combiner,
      initializer=initializer,
      ckpt_to_load_from=ckpt_to_load_from,
      tensor_name_in_ckpt=tensor_name_in_ckpt,
      max_norm=max_norm,
      trainable=trainable)
  return embedding_column


class EmbeddingWithMapColumn(fcv2.EmbeddingColumn):

  def set_new_embedding_size(self, size):
    self.non_zero_embedding_size = size

  def get_embedding_maps_var(self):
    return self.embedding_state_manager.get_variable(self, 'embedding_weights_maps')

  def create_state(self, state_manager):
    """Creates the embedding lookup variable."""
    embedding_shape = (self.non_zero_embedding_size, self.dimension)  # pylint: disable=protected-access
    state_manager.create_variable(
        self,
        name='embedding_weights',
        shape=embedding_shape,
        dtype=dtypes.float32,
        trainable=self.trainable,
        # TODO(rohanj): Make this True when b/118500434 is fixed.
        use_resource=False,
        initializer=self.initializer)
    embedding_map_shape = (self.categorical_column.num_buckets,)  # pylint: disable=protected-access
    state_manager.create_variable(
        self,
        name='embedding_weights_maps',
        shape=embedding_map_shape,
        dtype=dtypes.int64,
        trainable=self.trainable,
        # TODO(rohanj): Make this True when b/118500434 is fixed.
        use_resource=False,
        initializer=init_ops.zeros_initializer(dtypes.int64))
    self.embedding_state_manager = state_manager

  def _get_dense_tensor_internal_helper(self, sparse_tensors,
                                        embedding_weights):
    sparse_ids = sparse_tensors.id_tensor
    sparse_weights = sparse_tensors.weight_tensor

    if self.ckpt_to_load_from is not None:
      to_restore = embedding_weights
      if isinstance(to_restore, variables.PartitionedVariable):
        to_restore = to_restore._get_variable_list()  # pylint: disable=protected-access
      checkpoint_utils.init_from_checkpoint(self.ckpt_to_load_from, {
          self.tensor_name_in_ckpt: to_restore
      })
     
    embedding_map = self.get_embedding_maps_var()
    new_embedding_ids = array_ops.gather(embedding_map, sparse_ids.values)
    new_sparse_ids = sparse_tensor_lib.SparseTensor(
        indices=sparse_ids.indices,
        values=new_embedding_ids,
        dense_shape=sparse_ids.dense_shape)

    # Return embedding lookup result.
    return embedding_ops.safe_embedding_lookup_sparse(
        embedding_weights=embedding_weights,
        sparse_ids=new_sparse_ids,
        sparse_weights=sparse_weights,
        combiner=self.combiner,
        name='%s_weights' % self.name,
        max_norm=self.max_norm)

  def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections,
                                     trainable):
    raise Exception("Undefined _old_get_dense_tensor_internal")

