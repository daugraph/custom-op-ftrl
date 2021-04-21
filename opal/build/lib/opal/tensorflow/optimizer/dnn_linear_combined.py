import six
import tensorflow as tf
from tensorflow.python.ops.losses import losses
# import tensorflow_estimator.python.estimator.canned.dnn_linear_combined as wd
from tensorflow_estimator.python.estimator.canned import head as head_lib

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import estimator_export
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import dnn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import linear
# from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from  tensorflow_estimator.python.estimator.canned.dnn_linear_combined \
    import _validate_feature_columns, _check_no_sync_replicas_optimizer, _linear_learning_rate, \
    _add_layer_summary, _DNN_LEARNING_RATE, _LINEAR_LEARNING_RATE

# from jarvis.tensorflow.id_column import JarvisEstimatorScope
from opal.tensorflow.optimizer import optimizers


def _dnn_linear_combined_model_fn(features,
                                  labels,
                                  mode,
                                  head,
                                  linear_feature_columns=None,
                                  linear_optimizer='Ftrl',
                                  dnn_feature_columns=None,
                                  dnn_optimizer='Adagrad',
                                  dnn_hidden_units=None,
                                  dnn_activation_fn=nn.relu,
                                  dnn_dropout=None,
                                  embedding_optimizer='GFtrl',
                                  input_layer_partitioner=None,
                                  config=None,
                                  batch_norm=False,
                                  linear_sparse_combiner='sum'):
  """Deep Neural Net and Linear combined model_fn.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `Head` instance.
    linear_feature_columns: An iterable containing all the feature columns used
      by the Linear model.
    linear_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the Linear model. Defaults to the Ftrl
      optimizer.
    dnn_feature_columns: An iterable containing all the feature columns used by
      the DNN model.
    dnn_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the DNN model. Defaults to the Adagrad
      optimizer.
    dnn_hidden_units: List of hidden units per DNN layer.
    dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
      will use `tf.nn.relu`.
    dnn_dropout: When not `None`, the probability we will drop out a given DNN
      coordinate.
    embedding_optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training the embedding matrix. Defaults to the GFtrl
      optimizer.
    input_layer_partitioner: Partitioner for input layer.
    config: `RunConfig` object to configure the runtime settings.
    batch_norm: Whether to use batch normalization after each hidden layer.
    linear_sparse_combiner: A string specifying how to reduce the linear model
      if a categorical column is multivalent.  One of "mean", "sqrtn", and
      "sum".
  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If both `linear_feature_columns` and `dnn_features_columns`
      are empty at the same time, or `input_layer_partitioner` is missing,
      or features has the wrong type.
  """
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))
  if not linear_feature_columns and not dnn_feature_columns:
    raise ValueError(
        'Either linear_feature_columns or dnn_feature_columns must be defined.')

  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Build DNN Logits.
  dnn_parent_scope = 'dnn'
  if not dnn_feature_columns:
    dnn_logits = None
  else:
    dnn_optimizer = optimizers.get_optimizer_instance(
        dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
    _check_no_sync_replicas_optimizer(dnn_optimizer)
    embedding_optimizer = optimizers.get_optimizer_instance(
        embedding_optimizer, learning_rate=_DNN_LEARNING_RATE)

    if not dnn_hidden_units:
      raise ValueError(
          'dnn_hidden_units must be defined when dnn_feature_columns is '
          'specified.')
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner) as scope:
      dnn_absolute_scope = scope.name
      dnn_logit_fn = dnn.dnn_logit_fn_builder(
          units=head.logits_dimension,
          hidden_units=dnn_hidden_units,
          feature_columns=dnn_feature_columns,
          activation_fn=dnn_activation_fn,
          dropout=dnn_dropout,
          batch_norm=batch_norm,
          input_layer_partitioner=input_layer_partitioner)
      dnn_logits = dnn_logit_fn(features=features, mode=mode)

  linear_parent_scope = 'linear'
  if not linear_feature_columns:
    linear_logits = None
  else:
    linear_optimizer = optimizers.get_optimizer_instance(
        linear_optimizer,
        learning_rate=_linear_learning_rate(len(linear_feature_columns)))
    _check_no_sync_replicas_optimizer(linear_optimizer)
    with variable_scope.variable_scope(
        linear_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner) as scope:
      linear_absolute_scope = scope.name
      logit_fn = linear.linear_logit_fn_builder(
          units=head.logits_dimension,
          feature_columns=linear_feature_columns,
          sparse_combiner=linear_sparse_combiner)
      linear_logits = logit_fn(features=features)
      _add_layer_summary(linear_logits, scope.name)

  # Combine logits and build full model.
  if dnn_logits is not None and linear_logits is not None:
    logits = dnn_logits + linear_logits
  elif dnn_logits is not None:
    logits = dnn_logits
  else:
    logits = linear_logits

  def _train_op_fn(loss):
    """Returns the op to optimize the loss."""
    train_ops = []
    global_step = training_util.get_global_step()

    if dnn_logits is not None:
      var_list_whole = ops.get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES,
          scope=dnn_absolute_scope)
      var_list_embedding = ops.get_collection(
          ops.GraphKeys.TRAINABLE_VARIABLES,
          scope=dnn_absolute_scope + '/input_from_feature_columns')
      var_list_dnn = list(set(var_list_whole) - set(var_list_embedding))

      train_ops.append(
          embedding_optimizer.minimize(
              loss,
              var_list=var_list_embedding))
      train_ops.append(
          dnn_optimizer.minimize(
              loss,
              var_list=var_list_dnn))
    if linear_logits is not None:
      train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=linear_absolute_scope)))

    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      return state_ops.assign_add(global_step, 1).op

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)


class DNNLinearCombinedClassifier(tf.estimator.Estimator):

    def __init__(self,
                 model_dir=None,
                 linear_feature_columns=None,
                 linear_optimizer='Ftrl',
                 dnn_feature_columns=None,
                 dnn_optimizer='Adagrad',
                 dnn_hidden_units=None,
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout=None,
                 embedding_optimizer='GFtrl',
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None,
                 warm_start_from=None,
                 loss_reduction=losses.Reduction.SUM,
                 batch_norm=False,
                 linear_sparse_combiner='sum'):
        self._feature_columns =_validate_feature_columns(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns)

        head = head_lib._binary_logistic_or_multi_class_head(  # pylint: disable=protected-access
            n_classes, weight_column, label_vocabulary, loss_reduction)

        def _model_fn(features, labels, mode, config):
            """Call the _dnn_linear_combined_model_fn."""
            # with JarvisEstimatorScope(mode):
            return _dnn_linear_combined_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                dnn_feature_columns=dnn_feature_columns,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=dnn_activation_fn,
                dnn_dropout=dnn_dropout,
                embedding_optimizer=embedding_optimizer,
                input_layer_partitioner=input_layer_partitioner,
                config=config,
                batch_norm=batch_norm,
                linear_sparse_combiner=linear_sparse_combiner)

        super(DNNLinearCombinedClassifier, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_from)
