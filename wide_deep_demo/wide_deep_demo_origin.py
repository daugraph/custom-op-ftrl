# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core run logic for TensorFlow Wide & Deep Tutorial using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import json

from absl import app as absl_app
from absl import flags

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}

TRAINING_FILE = 'adult.data'
EVAL_FILE = 'adult.test'

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_HASH_BUCKET_SIZE = 128

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns(embedding_map_dict=None):
  """Builds a set of wide and deep feature columns."""
  # Continuous variable columns
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

  workclass = tf.feature_column.categorical_column_with_hash_bucket(
      'workclass',  hash_bucket_size=_HASH_BUCKET_SIZE)

  # To show an example of hashing:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [
      education, marital_status, relationship, workclass, occupation,
      age_buckets,
  ]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'],
          hash_bucket_size=_HASH_BUCKET_SIZE),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [ # 51 total
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(education), # 16
      tf.feature_column.indicator_column(marital_status), # 7
      tf.feature_column.indicator_column(relationship), # 6
      # To show an example of embedding
      tf.feature_column.embedding_column(workclass, dimension=8,
                                         combiner='sqrtn',
                                         initializer=tf.uniform_unit_scaling_initializer()), # 8 
      tf.feature_column.embedding_column(occupation, dimension=8,
                                         combiner='sqrtn',
                                         initializer=tf.uniform_unit_scaling_initializer()), # 8 
  ]

  return wide_columns, deep_columns


def train_input_fn(flags_obj):
  train_file = os.path.join(flags_obj.data_dir, TRAINING_FILE)
  return lambda: input_fn(
      train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

def eval_input_fn(flags_obj):
  test_file = os.path.join(flags_obj.data_dir, EVAL_FILE)
  return lambda: input_fn(test_file, 1, False, flags_obj.batch_size)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run census_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    classes = tf.equal(labels, '>50K')  # binary classification
    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=2)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=batch_size*10)
  return dataset


def export_model(model, export_dir, model_column_fn):
  """Export to SavedModel format.

  Args:
    model: Estimator object
    export_dir: directory to export the model.
    model_column_fn: Function to generate model feature columns.
  """
  wide_columns, deep_columns = model_column_fn()
  columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_savedmodel(export_dir, example_input_fn,  as_text=False)


def define_wide_deep_flags():
  """Add supervised learning flags, as well as wide-deep model type."""
  flags_core.define_base()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name="model_type", short_name="mt", default="wide_deep",
      enum_values=['wide', 'deep', 'wide_deep'],
      help="Select model topology.")
  flags.DEFINE_integer(name='max_step',  default=2000, help='Max training step')


def define_census_flags():
  define_wide_deep_flags()
  flags_core.set_defaults(data_dir='/tmp/census_data',
                          model_dir='/tmp/census_model',
                          train_epochs=40,
                          epochs_between_evals=10,
                          batch_size=10)


def build_estimator(model_dir, model_type, model_column_fn, embedding_map_dict):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn(embedding_map_dict)
  hidden_units = [128, 64, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  config = tf.ConfigProto(device_count={'GPU':0},
                          intra_op_parallelism_threads=8,
                          inter_op_parallelism_threads=8)
  run_config = tf.estimator.RunConfig().replace(session_config = config,
                                                log_step_count_steps=100,
                                                save_summary_steps=100,
                                                save_checkpoints_steps=1000)
  # original estimator of DNNLinearCombinedClassifier
  # provided by tensorflow
  return tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=hidden_units,
    config=run_config,
    input_layer_partitioner=tf.min_max_variable_partitioner(
        max_partitions=2,
        min_slice_size=64 << 20))


def get_non_zero_embedding_map_dict(model_dir):
    file_name = os.path.join(model_dir, "feature_spec.json")
    if not tf.gfile.Exists(file_name):
        return dict()
    with tf.gfile.GFile(file_name, 'rb') as f:
        feature_dict = json.loads(f.read())
        print("=== non zero embedding map ===")
        print(feature_dict)
        return feature_dict


def train_and_evaluate(name, train_input_fn, eval_input_fn, model_column_fn,
                       build_estimator_fn, flags_obj):
  """Define training loop."""
  embedding_map_dict = get_non_zero_embedding_map_dict(flags_obj.model_dir)
  model = build_estimator_fn(
      model_dir=flags_obj.model_dir, model_type=flags_obj.model_type,
      model_column_fn=model_column_fn, embedding_map_dict=embedding_map_dict)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn(flags_obj), max_steps=flags_obj.max_step, hooks=[])

  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn(flags_obj), steps=200, hooks=eval_hooks)

  tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
  export_model(model, flags_obj.export_dir, model_column_fn)


def run_census(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """

  train_and_evaluate(name="Census Income", 
                     train_input_fn=train_input_fn,
                     eval_input_fn=eval_input_fn,
                     model_column_fn=build_model_columns,
                     build_estimator_fn=build_estimator,
                     flags_obj=flags_obj)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_census(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_census_flags()
  absl_app.run(main)
