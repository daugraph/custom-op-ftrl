# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for Ftrl operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import ftrl
import gftrl
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import gradient_descent


def test_gftrl_dim2_l1l2_128_sparse():
    for dtype in [dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable(tf.random_normal(shape=[10,128], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        var1 = variables.Variable(tf.random_normal(shape=[10,128], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(128), shape=[1, 128], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 128]))
        grads1 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(128), shape=[1, 128], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 128]))

        opt = gftrl.GFtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=1,
            l2_regularization_strength=2.0)
        #update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update = opt.apply_gradients(zip([grads0], [var0]))
        variables.global_variables_initializer().run()

        #v0_val, v1_val = sess.run([var0, var1])
        #self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        #self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)
        v0_val = sess.run(var0)
        print("v0_val", v0_val[0])

        # Run 10 steps FTRL
        for _ in range(2):
          update.run()
          #v0_val, v1_val = sess.run([var0, var1])
          v0_val = sess.run(var0)
          print("v0_val", v0_val[0])
          #print("v1_val", v1_val[1])


def test_gftrl_dim2_l1l2_128_sparse_const():
    for dtype in [dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable(tf.random_normal(shape=[10,4], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        var1 = variables.Variable(tf.random_normal(shape=[10,4], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(4), shape=[1, 4], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 4]))
        grads1 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(4), shape=[1, 4], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 4]))

        opt = gftrl.GFtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=1,
            l2_regularization_strength=2.0)
        #update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update = opt.apply_gradients(zip([grads0], [var0]))
        variables.global_variables_initializer().run()

        #v0_val, v1_val = sess.run([var0, var1])
        #self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        #self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)
        v0_val = sess.run(var0)
        print("v0_val", v0_val[0])

        # Run 10 steps FTRL
        for _ in range(2):
          update.run()
          #v0_val, v1_val = sess.run([var0, var1])
          v0_val = sess.run(var0)
          print("v0_val", v0_val[0])
          #print("v1_val", v1_val[1])


def test_ftrl_dim2_l1l2_128_sparse():
    for dtype in [dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable(tf.random_normal(shape=[10,128], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        var1 = variables.Variable(tf.random_normal(shape=[10,128], mean=0, stddev=1, dtype=dtype), dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(128), shape=[1, 128], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 128]))
        grads1 = ops.IndexedSlices(
            constant_op.constant(np.random.randn(128), shape=[1, 128], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 128]))

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        #self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        #self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

        # Run 10 steps FTRL
        for _ in range(50000):
          update.run()

        v0_val, v1_val = sess.run([var0, var1])
        #print("v0_val", v0_val)
        #print("v1_val", v1_val)


'''
v0_val [[1.  1.1]
 [2.  2.1]]
e 0.610418 0.610418 
c 0.330686 0.539663
x 0 0
y  4.1972 4.31798
v0_val [0. 0.]
e 3.27773 3.27773
c -0.198771 -0.342775
x -0.198771 -0.342775
y  4.2582 4.43716
v0_val [-0.04667969 -0.07725089]
'''
def test_gftrl_l1l2_sparse_one():
    for dtype in [dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable([[1.0, 1.1], [2.0, 2.1]], dtype=dtype)
        var1 = variables.Variable([[4.0, 4.1], [3.0, 3.1]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.5, 0.9], shape=[1, 2], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.02, 0.04], shape=[1, 2], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))

        opt = gftrl.GFtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=1,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0], [var0]))
        variables.global_variables_initializer().run()

        v0_val = sess.run([var0])
        print("v0_val", v0_val[0])

        # Run 10 steps FTRL
        for _ in range(2):
          update.run()
          v0_val= sess.run(var0)
          print("v0_val", v0_val[0])


def test_gftrl_l1l2_sparse():
    for dtype in [dtypes.half, dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable([[1.0, 1.1], [2.0, 2.1]], dtype=dtype)
        var1 = variables.Variable([[4.0, 4.1], [3.0, 3.1]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1, 0.2], shape=[1, 2], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.02, 0.04], shape=[1, 2], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))

        opt = gftrl.GFtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        #self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        #self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = sess.run([var0, var1])
        print("v0_val", v0_val)
        print("v1_val", v1_val)

def test_ftrl_l1l2_sparse():
    for dtype in [dtypes.half, dtypes.float32]:
      with tf.Session() as sess:
        var0 = variables.Variable([[1.0, 1.1], [2.0, 2.1]], dtype=dtype)
        var1 = variables.Variable([[4.0, 4.1], [3.0, 3.1]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1, 0.2], shape=[1, 2], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.02, 0.04], shape=[1, 2], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        v0_val, v1_val = sess.run([var0, var1])
        #self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        #self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = sess.run([var0, var1])
        print("v0_val", v0_val)
        print("v1_val", v1_val)


class FtrlOptimizerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testGFtrlWithL1_L2_Sparse(self):
    """Tests the new FTRL op with support for l2 shrinkage on sparse grads."""
    for dtype in [dtypes.float32]:
      with self.cached_session() as sess:
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[4.0], [3.0]], dtype=dtype)
        grads0 = ops.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]), constant_op.constant([2, 1]))
        grads1 = ops.IndexedSlices(
            constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]), constant_op.constant([2, 1]))

        opt = gftrl.GFtrlOptimizer(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
        self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        print("v0_val", v0_val)
        print("v1_val", v1_val)

  @test_util.run_deprecated_v1
  def testFtrlWithL1_L2(self):
    for dtype in [dtypes.half, dtypes.float32]:
      with self.cached_session() as sess:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = ftrl.Ftrl(
            3.0,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run 10 steps FTRL
        for _ in range(10):
          update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.24059935, -0.46829352]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.02406147, -0.04830509]), v1_val)

  def applyOptimizer(self, opt, dtype, steps=5, is_sparse=False):
    if is_sparse:
      var0 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      var1 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      grads0 = ops.IndexedSlices(
          constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
          constant_op.constant([0]), constant_op.constant([2, 1]))
      grads1 = ops.IndexedSlices(
          constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
          constant_op.constant([1]), constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([0.0, 0.0], dtype=dtype)
      var1 = variables.Variable([0.0, 0.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    variables.global_variables_initializer().run()

    sess = ops.get_default_session()
    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v0_val)
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v1_val)
    else:
      self.assertAllCloseAccordingToType([0.0, 0.0], v0_val)
      self.assertAllCloseAccordingToType([0.0, 0.0], v1_val)

    # Run Ftrl for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val



if __name__ == "__main__":
    #test_gftrl_dim2_l1l2_128_sparse()
    test_gftrl_l1l2_sparse_one()
    #test_ftrl_dim2_l1l2_128_sparse()
    #test_gftrl_l1l2_sparse()
    #test_ftrl_l1l2_sparse()
  

