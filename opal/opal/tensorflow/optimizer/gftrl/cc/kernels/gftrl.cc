/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#define EIGEN_USE_THREADS
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "gftrl.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

namespace {
template <class T>
inline T sgn(const T x) {
  T zero(0);
  T one(1);
  return (x == zero ? zero : (x < zero ? -one : one));
}
}  // namespace

namespace functor {

template <typename T>
struct ApplyGFtrl<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::Matrix linear,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
      const int64 input_dim = var.dimension(0);
      const int64 output_dim = var.dimension(1);
      T alpha_inv = static_cast<T>(1.0) / lr();
      T beta = static_cast<T>(1.0);
      
      for (auto i = 0; i < input_dim; ++i) {
        T z_norm = static_cast<T>(0.0);
        for (auto j = 0; j < output_dim; ++j) {
          auto new_n = accum(i, j) + grad(i, j) * grad(i, j);
          auto sigma = (Eigen::numext::sqrt(new_n) - Eigen::numext::sqrt(accum(i, j))) * alpha_inv;
          auto new_z = linear(i, j) + grad(i, j) - sigma * var(i, j);
          z_norm = z_norm + new_z * new_z;
        }
      
        z_norm = Eigen::numext::sqrt(z_norm);
        for (auto j = 0; j < output_dim; ++j) {
            auto new_n = accum(i, j) + grad(i, j) * grad(i, j);
            auto sigma = (Eigen::numext::sqrt(new_n) - Eigen::numext::sqrt(accum(i, j))) * alpha_inv;
            accum(i, j) = new_n;
            linear(i, j) += grad(i, j) - sigma * var(i, j);
            
            // update the weight
            T sqrt_output_dim = static_cast<T>(Eigen::numext::sqrt(output_dim));
            if (z_norm > l1() * sqrt_output_dim) {
              var(i, j) = linear(i, j) * (l1() * sqrt_output_dim / z_norm - static_cast<T>(1)) /
                  ((beta + Eigen::numext::sqrt(new_n)) * alpha_inv + l2());
            } else {
              var(i, j) = static_cast<T>(0.0);
            }
        }
      }
  }
};

}

template <typename Device, typename T>
class ApplyGFtrlOp : public OpKernel {
 public:
  explicit ApplyGFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
      const bool sparse = false;
      auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
          ctx, use_exclusive_lock_, sparse, {0, 1, 2});
      
      Tensor var;
      OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                              ctx, 0, use_exclusive_lock_, sparse, &var));
      Tensor accum;
      OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                              ctx, 1, use_exclusive_lock_, sparse, &accum));
      Tensor linear;
      OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                              ctx, 2, use_exclusive_lock_, sparse, &linear));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = 7;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a"
                                        " non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() == static_cast<T>(-0.5),
                errors::InvalidArgument("lr_power must be"
                                        " -0.5: ",
                                        lr_power.shape().DebugString()));
    

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyGFtrl<Device, T>()(device, var.matrix<T>(), accum.matrix<T>(),
                                    linear.matrix<T>(), grad.matrix<T>(),
                                    lr.scalar<T>(), l1.scalar<T>(),
                                    l2.scalar<T>(), lr_power.scalar<T>());

    if (ctx->input_dtype(0) != DT_RESOURCE) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyGFtrl").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGFtrlOp<D##Device, T>);                                      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ResourceApplyGFtrl")                                    \
          .HostMemory("var")                                       \
          .HostMemory("accum")                                     \
          .HostMemory("linear")                                    \
          .Device(DEVICE_##D)                                      \
          .TypeConstraint<T>("T"),                                 \
      ApplyGFtrlOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tindex>
class SparseApplyGFtrlOp : public OpKernel {
 public:
  explicit SparseApplyGFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor linear;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &linear));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    lr.scalar<T>()() > static_cast<T>(0),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    l1.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    l2.scalar<T>()() >= static_cast<T>(0),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    lr_power.scalar<T>()() <= static_cast<T>(0),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    
    const Tindex N = indices.dim_size(0);
    
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "per_grad must be the same size as per_indices in the first dimension."));
  
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));
  
    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        typename TTypes<T>::Matrix var_flat = var.matrix<T>();
        typename TTypes<T>::Matrix accum_flat = accum.matrix<T>();
        typename TTypes<T>::Matrix linear_flat = linear.matrix<T>();
        typename TTypes<T>::ConstMatrix grad_flat = grad.matrix<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        //T lr_power_scalar = lr_power.scalar<T>()();         
        T beta = static_cast<T>(1.0);
        T sqrt_output_dim = static_cast<T>(Eigen::numext::sqrt(inner_dim));
  
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T z_norm = static_cast<T>(0);
          
          for (int j = 0; j < inner_dim; j++) {
              auto new_n = accum_flat(index, j) + grad_flat(i, j) * grad_flat(i, j);
              auto sigma = (Eigen::numext::sqrt(new_n) - Eigen::numext::sqrt(accum_flat(index, j))) / lr_scalar;
              auto new_z = linear_flat(index, j) + grad_flat(i, j) - sigma * var_flat(index, j);
              z_norm += new_z * new_z;
          }

          z_norm = Eigen::numext::sqrt(z_norm);
          for (int j = 0; j < inner_dim; j++) {
            auto new_n = accum_flat(index, j) + grad_flat(i, j) * grad_flat(i, j);
            auto sigma = (Eigen::numext::sqrt(new_n) - Eigen::numext::sqrt(accum_flat(index, j))) / lr_scalar;
            accum_flat(index, j) = new_n;
            linear_flat(index, j) += grad_flat(i, j) - sigma * var_flat(index, j);
  
            // update the weight
            if (z_norm > l1_scalar * sqrt_output_dim) {
              var_flat(index, j) = linear_flat(index, j) * ((l1_scalar * sqrt_output_dim) / z_norm - static_cast<T>(1)) /
                ((beta + Eigen::numext::sqrt(new_n)) / lr_scalar + l2_scalar);
            } else {
              var_flat(index, j) = static_cast<T>(0);
            }
          }
        }
      } else {
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        //T lr_power_scalar = lr_power.scalar<T>()();         
        T beta = static_cast<T>(1.0);
        T sqrt_output_dim = static_cast<T>(Eigen::numext::sqrt(inner_dim));
        
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto linear_flat = linear.flat<T>();
        auto grad_flat = grad.flat<T>();
        const Tindex first_dim_size = accum_flat.size();
  
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                           " in indices is out of range")));
          T& a = accum_flat(index);
          T& l = linear_flat(index);
          T& v = var_flat(index);
          T g = grad_flat(i);

          T z_norm = static_cast<T>(0.0);
          auto new_n = a + g * g;
          auto sigma = (Eigen::numext::sqrt(new_n) - Eigen::numext::sqrt(a)) / lr_scalar;
          auto new_z = l + g - sigma * v;
          z_norm = sgn(new_z) * new_z;
          a = new_n;
          l = new_z;
        
          // update the weight
          if (z_norm > l1_scalar * sqrt_output_dim) {
            v = l * (l1_scalar * sqrt_output_dim / z_norm - static_cast<T>(1)) /
              ((beta + Eigen::numext::sqrt(new_n)) / lr_scalar + l2_scalar);
          } else {
            v = static_cast<T>(0.0);
          }         
        }
      }
    }

    if (ctx->input_dtype(0) != DT_RESOURCE) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    }
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseApplyGFtrl")                                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyGFtrlOp<CPUDevice, T, Tindices>);                             \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ResourceSparseApplyGFtrl")                                         \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyGFtrlOp<CPUDevice, T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}
