#ifndef TENSORFLOW_GFTRL_OPS_H_
#define TENSORFLOW_GFTRL_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {
	template <typename Device, typename T>
	struct ApplyGFtrl {
	  void operator()(const Device& d, typename TTypes<T>::Flat var,
					  typename TTypes<T>::Flat accum,
					  typename TTypes<T>::Flat linear,
					  typename TTypes<T>::ConstFlat grad,
					  typename TTypes<T>::ConstScalar lr,
					  typename TTypes<T>::ConstScalar l1,
					  typename TTypes<T>::ConstScalar l2,
					  typename TTypes<T>::ConstScalar lr_power);
	};
	
}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_GFTRL_OPS_H_