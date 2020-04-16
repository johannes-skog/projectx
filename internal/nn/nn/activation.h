#ifndef N_ACTIVATION_H
#define N_ACTIVATION_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

#include <nn/activation_funcs.h>
#include <nn/layer.h>

namespace nn{ namespace activation{

template<typename Afunc, typename xpu, index_t stream_id, typename T>
std::vector< core::Tensor<xpu, stream_id, T> >
forward_inplace(std::vector< core::Tensor<xpu, stream_id, T> >& xv){

  DEBUG_ASSERT(xv.size() == 1);

  auto& x = xv[0];

  x = expression::F<typename Afunc::Forward<xpu>>(x);

  return {x};

}

template<typename Afunc, typename xpu, index_t stream_id, typename T>
std::vector< core::Tensor<xpu, stream_id, T> >
forward(std::vector< core::Tensor<xpu, stream_id, T> >& xv){

  DEBUG_ASSERT(xv.size() == 1);

  auto& x = xv[0];

  core::Tensor<xpu, stream_id, T> y(x.shape()); y.allocate();

  y = expression::F<typename Afunc::Forward<xpu>>(x);

  return {y};

}

template<typename Afunc, typename xpu, index_t stream_id, typename T>
std::vector< core::Tensor<xpu, stream_id, T> >
backward_inplace(std::vector< core::Tensor<xpu, stream_id, T> >& xv,
         std::vector< core::Tensor<xpu, stream_id, T> >& dyv){

  DEBUG_ASSERT(xv.size() == 1 && dyv.size() == 1);

  auto& x = xv[0];
  auto& dy = dyv[0];

  dy = dy * expression::F<typename Afunc::Backward<xpu>>(x);

  return {dy};

}

template<typename Afunc, typename xpu, index_t stream_id, typename T>
std::vector< core::Tensor<xpu, stream_id, T> >
backward(std::vector< core::Tensor<xpu, stream_id, T> >& xv,
         std::vector< core::Tensor<xpu, stream_id, T> >& dyv){

  DEBUG_ASSERT(xv.size() == 1 && dyv.size() == 1);

  auto& x = xv[0];
  auto& dy = dyv[0];

  core::Tensor<xpu, stream_id, T> dx(x.shape()); dx.allocate();

  dx = dy * expression::F<typename Afunc::Backward<xpu>>(x);

  return {dx};

}


}

}










#endif
