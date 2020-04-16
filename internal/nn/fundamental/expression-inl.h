#ifndef FUNDAMENTAL_FUNDAMENTAL_INL_H
#define FUNDAMENTAL_FUNDAMENTAL_INL_H

#include <fundamental/expression.h>
#include <fundamental/tensor.h>
#include <functional>

namespace expression{

  template<typename xpu, index_t stream_id, typename EL, typename ER,
           typename T, int tl, int tr>
  TENSOR_INLINE_HOST Binary<xpu, stream_id, typename primitives::Primitives<xpu>::plus,
                            EL, ER, T>
  operator+(const Exp<xpu, stream_id, EL, T, tl> &el,
            const Exp<xpu, stream_id, ER, T, tr> &er){
    return MakeExp<xpu, stream_id, typename primitives::Primitives<xpu>::plus>(el, er);
  }

  template<typename xpu, index_t stream_id, typename EL, typename ER,
           typename T, int tl, int tr>
  TENSOR_INLINE_HOST Binary<xpu, stream_id, typename primitives::Primitives<xpu>::minus,
                            EL, ER, T>
  operator-(const Exp<xpu, stream_id, EL, T, tl> &el,
            const Exp<xpu, stream_id, ER, T, tr> &er){
    return MakeExp<xpu, stream_id, typename primitives::Primitives<xpu>::minus>(el, er);
  }

  template<typename xpu, index_t stream_id, typename EL, typename ER,
           typename T, int tl, int tr>
  TENSOR_INLINE_HOST Binary<xpu, stream_id, typename primitives::Primitives<xpu>::mul,
                            EL, ER, T>
  operator*(const Exp<xpu, stream_id, EL, T, tl> &el,
            const Exp<xpu, stream_id, ER, T, tr> &er){
    return MakeExp<xpu, stream_id, typename primitives::Primitives<xpu>::mul>(el, er);
  }

  template<typename xpu, index_t stream_id, typename EL, typename ER,
           typename T, int tl, int tr>
  TENSOR_INLINE_HOST Binary<xpu, stream_id, typename primitives::Primitives<xpu>::div,
                            EL, ER, T>
  operator/(const Exp<xpu, stream_id, EL, T, tl> &el,
            const Exp<xpu, stream_id, ER, T, tr> &er) {
    return MakeExp<xpu, stream_id, typename primitives::Primitives<xpu>::div>(el, er);
  }

  template<typename xpu, index_t stream_id, typename SubType, typename T,
           index_t exp_type>
  TENSOR_INLINE_HOST auto Exp<xpu, stream_id, SubType, T, exp_type>::eval(){
    core::Tensor<xpu, stream_id, T> tensor(this->self().shape());
    tensor.allocate();
    Exceturer<xpu>::excetute(*this, tensor);
    return tensor;
  }

  template<typename Esrc, typename Edst, index_t stream_id, typename T,
           index_t exp_type>
  TENSOR_INLINE_HOST void Exceturer<cpu>::excetute(
    const Exp<cpu, stream_id, Esrc, T, exp_type>& src_exp,
    const Exp<cpu, stream_id, Edst, T, type::kLvalue>& dst_exp){

    DEBUG_ASSERT(src_exp.self().shape() == dst_exp.self().shape());

    auto task = [](Esrc src, Edst dst){
      #pragma omp simd
      for (int s = 0; s < src.shape().N; ++s){
        for (int i = 0; i < src.shape().stride; ++i){
          dst.Set(s, i, src.Eval(s, i));
        }
      }
    };

    STREAM_FORWARD(cpu, stream_id).put(task, src_exp.self(), dst_exp.self());

  }

  template<typename Esrc, typename Edst, index_t stream_id, typename T,
           index_t exp_typeSrc, index_t exp_typeDst>
  TENSOR_INLINE_HOST void Exceturer<cpu>::backward(
    const Exp<cpu, stream_id, Esrc, T, exp_typeSrc>& src_exp,
    const Exp<cpu, stream_id, Edst, T, exp_typeDst>& dst_exp){

    DEBUG_ASSERT(src_exp.self().shape() == dst_exp.self().shape());

    auto task = [](Esrc src, Edst dst){
      #pragma omp simd
      for (int s = 0; s < src.shape().N; ++s){
        for (int i = 0; i < src.shape().stride; ++i){
          dst.Backward(s, i, src.BackwardEval(s, i));
        }
      }
    };

    STREAM_BACKWARD(cpu, stream_id).put(task, src_exp.self(), dst_exp.self());

  }


}

#endif
