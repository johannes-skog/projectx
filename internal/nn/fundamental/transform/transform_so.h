#ifndef FUNDAMENTAL_TRANSFORM_SO_H
#define FUNDAMENTAL_TRANSFORM_SO_H

#include <fundamental/transform/transform_base.h>
#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

namespace core{

    class SOTransformer{

      public:

        Shape org_shape;
        Shape new_shape;

        index_t offset;
        index_t stride;

        SOTransformer(const Shape& t_shape, index_t offset, index_t stride,
                    index_t n): org_shape{t_shape}, offset{offset}, stride{stride}{

          DEBUG_ASSERT((org_shape.size()-offset) >= n);
          DEBUG_ASSERT( (n % stride)  == 0);

          index_t N_new = n / stride;
          this->new_shape = core::Shape(N_new);

        }

        TENSOR_INLINE const core::Shape& shape() const { return new_shape; }

        TENSOR_INLINE void transform(index_t s, index_t i, index_t& s_n) const{
          s_n = s * stride  + offset; // Stride is always one for one dim
          // we must transfer it back to the two
        }

    };

}

namespace expression{ namespace transform{

  namespace op{

    template<typename xpu, index_t stream_id, typename E, typename T>
    struct Stride: public Exp<xpu, stream_id, Stride<xpu, stream_id, E, T>, T, type::kLvalue>{

      E e;

      core::SOTransformer shape_trans;

      Stride(const E& e, index_t offset, index_t stride, index_t n): e{e},
             shape_trans{e.self().shape(), offset, stride, n} {}

      TENSOR_INLINE T Eval(index_t s, index_t i) const {
        index_t s_n;
        this->shape_trans.transform(s, i, s_n);
        return e.Eval(0, s_n);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        index_t s_n;
        this->shape_trans.transform(s, i, s_n);
        e.Backward(0, s_n, dy);
      }

      TENSOR_INLINE void Set(index_t s, index_t i, T v){
        index_t s_n;
        this->shape_trans.transform(s, i, s_n);
        e.Set(0, s_n, v);
      }

      TENSOR_INLINE const core::Shape& shape() const { return shape_trans.shape(); }

    };

  }

  template<typename xpu, index_t stream_id, typename E, typename T, index_t exp_type>
  TENSOR_INLINE_HOST op::Stride<xpu, stream_id, E, T>
          stride(Exp<xpu, stream_id, E, T, exp_type>& exp,
                 index_t offset, index_t stride, index_t n){
    return op::Stride<xpu, stream_id, E, T>(exp.self(), offset, stride, n);
  }


}}

#endif
