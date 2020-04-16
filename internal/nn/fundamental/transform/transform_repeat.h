#ifndef FUNDAMENTAL_TRANSFORM_REPEAT_H
#define FUNDAMENTAL_TRANSFORM_REPEAT_H

#include <fundamental/transform/transform_base.h>
#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/macro.h>

namespace core{

  class RepeatShape{

    Shape shapeB;

    public:

      RepeatShape(const Shape& shapeA, const int n){
        index_t _shape[SHAPE_MAX_DIM];
        _shape[0] = n; // infinite repeat on first dim
        for (int i = 0; i < shapeA.dim; ++i) _shape[i+1] = shapeA[i];
        shapeB = core::Shape(_shape, shapeA.dim + 1);
      }

      TENSOR_INLINE const core::Shape& shape(void) const{ return shapeB; }

  };

}

namespace expression{ namespace transform{

  namespace op{

    template<typename xpu, index_t stream_id, typename E, typename T>
    struct Repeat: public Exp<xpu, stream_id,  Repeat<xpu, stream_id, E, T>, T, type::kRvalue>{

      E e;
      const core::RepeatShape shape_trans;

      Repeat(const int n, const E& e): e{e}, shape_trans{ e.self().shape(), n} {}

      TENSOR_INLINE T Eval(index_t s, index_t i) const{
        return e.Eval(0, i);
      }

      TENSOR_INLINE T BackwardEval(index_t s, index_t i){
        return e.BackwardEval(0, i);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        return e.Backward(0, i, dy);
      }

      TENSOR_INLINE const core::Shape& shape(void) const { return shape_trans.shape(); }

    };

  }

  template<typename xpu, index_t stream_id, typename E, typename T, index_t exp_type>
  op::Repeat<xpu, stream_id, E, T> repeat(const int n,
                                          const Exp<xpu, stream_id, E, T, exp_type>& exp){
    return op::Repeat<xpu, stream_id, E, T>(n, exp.self());
  }

}}

#endif
