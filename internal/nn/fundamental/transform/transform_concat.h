#ifndef FUNDAMENTAL_TRANSFORM_CONCAT_H
#define FUNDAMENTAL_TRANSFORM_CONCAT_H

#include <fundamental/transform/transform_base.h>
#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/macro.h>

namespace core{

  class ConcatShape{

    const Shape shape1;
    const Shape shape2;

    index_t axis;

    public:

      Shape new_shape;

      ConcatShape(const Shape& shape1, const Shape& shape2, index_t axis):
                  shape1{shape1}, shape2{shape2},
                  axis{axis}{

         assert(shape1.dim == shape2.dim && shape1.dim > axis);

         index_t _new_shape[SHAPE_MAX_DIM];

         for (int i = 0; i < shape1.dim; ++i){
           if (i != axis){
              assert(shape1[i] == shape2[i]);
              _new_shape[i] = shape1[i];
           }
         }

         _new_shape[axis] = shape1[axis] + shape2[axis];
         this->new_shape = core::Shape(_new_shape, shape1.dim);

      }

      TENSOR_INLINE const core::Shape& shape(void) const { return new_shape; }

      TENSOR_INLINE void transform(index_t s, index_t i, index_t& s_n,
                                   index_t& i_n, index_t& order)  const {
        order = 0;
        index_t t_shape[SHAPE_MAX_DIM];

        this->new_shape.deduce_shape(s, i, t_shape);
        if (t_shape[axis] >= shape1[axis]){
          order = 1;
          t_shape[axis] = t_shape[axis] -  shape1[axis];
          shape2.index(s_n, i_n, t_shape);
        } else{
          order = 0;
          shape1.index(s_n, i_n, t_shape);
        }
      }

  };

}

namespace expression{ namespace transform{

  namespace op{

    template<typename xpu, index_t stream_id, typename E1, typename E2, typename T>
    struct Concat: public Exp<xpu, stream_id,
                    Concat<xpu, stream_id, E1, E2, T>, T,  type::kRvalue>{

      core::ConcatShape transformer;

      E1 e1;
      E2 e2;

      Concat(const E1& e1, const E2& e2, index_t axis): e1{e1},
                       e2{e2}, transformer(e1.shape(), e2.shape(), axis){}

      TENSOR_INLINE const core::Shape& shape() const {return transformer.shape();}

      TENSOR_INLINE T Eval(index_t s, index_t i) const{
        index_t s_n, i_n, order;
        transformer.transform(s, i, s_n, i_n, order);
        if (order == 0) return e1.Eval(s_n, i_n);
        return e2.Eval(s_n, i_n);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        index_t s_n, i_n, order;
        transformer.transform(s, i, s_n, i_n, order);
        if (order == 0) e1.Backward(s_n, i_n, dy);
        else e2.Backward(s_n, i_n, dy);
      }

    };

  }

 template<typename xpu, index_t stream_id, typename E1, typename E2,
         typename T, index_t exp_type1, index_t exp_type2>
  op::Concat<xpu, stream_id, E1, E2, T> concat(
                             const Exp<xpu, stream_id, E1, T, exp_type1>& exp1,
                             const Exp<xpu, stream_id, E2, T, exp_type2>& exp2,
                             index_t axis){
     return op::Concat<xpu, stream_id, E1, E2, T>(exp1.self(), exp2.self(), axis);
  }

  }

}


#endif
