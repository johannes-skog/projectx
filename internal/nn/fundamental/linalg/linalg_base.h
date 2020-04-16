#ifndef FUNDAMENTAL_LINALG_LINALG_BASE_H
#define FUNDAMENTAL_LINALG_LINALG_BASE_H

#include <fundamental/shape.h>
#include <fundamental/expression.h>
#include <fundamental/macro.h>

namespace expression{ namespace linalg{

  class TranposeTransformer{

    core::Shape shapeB;

    public:

      TranposeTransformer(const core::Shape& shapeA){
        assert(shapeA.dim == 2);
        index_t _shapeB[SHAPE_MAX_DIM];
        _shapeB[0] = shapeA[1];
        _shapeB[1] = shapeA[0];
        shapeB = core::Shape(_shapeB, 2);
      }

      TENSOR_INLINE const core::Shape& shape(void) const { return shapeB; }

  };


  namespace op{

    template<typename xpu, index_t stream_id, typename Texp, typename T>
    struct Transpose: Exp<xpu, stream_id, Transpose<xpu, stream_id, Texp, T>, T,  type::kRvalue>{

      const TranposeTransformer transformer;

      const Texp A;

      Transpose(const Texp& A): A{A}, transformer{ A.shape() }{}

      TENSOR_INLINE const core::Shape& shape() const {return transformer.shape();}

      TENSOR_INLINE T Eval(index_t i, index_t j) const{ return A.Eval(j, i); }

    };

  }


  template<typename xpu, index_t stream_id, typename Texp, typename T, index_t exp_type>
   op::Transpose<xpu, stream_id, Texp, T> transpose(
                              const Exp<xpu, stream_id, Texp, T, exp_type>& exp){
      return op::Transpose<xpu, stream_id, Texp, T>(exp.self());
   }

}}

#include <fundamental/linalg/matmul.h>

#endif
