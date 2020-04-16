#ifndef FUNDAMENTAL_LINALG_MATMUL_H
#define FUNDAMENTAL_LINALG_MATMUL_H

#include <fundamental/shape.h>
#include <fundamental/expression.h>
#include <fundamental/macro.h>

namespace expression{ namespace linalg{

  class MatmulTransformer{

    core::Shape shapeC;

    public:

      MatmulTransformer(const core::Shape& shapeA, const core::Shape& shapeB){
        assert(shapeA.dim == 2 && shapeB.dim == 2);
        assert(shapeA[1] == shapeB[0]);
        index_t _shapeC[SHAPE_MAX_DIM];
        _shapeC[0] = shapeA[0];  _shapeC[1] = shapeB[1];
        shapeC = core::Shape(_shapeC, 2);
      }

      TENSOR_INLINE const core::Shape& shape(void) const { return shapeC; }

  };


  namespace op{

    template<typename xpu, index_t stream_id, typename Texp1, typename Texp2, typename T>
    struct Matmul: Exp<xpu, stream_id, Matmul<xpu, stream_id, Texp1, Texp2, T>, T,  type::kRvalue>{

      MatmulTransformer transformer;

      Texp1 A;
      Texp2 B;

      index_t nA, nB, mA, mB;

      Matmul(const Texp1& A, const Texp2& B): A{A},
             B{B}, transformer{A.shape(), B.shape()},
             nA{A.shape()[0]}, nB{B.shape()[0]},
             mA{A.shape()[1]}, mB{B.shape()[1]} {}

      TENSOR_INLINE const core::Shape& shape() const { return transformer.shape(); }

      TENSOR_INLINE T Eval(index_t i, index_t j) const{
        T v = T{0};
        for (int k = 0; k < mA ; ++k){
          v += A.Eval(i, k) * B.Eval(k, j);
        }
        return v;
      }

    };

  }

  template<typename xpu, index_t stream_id, typename Texp1, typename Texp2,
          typename T, index_t exp_type1, index_t exp_type2>
   op::Matmul<xpu, stream_id, Texp1, Texp2, T> matmul(
                              const Exp<xpu, stream_id, Texp1, T, exp_type1>& exp1,
                              const Exp<xpu, stream_id, Texp2, T, exp_type2>& exp2){
      return op::Matmul<xpu, stream_id, Texp1, Texp2, T>(exp1.self(), exp2.self());
   }


}}


#endif
