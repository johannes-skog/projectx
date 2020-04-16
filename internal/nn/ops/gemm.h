#ifndef OPS_GEMM_H
#define OPS_GEMM_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/transform/transform_base.h>
#include <fundamental/macro.h>

#include <nn/layer.h>
#include <nn/container.h>

#include <algorithm> // std::swap

#include <fundamental/linalg/linalg_base.h>

#ifdef TENSOR_USE_CUBLAS
    #include <cublas_v2.h>
#endif

namespace operators{

  template<index_t stream_id>
  TENSOR_INLINE_HOST void gemm(core::Tensor<gpu, stream_id, float>& A,
                               core::Tensor<gpu, stream_id, float>& B,
                               core::Tensor<gpu, stream_id, float>& C,
                               float alpha, float beta, bool TA, bool TB){

      #ifdef TENSOR_USE_CUBLAS

        auto cublasHandle = STREAM(gpu, stream_id).cublasHandle();

        /* We want to calculate y = x * w^T
        /* cublas accept only column major, meening, to cublas each
        /* matrix is transposed y^T, x^T , w^T
        /* So, instead of calculating y = x * w^T
        /* we calculate y^T = w * x^T
        /* remembering that cuclas will interpete all matricies as y^T, x^T , w^T
        */

        float* ptrA = A.ptr();
        float* ptrB = B.ptr();
        float* ptrC = C.ptr();

        const core::Shape* shapeA = &A.shape();
        const core::Shape* shapeB = &B.shape();
        const core::Shape* shapeC = &C.shape();

        std::swap(ptrA,ptrB);
        std::swap(shapeA, shapeB);
        std::swap(TA, TB);

        cublasOperation_t CU_TA =  TA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t CU_TB =  TB ? CUBLAS_OP_T : CUBLAS_OP_N;

        /*
        /* A = w [k, m] -> according cublas [m, k] -> transpose op -> [k, m]
        /* B = x [n, m] -> according cublas [m, n]
        /* C = y [n, k] -> according cublas [k, n]
        */

        /* the number of rows  of the  matrix op( A ) and of the  matrix  C*/
        int k = TA ? (*shapeA)[0] : (*shapeA)[1];
        ASSERT_ERROR(k == (*shapeC)[1], "The number of rows of op(A) and C are not the same");
        /* the number of columns of the matrix op( A ) and the number of rows of the matrix op( B ) */
        int m = TA ? (*shapeA)[1] : (*shapeA)[0];
        /* the number of columns of the matrix op( B ) and the number of columns of the matrix C */
        int n = TB ? (*shapeB)[1] : (*shapeB)[0];
        ASSERT_ERROR(n == (*shapeC)[0], "The number of columns of op(B) and rows of C are not the same");

        CUDBLAS_CHECK_ERROR_ENFORCE(
                  cublasSgemm(cublasHandle,
                  CU_TA /* The transpose operation is selected for A */,
                  CU_TB /* The non-transpose operation is selected */,
                  k,  /* the number of rows of the  matrix op( A ) and of the  matrix  C*/
                  n,  /* the number of columns of the matrix op( B ) and the number of columns of the matrix C */
                  m,  /* the number of columns of the matrix op( A ) and the number of rows of the matrix op( B ) */
                  &alpha, /* alpha */
                  ptrA, /* A */
                  (*shapeA)[1], /* First dimension of A without any transform */
                  ptrB, /* B */
                  (*shapeB)[1], /* First dimension of B without any transform */
                  &beta, /* beta */
                  ptrC, /* C */
                  (*shapeC)[1])); /* First dimension of C */

      #else

        core::Scalar<gpu> _alpha(alpha);
        core::Scalar<gpu> _beta(beta);

        if (TA && TB)
          C = expression::linalg::matmul(expression::linalg::transpose(A),
                                         expression::linalg::transpose(B)) + _beta*C;
        else if(TA)
          C = expression::linalg::matmul(expression::linalg::transpose(A),
                                         B) + _beta*C;
        else if(TB)
          C = expression::linalg::matmul(A, expression::linalg::transpose(B))
                                         + _beta*C;
        else
          C = expression::linalg::matmul(A, B) + _beta*C;

      #endif

  }

  template<typename xpu, index_t stream_id, typename T>
  TENSOR_INLINE_HOST void gemm(core::Tensor<xpu, stream_id, T>& A,
                               core::Tensor<xpu, stream_id, T>& B,
                               core::Tensor<xpu, stream_id, T>& C,
                               T alpha, T beta, bool TA, bool TB){

        // TODO, include openBLAS

        core::Scalar<xpu, T, stream_id> _alpha(alpha);
        core::Scalar<xpu, T, stream_id> _beta(beta);

        if (TA && TB)
          C = expression::linalg::matmul(expression::linalg::transpose(A),
                                         expression::linalg::transpose(B)) + _beta*C;
        else if(TA)
          C = expression::linalg::matmul(expression::linalg::transpose(A),
                                         B) + _beta*C;
        else if(TB)
          C = expression::linalg::matmul(A, expression::linalg::transpose(B))
                                         + _beta*C;
        else
          C = expression::linalg::matmul(A, B) + _beta*C;

  }

  template<typename xpu, index_t stream_id, typename T>
  TENSOR_INLINE_HOST void gemm(
       std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> A,
       std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> B,
       std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> C,
       T alpha, T beta, bool TA, bool TB){

    auto fwd = [](core::Tensor<xpu, stream_id, T> A,
                  core::Tensor<xpu, stream_id, T> B,
                  core::Tensor<xpu, stream_id, T> C,
                  T alpha, T beta, bool TA, bool TB)
    {
      operators::gemm(A, B, C, alpha, beta, TA, TB);
    };

    STREAM_FORWARD(xpu, stream_id).put(fwd, A->data(), B->data(), C->data(),
                                       alpha, beta, TA, TB);


    if (A->require_grad() && C->require_grad()){

      auto bwd = [](core::Tensor<xpu, stream_id, T> A,
                    core::Tensor<xpu, stream_id, T> B,
                    core::Tensor<xpu, stream_id, T> C,
                    T alpha, T beta, bool TA, bool TB)
      {
        operators::gemm(A, B, C, alpha, beta, TA, TB);
      };

      STREAM_BACKWARD(xpu, stream_id).put(bwd, C->gradient(), B->data(),
                      A->gradient(), alpha, A->accumulate_grad(), false, false);

    }

    if (B->require_grad() && C->require_grad()){

      auto bwd = [](core::Tensor<xpu, stream_id, T> A,
                    core::Tensor<xpu, stream_id, T> B,
                    core::Tensor<xpu, stream_id, T> C,
                    T alpha, T beta, bool TA, bool TB)
      {
        operators::gemm(A, B, C, alpha, beta, TA, TB);
      };

      STREAM_BACKWARD(xpu, stream_id).put(bwd, C->gradient(), A->data(),
                      B->gradient(), alpha, B->accumulate_grad(), !TA, !TB);

    }

  }




}

#endif
