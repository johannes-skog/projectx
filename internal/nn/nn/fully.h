#ifndef NN_FULLY_H
#define NN_FULLY_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/transform/transform_base.h>
#include <fundamental/macro.h>

#include <nn/layer.h>

#include <fundamental/linalg/linalg_base.h>

#ifdef TENSOR_USE_CUDA
  #include <cudnn.h>
#endif

#ifdef TENSOR_USE_CUBLAS
    #include <cublas_v2.h>
#endif

#include <ops/gemm.h>

namespace nn{

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  class FullyConnected: public Layer<xpu, stream_id, T>{

  protected:

    const int n_output;

    const int n_input;

  public:

    FullyConnected() = default;

    FullyConnected(int n_input, int n_output,
                   bool requires_grad = true):
                   Layer<xpu, stream_id, T>( "FullyConnected", GEN_UNIQUE_ID(decltype(this)), requires_grad),
                   n_output{n_output}, n_input{n_input}{

      auto w = this->register_param("weight");

      w->set(core::Shape(n_output, n_input));

    }

    ~FullyConnected(){}

    core::Shape shape_output(const core::Shape& in){
      DEBUG_ASSERT(in.dim == 2);
      DEBUG_ASSERT(in[1] == this->n_input);
      return core::Shape(in[0], this->n_output);
    }

    void _forward(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> xc,
                  std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> yc){

      core::Shape output_shape = this->shape_output(xc->shape());

      yc->conditional_override(output_shape);

      T alpha = 1;
      T beta = 0;

      operators::gemm(xc, this->param("weight"), yc,  alpha, beta, false, true);

    }


    void _backward(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> xc,
                   std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> yc){

      /*
      DEBUG_ASSERT(this->is_grad());

      auto& x = xc.data();
      auto& dy = yc.gradient();

      auto& w = this->param("weight").data();
      auto& dw = this->param("weight").gradient();

      T alpha = 1.0;

      core::Shape output_shape = this->shape_output(x.shape());

      DEBUG_ASSERT(dy.shape() == output_shape);

      auto& dx = xc.gradient();

      */

      /* ----- dw -----
      /*
      /* dw = dy^T * x -> calculate instead dw^T = x^T * dy
      /* we will only need to transpose dy
      /* A = dy [n, k]
      /* B = dy [n, m]
      /* C = dw [k, m]
      /*
      */
      // beta = 1 accumulate the grad

      //operators::gemm(dy, x, dw, alpha, (T)this->accumulate_grad(),
      //                true, false);

      /* ----- dx -----
      /*
      /* dx = dy * w
      /* A = dy [n, k]
      /* B = w [k, m]
      /* C = dx [n, m]
      /*
      */
      //operators::gemm(dy, w, dx, alpha, 0.0, false, false);


    }




};

}
#endif
