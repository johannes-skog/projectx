#ifndef NN_BIAS_H
#define NN_BIAS_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/transform/transform_base.h>
#include <fundamental/macro.h>

#include <nn/layer.h>
#include <fundamental/linalg/linalg_base.h>

#include <ops/gemm.h>

namespace nn{

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  class Bias: public Layer<xpu, stream_id, T>{

  protected:

    const index_t stop_dim;

    const index_t n;

  public:

    Bias() = default;

    Bias(index_t stop_dim,
         index_t n, bool requires_grad = true):
         Layer<xpu, stream_id, T>("Bias", GEN_UNIQUE_ID(decltype(this)), requires_grad),
         stop_dim{stop_dim}, n{n}{

      auto w = this->register_param("weight");
      w->set(core::Shape(n, 1));

      this->register_param("ones", false, false);

    }

    ~Bias() {}

    core::Shape shape_output(const core::Shape& in){
      DEBUG_ASSERT(in.dim > stop_dim );
      DEBUG_ASSERT(in[stop_dim] == this->n);
      return in;
    }

    void _forward_inplace(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> xc){

      auto& shape = xc->shape();

      this->shape_output(shape); // Verify the input

      index_t first_dim = shape[0];
      for (int i = 1; i < this->stop_dim; ++i)
        first_dim *= shape[i];

      auto wc = this->param("weight");

      auto xc_view = nn::view(xc, core::Shape(first_dim, this->n) );

      this->param("ones")->conditional_override(core::Shape(xc_view->shape()[0], 1));

      T alpha = 1;
      T beta  = 1;

      operators::gemm(this->param("ones"), wc, xc_view, alpha, beta, false, true);

    }

    void _forward(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> xc,
                  std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>> yc){

      yc->conditional_override(xc->shape());

      yc->data().copy_data(xc->data());

      this->_forward_inplace(yc);

    }

    void _backward(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>& xc,
                   std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>& yc){}

    void _backward_inplace(std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>& yc){}

};

}
#endif
