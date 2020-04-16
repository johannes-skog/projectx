#ifndef N_OPTIMIZER_H
#define N_OPTIMIZER_H

#include <string>

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

#include <nn/layer.h>

#include <nn/container.h>
#include <nn/initializer.h>


namespace nn{

template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
class Optimizer{

  core::Scalar<xpu, T, stream_id> alpha;

  public:

    Optimizer(T _alpha): alpha{_alpha} {}

    Optimizer(void) = default;

    virtual ~Optimizer() {}

    void zero_grad(std::string s){

      nn::initilizer::zeros<xpu, stream_id, T>(s,
                          nn::initilizer::datatype_t::NN_INITILIZER_GRADIENT);

    }

    void zero_grad(){
      this->zero_grad(DIFFERENTIABLE);
    }

    virtual void step(std::string s){

      STREAM_BACKWARD(xpu, stream_id).hold_off().synchronize();

      auto trainable = nn::ContainerContext<xpu, stream_id, T>().search(s);

      for (auto& w : trainable){
        //printf("OPTIMZER FOUND %s\n", w.first.c_str() );
        auto d = w.second->data(); auto g = w.second->gradient();
        d = d + alpha * g;
      }

    }

    virtual void step(void){
      this->step(TRAINABLE + ".*" + DIFFERENTIABLE + '|' +
                 DIFFERENTIABLE + ".*" + TRAINABLE);
    }

};


}


#endif
