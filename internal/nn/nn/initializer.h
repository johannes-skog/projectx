#ifndef NN_INITILIZER_H
#define NN_INITILIZER_H

#include <string>

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

#include <nn/layer.h>

#include <nn/container.h>



namespace nn{

  namespace initilizer{

    typedef enum{

      NN_INITILIZER_DATA = 0,
      NN_INITILIZER_GRADIENT = 1,

    } datatype_t;

    template<typename xpu, index_t stream_id, typename T>
    void _execute(std::string s, std::function<void(core::Tensor<xpu, stream_id, T>)> f,
                  datatype_t datat){

     auto x = nn::ContainerContext<xpu, stream_id, T>().search(s);

     for (auto& c : x){

        auto w  = c.second;

        if(datat == NN_INITILIZER_DATA){
          auto d = w->data();
          f(d);
        } else if (datat == NN_INITILIZER_GRADIENT && w->require_grad()){
          auto g = w->gradient();
          f(g);
        }

      }

    }

    template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
    void zeros(std::string s, datatype_t datat = NN_INITILIZER_DATA){

      auto f = [](core::Tensor<xpu, stream_id, T> x){
        expression::initilize::zeros(x);
      };

      _execute<xpu, stream_id, T>(s, f, datat);

    }


    template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
    void ones(std::string s, datatype_t datat = NN_INITILIZER_DATA){

      auto f = [](core::Tensor<xpu, stream_id, T> x){
        expression::initilize::ones(x);
      };

      _execute<xpu, stream_id, T>(s, f, datat);

    }


    template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
    void uniform(std::string s, T l, T h, datatype_t datat = NN_INITILIZER_DATA){

      auto f = [l, h](core::Tensor<xpu, stream_id, T> x){
        expression::initilize::uniform(x, l, h);
      };

      _execute<xpu, stream_id, T>(s, f, datat);

    }


    template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
    void gaussian(std::string s, T mu, T sigma, datatype_t datat = NN_INITILIZER_DATA){

      auto f = [mu, sigma](core::Tensor<xpu, stream_id, T> x){
        expression::initilize::gaussian(x, mu, sigma);
      };

      _execute<xpu, stream_id, T>(s, f, datat);

    }

  }

}




#endif
