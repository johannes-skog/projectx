#ifndef FUNDAMENTAL_INITIALIZER_H
#define FUNDAMENTAL_INITIALIZER_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <random>

namespace expression{ namespace initilize {

  template<typename InitOp, typename... Args>
  InitOp Initilizer(const core::Shape& _shape, Args&&... values){
    return InitOp(std::forward<Args>(values)..., _shape);
  }

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  struct Gaussian: public Exp<xpu, stream_id, Gaussian<xpu, stream_id,
                                                         T>, T, type::kRvalue>{};

  template<typename T, index_t stream_id>
  struct Gaussian<cpu, stream_id, T>: public  Exp<cpu, stream_id,
                                Gaussian<cpu, stream_id,T>, T, type::kRvalue>{

   std::shared_ptr<std::mt19937> gen;
   std::shared_ptr<std::normal_distribution<T>> dis;

   const core::Shape& _shape;

   TENSOR_INLINE_HOST Gaussian(T mu, T sigma, const core::Shape& _shape):
                    gen{std::make_shared<std::mt19937>(std::random_device{}())},
                    dis{std::make_shared<std::normal_distribution<T>>(mu, sigma)},
                    _shape{_shape} {}

   TENSOR_INLINE_HOST T Eval(index_t s, index_t i) const {
     return (*dis)(*gen);
   }

   TENSOR_INLINE_HOST const core::Shape& shape() const {return _shape;}

  };

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  struct Uniform: public Exp<xpu, stream_id, Uniform<xpu, stream_id, T>, T, type::kRvalue>{};


  template<typename T, index_t stream_id>
  struct Uniform<cpu, stream_id, T>: public  Exp<cpu, stream_id,
                                  Uniform<cpu, stream_id, T>, T, type::kRvalue>{

    std::shared_ptr<std::mt19937> gen;
    std::shared_ptr<std::uniform_real_distribution<T>> dis;

    const core::Shape& _shape;

    TENSOR_INLINE_HOST Uniform(T l, T h, const core::Shape& _shape):
                   gen{std::make_shared<std::mt19937>(std::random_device{}())},
                   dis{std::make_shared<std::uniform_real_distribution<T>>(l, h)},
                   _shape{_shape} {}

    TENSOR_INLINE_HOST T Eval(index_t s, index_t i) const {
      return (*dis)(*gen);
    }

    TENSOR_INLINE_HOST const core::Shape& shape() const {return _shape;}

  };

  template <typename xpu, index_t stream_id, typename SubType, typename T>
  void constant(expression::Exp<xpu, stream_id, SubType, T, type::kLvalue>& e, T v){
    SubType x = e.self();
    x = Initilizer<core::Constant<xpu, T>>(x.shape(), v);
  }

  template <typename xpu, index_t stream_id, typename SubType, typename T>
  void zeros(expression::Exp<xpu, stream_id, SubType, T, type::kLvalue>& e){
    constant(e, (T) 0);
  }

  template <typename xpu, index_t stream_id, typename SubType, typename T>
  void ones(expression::Exp<xpu, stream_id, SubType, T, type::kLvalue>& e){
    constant(e, (T) 1);
  }

  template <typename xpu, index_t stream_id, typename SubType, typename T>
  void gaussian(expression::Exp<xpu, stream_id, SubType, T, type::kLvalue>& e,
                T mu, T sigma){
    SubType x = e.self();
    x = Initilizer<Gaussian<xpu, stream_id, T>>(x.shape(), mu, sigma);
  }

  template <typename xpu, index_t stream_id, typename SubType, typename T>
  void uniform(expression::Exp<xpu, stream_id, SubType, T, type::kLvalue>& e,
               T l, T h){
    SubType x = e.self();
    x = Initilizer<Uniform<xpu, stream_id, T>>(x.shape(), l, h);
  }


} // expression

} // initilize

#endif

#ifdef TENSOR_USE_CUDA
  #include <fundamental/cuda/initializer.cuh>
#endif
