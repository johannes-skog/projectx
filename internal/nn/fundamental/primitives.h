#ifndef  FUNDAMENTAL_PRIMITIVES_H
#define  FUNDAMENTAL_PRIMITIVES_H

#include <math.h>

namespace primitives{

  struct PrimitivesBase{

    struct mul{

      template<typename T>
      TENSOR_INLINE static T Eval(T l, T r){
        return l * r;
      }

      template<typename T>
      TENSOR_INLINE static void Backward(T l, T r, T& dl, T& dr){
        dl = r; dr = l;
      }

    };

    struct div{

      template<typename T>
      TENSOR_INLINE static T Eval(T n, T d){
        return n / d;
      }

      template<typename T>
      TENSOR_INLINE static void Backward(T n, T d, T& dn, T& dd){
        dn = (T)1 / d;
        dd = n / (d * d);
      }

    };

    struct plus{

      template<typename T>
      TENSOR_INLINE static T Eval(T l, T r){
        return l + r;
      }

      template<typename T>
      TENSOR_INLINE static void Backward(T l, T r, T& dl, T& dr){
        dl = (T)1; dr = (T)1;
      }

    };

    struct minus{

      template<typename T>
      TENSOR_INLINE static T Eval(T l, T r){
        return l - r;
      }

      template<typename T>
      TENSOR_INLINE static void Backward(T l, T r, T& dl, T& dr){
        dl = (T)1; dr = -(T)1;
      }

    };

    struct square{

      template<typename T>
      TENSOR_INLINE static T Eval(T v){
        return v * v;
      }

      template<typename T>
      TENSOR_INLINE static T Backward(T v){
        return 2*v;
      }

    };

  };


  template<typename xpu>
  struct Primitives: public PrimitivesBase{};

  template<>
  struct Primitives<cpu>: public PrimitivesBase{

    struct exp{

      template<typename T>
      TENSOR_INLINE static T Eval(T v){
        return std::exp(v);
      }

      template<typename T>
      TENSOR_INLINE static T Backward(T v){
        return std::exp(v);
      }

    };

    struct abs{

      template<typename T>
      TENSOR_INLINE static T Eval(T v){
        return std::abs(v);
      }

      template<typename T>
      TENSOR_INLINE static T Backward(T v){
        return v > 0 ? 1 : -1;
      }

    };

    struct sqrt{

      template<typename T>
      TENSOR_INLINE static T Eval(T v){
        return std::sqrt(v);
      }

      template<typename T>
      TENSOR_INLINE static T Backward(T v){
        return T(1) / (2 * sqrt::Eval(v));
      }

    };

    struct ln{

      template<typename T>
      TENSOR_INLINE static T Eval(T v){
        return std::log(v);
      }

      template<typename T>
      TENSOR_INLINE static T Backward(T v){
        return T(1) / v ;
      }

    };

    struct tanh{

      template<typename T>
      TENSOR_INLINE_CUDA static T Eval(T v){
        return std::tanh(v);
      }

      template<typename T>
      TENSOR_INLINE_CUDA static T Backward(T v){
        return 1 - square::Eval(tanh::Eval(v));
      }

    };

  };

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::square, Texp, T>
  square(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::square, Texp, T>
            (e.self(), e.self().shape());
  }

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::abs, Texp, T>
  abs(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::abs, Texp, T>
            (e.self(), e.self().shape());
  }

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::sqrt, Texp, T>
  sqrt(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::sqrt, Texp, T>
            (e.self(), e.self().shape());
  }

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::exp, Texp, T>
  exp(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::exp, Texp, T>
            (e.self(), e.self().shape());
  }

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::ln, Texp, T>
  ln(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::ln, Texp, T>
            (e.self(), e.self().shape());
  }

  template<typename xpu, index_t stream_id, typename Texp,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST expression::Map<xpu, stream_id, typename Primitives<xpu>::tanh, Texp, T>
  tanh(const expression::Exp<xpu, stream_id, Texp, T, exp_type>& e){
    return expression::Map<xpu, stream_id, typename Primitives<xpu>::tanh, Texp, T>
            (e.self(), e.self().shape());
  }

}

#endif
