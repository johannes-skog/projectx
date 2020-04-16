#ifndef FUNDAMENTAL_FUNDAMENTAL_H
#define FUNDAMENTAL_FUNDAMENTAL_H

#include <memory>
#include <fundamental/macro.h>
#include <fundamental/shape.h>


namespace expression{

template<typename xpu, index_t stream_id, typename SubType,
         typename T, index_t exp_type>
struct Exp{
  // returns const reference of the actual type of this expression
  TENSOR_INLINE_HOST const SubType& self(void) const{
    return *static_cast<const SubType*>(this);
  }
  TENSOR_INLINE_HOST auto eval();
};

namespace type{

  const index_t kRvalue = 0;
  const index_t kLvalue = 1;
  const index_t kscalar = 3;

}

typedef enum{

  EXP_TENSOR = 0,
  EXP_MAP = 1,
  EXP_BINARY = 2

} nn_exp_types;




template<typename xpu, index_t stream_id, typename OP, typename EL, typename ER,
         typename T>
struct Binary: public expression::Exp<xpu, stream_id,
       Binary<xpu, stream_id, OP, EL, ER, T>, T, expression::type::kRvalue>{

  EL el;
  ER er;

  const core::Shape& _shape;

  TENSOR_INLINE_HOST Binary(const EL& el, const ER& er, const core::Shape& _shape):
    el{el}, er{er}, _shape{_shape} {}

  TENSOR_INLINE T Eval(index_t s, index_t i) const{
    return OP::Eval(el.Eval(s, i), er.Eval(s, i));
  }

  TENSOR_INLINE void Backward(index_t s, index_t i, T dy){

    T del, der;

    OP::Backward(el.Eval(s, i), er.Eval(s, i), del, der);

    el.Backward(s, i, del*dy);
    er.Backward(s, i, der*dy);

  }

  TENSOR_INLINE const core::Shape& shape(void) const { return _shape; }

};

template<typename xpu, index_t stream_id, typename OP, typename E, typename T>
struct Map: public expression::Exp<xpu, stream_id,
       Map<xpu, stream_id, OP, E, T>, T, expression::type::kRvalue>{

  E e;

  const core::Shape& _shape;

  TENSOR_INLINE_HOST Map(const E& e, const core::Shape& s):e{e}, _shape{s} {}

  TENSOR_INLINE auto Eval(index_t s, index_t i) const{
    return OP::Eval(e.Eval(s, i));
  }

  TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
    e.Backward(s, i, OP::Backward(e.Eval(s, i)*dy  ));
  }

  TENSOR_INLINE const core::Shape& shape(void) const {return _shape;}

};

template<typename xpu, index_t stream_id, typename E, typename T, typename Tprev>
struct Cast: public Exp<xpu, stream_id, Cast<xpu, stream_id, E, T, Tprev>,
                        T, type::kRvalue>{

  E e;

  const core::Shape& _shape;

  TENSOR_INLINE_HOST Cast(const E& e, const core::Shape& _shape):
                          e{e}, _shape{_shape} {}

  TENSOR_INLINE T Eval(index_t s, index_t i) const{
   return (T) e.Eval(s, i);
  }

  TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
    e.Backward(s, i, (Tprev) dy);
  }

  TENSOR_INLINE const core::Shape& shape(void) const {return _shape;}

};

template<typename T, typename xpu, index_t stream_id, typename E, index_t exp_type,
         typename Tprev>
TENSOR_INLINE_HOST Cast<xpu, stream_id, E, T, Tprev>
  cast(const Exp<xpu, stream_id, E, Tprev, exp_type>& e){
  return Cast<xpu, stream_id, E, T, Tprev>(e.self(), e.self().shape());
}

template<typename xpu, index_t stream_id, typename OP, typename EL,
         typename ER, typename T, int tl, int tr>
Binary<xpu, stream_id, OP, EL, ER, T>
TENSOR_INLINE_HOST MakeExp(const Exp<xpu, stream_id, EL, T, tl> &el,
                           const Exp<xpu, stream_id, ER, T, tr> &er){
  // should be replaced with if constexp, where we chech tl tr for type::kscalar
  if (tl == type::kscalar){
    return Binary<xpu, stream_id, OP, EL, ER, T>
              (el.self(), er.self(), er.self().shape());
  }
  else if (tr == type::kscalar)
    return Binary<xpu, stream_id, OP, EL, ER, T>
               (el.self(), er.self(), el.self().shape());
  else {
    assert(el.self().shape() == er.self().shape());
    return Binary<xpu, stream_id, OP, EL, ER, T>
                (el.self(), er.self(), er.self().shape());
  }

}

template<typename xpu, index_t stream_id, typename OP, typename EL,
         typename ER, typename T, int tl, int tr>
TENSOR_INLINE_HOST Binary<xpu, stream_id, OP, EL, ER, T>
F(const Exp<xpu, stream_id, EL, T, tl>& el,
  const Exp<xpu, stream_id, ER, T, tr>& er){
  return MakeExp<xpu, stream_id, OP>(el, er);
}

template<typename OP, typename xpu, index_t stream_id, typename Texp,
         typename T, index_t exp_type>
TENSOR_INLINE_HOST Map<xpu, stream_id, OP, Texp, T>
F(const Exp<xpu, stream_id, Texp, T, exp_type>& e){
  return Map<xpu, stream_id, OP, Texp, T>(e.self(), e.self().shape());
}

template<typename xpu>
struct Exceturer {
  template<typename Esrc, typename Edst, index_t stream_id,
           typename T, index_t exp_type>
  TENSOR_INLINE_HOST static void excetute(
        const Exp<xpu, stream_id, Esrc, T, exp_type>&,
        const Exp<xpu, stream_id, Edst, T, type::kLvalue>&);
};

template<>
struct Exceturer<cpu>{

    template<typename Esrc, typename Edst, index_t stream_id,
             typename T, index_t exp_type>
    TENSOR_INLINE_HOST static void excetute(
          const Exp<cpu, stream_id, Esrc, T, exp_type>&,
          const Exp<cpu, stream_id, Edst, T, type::kLvalue>&);


    template<typename Esrc, typename Edst, index_t stream_id, typename T,
             index_t exp_typeSrc, index_t exp_typeDst>
    TENSOR_INLINE_HOST static void backward(
          const Exp<cpu, stream_id, Esrc, T, exp_typeSrc>&,
          const Exp<cpu, stream_id, Edst, T, exp_typeDst>&);

};

template<>
struct Exceturer<gpu>{

    template<typename Esrc, typename Edst, index_t stream_id,
             typename T, index_t exp_type>
    TENSOR_INLINE_HOST static void excetute(
          const Exp<gpu, stream_id, Esrc, T, exp_type>&,
          const Exp<gpu, stream_id, Edst, T, type::kLvalue>&);


    template<typename Esrc, typename Edst, index_t stream_id, typename T,
             index_t exp_typeSrc, index_t exp_typeDst>
    TENSOR_INLINE_HOST static void backward(
          const Exp<gpu, stream_id, Esrc, T, exp_typeSrc>&,
          const Exp<gpu, stream_id, Edst, T, exp_typeDst>&);

};

}

#include <fundamental/primitives.h>

#endif
