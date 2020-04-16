#ifndef FUNDAMENTAL_TRANSFORM_SLICE_H
#define FUNDAMENTAL_TRANSFORM_SLICE_H

#include <fundamental/transform/transform_base.h>
#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

namespace core{

    class SliceTransformer{

      public:

        Shape org_shape;
        Shape new_shape;

        index_t offset[SHAPE_MAX_DIM];

        SliceTransformer(const Shape& t_shape, std::vector<index_t> slice):
                         org_shape{t_shape}{

          DEBUG_ASSERT(slice.size() == 2*this->org_shape.dim);
          // We have start and end for each dim -> 2*dim elements
          index_t t_sliced_shape[SHAPE_MAX_DIM];
          for (int i = 0; i < this->org_shape.dim; ++i){
            index_t start = slice[2*i]; index_t end = slice[2*i+1];
            index_t lim = this->org_shape[i];
            if (end < 0) end = this->org_shape[i] + end; // warp
            DEBUG_ASSERT ( start >= 0 ); DEBUG_ASSERT ( start < end );
            DEBUG_ASSERT ( end <= lim );
            t_sliced_shape[i] = end - start;
            this->offset[i] = start;
          }
          this->new_shape = core::Shape(t_sliced_shape, org_shape.dim);

        }

        TENSOR_INLINE const core::Shape& shape() const { return new_shape; }

        TENSOR_INLINE void transform(index_t s, index_t i, index_t& s_n, index_t& i_n) const {
          index_t values[SHAPE_MAX_DIM];
          this->new_shape.deduce_shape(s, i, values);
          for(int i = 0; i < this->new_shape.dim; ++i) values[i] += this->offset[i];
          this->org_shape.index(s_n, i_n, values);
        }

    };

}

namespace expression{ namespace transform{

  namespace op{

    template<typename xpu, index_t stream_id, typename E, typename T>
    struct Slice: public Exp<xpu, stream_id, Slice<xpu, stream_id, E, T>, T, type::kLvalue>{

      E e;

      core::SliceTransformer shape_trans;

      Slice(const E& e, std::vector<index_t> slice): e{e},
            shape_trans{e.self().shape(), std::forward<std::vector<index_t>>(slice)} {}

      template<typename Esrc, index_t exp_type>
      TENSOR_INLINE_HOST Slice<xpu, stream_id, E, T>&
            operator=(const expression::Exp<xpu, stream_id, Esrc, T, exp_type>& src){
        DEBUG_ASSERT(src.self().shape() == this->shape());
        expression::Exceturer<xpu>::excetute(src, *this);
        return *this;
      }

      TENSOR_INLINE T Eval(index_t s, index_t i) const {
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        return e.Eval(s_n, i_n);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        e.Backward(s_n, i_n, dy);
      }

      TENSOR_INLINE void Set(index_t s, index_t i, T v){
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        e.Set(s_n, i_n, v);
      }

      TENSOR_INLINE const core::Shape& shape() const { return shape_trans.shape(); }

    };


    template<typename xpu, index_t stream_id, typename E, typename T>
    struct Slice: public Exp<xpu, stream_id, Slice<xpu, stream_id, E, T>, T, type::kLvalue>{

      E e;

      core::SliceTransformer shape_trans;

      Slice(const E& e, std::vector<index_t> slice): e{e},
            shape_trans{e.self().shape(), std::forward<std::vector<index_t>>(slice)} {}

      template<typename Esrc, index_t exp_type>
      TENSOR_INLINE_HOST Slice<xpu, stream_id, E, T>&
            operator=(const expression::Exp<xpu, stream_id, Esrc, T, exp_type>& src){
        DEBUG_ASSERT(src.self().shape() == this->shape());
        expression::Exceturer<xpu>::excetute(src, *this);
        return *this;
      }

      TENSOR_INLINE T Eval(index_t s, index_t i) const {
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        return e.Eval(s_n, i_n);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        e.Backward(s_n, i_n, dy);
      }

      TENSOR_INLINE void Set(index_t s, index_t i, T v){
        index_t s_n, i_n;
        this->shape_trans.transform(s, i, s_n, i_n);
        e.Set(s_n, i_n, v);
      }

      TENSOR_INLINE const core::Shape& shape() const { return shape_trans.shape(); }

    };

  }

  template<typename xpu, index_t stream_id, typename E, typename T, index_t exp_type>
  TENSOR_INLINE_HOST op::Slice<xpu, stream_id, E, T>
          slice(Exp<xpu, stream_id, E, T, exp_type>& exp,
                std::vector<index_t> slice){
    return op::Slice<xpu, stream_id, E, T>(exp.self(),
                std::forward<std::vector<index_t>>(slice));
  }


}}

#endif
