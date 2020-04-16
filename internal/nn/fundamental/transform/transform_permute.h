#ifndef FUNDAMENTAL_TRANSFORM_PERMUTE_H
#define FUNDAMENTAL_TRANSFORM_PERMUTE_H

#include <fundamental/transform/transform_base.h>
#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/macro.h>


namespace core{

  class PermuteShape{

    public:

      Shape new_shape;

      const Shape org_shape;

      index_t dim_mapping[SHAPE_MAX_DIM];

      index_t dim;

      PermuteShape(const Shape& t_shape, std::vector<index_t>& values):
                   org_shape{t_shape}, dim{static_cast<index_t>(values.size())}{
        for (int i = 0; i < dim; ++i) dim_mapping[i] = values[i];
        DEBUG_ASSERT( (dim == this->org_shape.dim) );
        index_t rearranged[SHAPE_MAX_DIM];
        for (int i = 0; i < dim; ++i){
          DEBUG_ASSERT( (dim_mapping[i] < dim) );
          rearranged[i] = this->org_shape[dim_mapping[i]];
        }
        this->new_shape = core::Shape(rearranged, this->org_shape.dim);
        DEBUG_ASSERT(new_shape.size() == org_shape.size());
      }

      TENSOR_INLINE void transform(index_t s, index_t i, index_t& s_n, index_t& i_n) const{
        index_t new_shape_map[SHAPE_MAX_DIM];
        index_t org_shape_map[SHAPE_MAX_DIM];
        this->new_shape.deduce_shape(s, i, new_shape_map);
        for (int idx = 0; idx < this->dim; ++idx){
          org_shape_map[this->dim_mapping[idx]] = new_shape_map[idx];
        }
        this->org_shape.index(s_n, i_n, org_shape_map);
      }

      TENSOR_INLINE const core::Shape& shape() const { return new_shape; }

  };

}

namespace expression{ namespace transform{

  namespace op{

    template<typename xpu, index_t stream_id, typename Texp, typename T>
    struct Permute: public Exp<xpu, stream_id,
                           Permute<xpu, stream_id, Texp, T>, T, type::kRvalue>{

      Texp e;

      const core::PermuteShape shape_trans;

      Permute(const Texp& e, std::vector<index_t>& values): e{e},
              shape_trans{e.self().shape(), values}{}

      TENSOR_INLINE T Eval(index_t s, index_t i) const {
        index_t s_n, i_n;
        shape_trans.transform(s, i, s_n, i_n);
        return e.Eval(s_n, i_n);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T dy){
        index_t s_n, i_n;
        shape_trans.transform(s, i, s_n, i_n);
        e.Backward(s_n, i_n, dy);
      }

      TENSOR_INLINE const core::Shape& shape() const {return shape_trans.shape();}

    };

  }

  template<typename xpu, index_t stream_id, typename Texp, typename T, index_t exp_type>
  op::Permute<xpu, stream_id, Texp, T> permute(const Exp<xpu, stream_id, Texp, T, exp_type>& exp,
                                               std::vector<index_t> values){
    return op::Permute<xpu, stream_id, Texp, T>(exp.self(), values);
  }


}}


#endif
