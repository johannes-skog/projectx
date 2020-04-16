#ifndef FUNDAMENTAL_TRANSFORM_REDUCER_H
#define FUNDAMENTAL_TRANSFORM_REDUCER_H

namespace core{

    class SliceMapTransformer{

      public:

        Shape org_shape;
        Shape new_shape;
        index_t axis;

        SliceTransformer(const Shape& t_shape,  index_t _axis)
                         org_shape{t_shape}, new_shape{t_shape.reduce_axis(_axis)},
                         axis{_axis} {}

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
    struct SliceMap: public Exp<xpu, stream_id,
                            SliceMap<xpu, stream_id, E, T>, T, type::kLvalue>{

      E e;

      core::SliceMapTransformer shape_trans;

      core::Tensor<xpu, stream_id, indices_t> map;

      SliceMap(const E& _e, core::Tensor<xpu, stream_id, indices_t> _map, index_t _axis):
               e{_e}, map{_map}, shape_trans{e.self().shape(), _axis} {
        DEBUG_ASSERT(map.shape() == shape_trans.shape());
      }

      template<typename Esrc, index_t exp_type>
      TENSOR_INLINE_HOST SliceMap<xpu, stream_id, E, T>&
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

        index_t deduced[SHAPE_MAX_DIM];

        this->shape.deduce_shape(s, i, deduced);

        deduced[this->shape_trans.axis] =
                    std::static_cast<indices_t>(this->map.at(deduced));

        index_t s_n, i_n;

        this->shape_trans.org_shape.index(s_n, i_n, deduced);

        e.Set(s_n, i_n, v);

      }

      TENSOR_INLINE const core::Shape& shape() const { return shape_trans.shape(); }

    };

  }

  template<typename xpu, index_t stream_id, typename E, typename T, index_t exp_type>
  TENSOR_INLINE_HOST op::Slice<xpu, stream_id, E, T>
          slice_map(Exp<xpu, stream_id, E, T, exp_type>& exp,
                    core::Tensor<xpu, stream_id, indices_t>& map, index_t axis>){
    return op::SliceMap<xpu, stream_id, E, T>(exp.self(), map, axis);
  }


}}

#endif
