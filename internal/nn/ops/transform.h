#ifndef OPS_TRANSFORM_H
#define OPS_TRANSFORM_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>
#include <fundamental/transform/transform_slice.h>

namespace operators{

  template<typename T>
  struct Test{

    TENSOR_INLINE static T Eval(T v) { return 2.2; }

  };

  struct Test2{

    struct Forward{
      template<typename T>
      TENSOR_INLINE static T Eval(T v) { return 2.2; }
    };

    struct Backward{
      template<typename T>
      TENSOR_INLINE static T Eval(T v) { return 2.2; }
    };

  };

  template<typename T>
  struct Zero{
    TENSOR_INLINE static T Forward(T) { return 0;}
  };

  template<typename MapT, typename xpu, index_t stream_id, typename T>
  auto& slice_map(core::Tensor<xpu, stream_id, T>& tensor,
                  std::vector<index_t> _slice){
     DEBUG_ASSERT(_slice.size() == 2 * tensor.shape().dim);
     expression::transform::slice(tensor, _slice) =
          expression::F<MapT>(expression::transform::slice(tensor, _slice));
     return tensor;
  }

  template<typename xpu, index_t stream_id, typename T>
  void slice(core::Tensor<xpu, stream_id, T>& tensorA,
             core::Tensor<xpu, stream_id, T>& tensorB,
             std::vector<index_t> _slice){
     DEBUG_ASSERT(_slice.size() == 2 * tensorA.shape().dim);
     tensorB = expression::transform::slice(tensorA, _slice);
  }

  template<typename xpu, index_t stream_id, typename T>
  auto slice(core::Tensor<xpu, stream_id, T>& tensorA,
             std::vector<index_t> _slice){

     core::SliceTransformer slice_trans(tensorA.shape(), _slice);
     core::Tensor<xpu, stream_id, T> tensorB(slice_trans.shape());
     tensorB.allocate();

     slice(tensorA, tensorB, _slice);

     return tensorB;

  }

  template<typename xpu, index_t stream_id, typename T>
  void concat(core::Tensor<xpu, stream_id, T>& tensorA,
              core::Tensor<xpu, stream_id, T>& tensorB,
              core::Tensor<xpu, stream_id, T>& tensorC,
              index_t axis){

     DEBUG_ASSERT(tensorB.shape().dim == tensorA.shape().dim);

     core::ConcatShape concat_trans(tensorA.shape(), tensorB.shape(), axis);
     tensorC = expression::transform::concat(tensorA, tensorB, axis);

  }

  template<typename xpu, index_t stream_id, typename T>
  auto concat(core::Tensor<xpu, stream_id, T>& tensorA,
              core::Tensor<xpu, stream_id, T>& tensorB,
              index_t axis){

     DEBUG_ASSERT(tensorB.shape().dim == tensorA.shape().dim);

     core::ConcatShape concat_trans(tensorA.shape(), tensorB.shape(), axis);

     core::Tensor<xpu, stream_id, T> tensorC(concat_trans.shape());
     tensorC.allocate(); // We must make the allocation before its passed to the stream

     concat(tensorA, tensorB, tensorC, axis);

     return tensorC;

  }

  template<typename xpu, index_t stream_id, typename T>
  void permute(core::Tensor<xpu, stream_id, T>& tensorA,
               core::Tensor<xpu, stream_id, T>& tensorB,
               std::vector<index_t> values){
     tensorB = expression::transform::permute(tensorA, values);
  }

  template<typename xpu, index_t stream_id, typename T>
  auto permute(core::Tensor<xpu, stream_id, T>& tensorA,
               std::vector<index_t> values){

     core::PermuteShape permute_trans(tensorA.shape(), values);
     core::Tensor<xpu, stream_id, T> tensorB(permute_trans.shape());
     tensorB.allocate();

     permute(tensorA, tensorB, values);
     return tensorB;

  }

}

#endif
