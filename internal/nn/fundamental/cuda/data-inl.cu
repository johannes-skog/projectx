#include <fundamental/tensor.h>


#ifdef TENSOR_USE_CUDNN

  namespace cudnn{

    template<>
    cudnnDataType_t DataType<float>::get(void)  { return CUDNN_DATA_FLOAT; }

  }

#endif

namespace core{

    Descriptor<gpu, float>::Descriptor(const core::Shape& shape) {

      #ifdef TENSOR_USE_CUDNN

        CHECK_CUDNN( cudnnCreateTensorDescriptor(&cudnn_desc));

        if (shape.dim <= 4){

          int n = shape[0];
          int h = shape.dim > 1 ? shape[1] : 1;
          int w = shape.dim > 2 ? shape[2] : 1;
          int c = shape.dim > 3 ? shape[3] : 1;

          CHECK_CUDNN( cudnnSetTensor4dDescriptor(cudnn_desc,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_DATA_FLOAT,
                                       n, c, h, w) );
        } else {

          CHECK_CUDNN( cudnnSetTensorNdDescriptorEx(
                                       cudnn_desc,
                                       CUDNN_TENSOR_NHWC,
                                       CUDNN_DATA_FLOAT,
                                       shape.dim,
                                       shape.shape) );
        }

      #endif

    }

    cudnnTensorDescriptor_t Descriptor<gpu, float>::get_descriptor() const{

      #ifdef TENSOR_USE_CUDNN
        return cudnn_desc;
      #else
        return nullptr;
      #endif

    }

    void Descriptor<gpu, float>::deallocate(void) {

      #ifdef TENSOR_USE_CUDNN

        CHECK_CUDNN(cudnnDestroyTensorDescriptor(cudnn_desc));

      #endif

    }

    void Descriptor<gpu, float>::operator=(const Descriptor<gpu, float>& src){

      cudnn_desc = src.get_descriptor();

    }


}
