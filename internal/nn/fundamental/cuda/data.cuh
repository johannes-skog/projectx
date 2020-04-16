#ifndef FUNDAMENTAL_DATACU_H
#define FUNDAMENTAL_DATACU_H

#include <fundamental/macro.h>
#include <fundamental/cuda/stream-inl.cuh>
#include <cuda_runtime.h> // for cudaMalloc
#include "device_launch_parameters.h" // FOr blockdim etc
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>

#ifdef TENSOR_USE_CUDNN
  #include <cudnn.h>
#endif

#ifdef TENSOR_USE_CUDNN

  namespace cudnn{

    template <typename T>
    struct DataType{
      static cudnnDataType_t get(void);
    };


  }
#endif

namespace cuda{

    template<typename T = void>
    TENSOR_INLINE_HOST T* generate_workspace(size_t);

    template<typename T>
    TENSOR_INLINE_HOST void allocate(T**, size_t);

    template<typename T>
    TENSOR_INLINE_HOST void allocate_bytes(T**, size_t);

    template<typename T>
    TENSOR_INLINE_HOST void deallocate(T**);

    template<typename T, index_t stream_id>
    TENSOR_INLINE_HOST void to_host(T*, const T*, size_t);

    template<typename T, index_t stream_id>
    TENSOR_INLINE_HOST void to_device(T*, const T*, size_t);

    template<typename T>
    TENSOR_INLINE void memcpy(T*, const T*, size_t);

    template<typename T>
    __global__ void add(T);

     template<typename Esrc, typename Edst, typename T>
     __global__ void excetute(Esrc, Edst, const index_t, const index_t);

     template<typename Esrc, typename Edst, typename T>
     __global__ void backward(Esrc, Edst, const index_t, const index_t);

     template<typename T>
     void add_(T&);

     template<typename Esrc, typename Edst, index_t stream_id>
     void _excetute(const Esrc&, const Edst&);

     template<typename Esrc, typename Edst, index_t stream_id>
     void _backward(const Esrc&, const Edst&);

}

namespace expression{ namespace initilize{

  __global__ void initilize_rand(unsigned int, curandState_t*, index_t);

  }

}

namespace core{

  template<>
  class Descriptor<gpu, float>{

    #ifdef TENSOR_USE_CUDNN
      cudnnTensorDescriptor_t cudnn_desc;
    #endif

    public:

      Descriptor(){}

      Descriptor(const core::Shape&);

      void deallocate(void);

      cudnnTensorDescriptor_t get_descriptor() const;

      void operator=(const Descriptor<gpu, float>&);

  };

}


#endif
