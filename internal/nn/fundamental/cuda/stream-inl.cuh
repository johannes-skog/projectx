#ifndef FUNDAMENTAL_STREAM_INL_CUH
#define FUNDAMENTAL_STREAM_INL_CUH

#include "cuda_runtime.h" // for cudaMalloc
#include "cuda.h"
#include <fundamental/stream.h>

#ifdef TENSOR_USE_CUBLAS
    #include <cublas_v2.h>
#endif

#ifdef TENSOR_USE_CUDNN
  #include <cudnn.h>
#endif

template<index_t stream_id>
class Stream<gpu, stream_id>: public StreamBase<gpu, stream_id, void>{

  #ifdef TENSOR_USE_CUBLAS
    cublasHandle_t _cublasHandle;
  #endif

  #ifdef TENSOR_USE_CUDNN
    cudnnHandle_t _cudnnHandle;
  #endif

  cudaStream_t stream;

public:

    Stream(stream_direction_t dir): StreamBase<gpu, stream_id, void>(dir) {

      cudaStreamCreate(&stream);

      #ifdef TENSOR_USE_CUBLAS
      CUDBLAS_CHECK_ERROR_ENFORCE(cublasCreate(&_cublasHandle));
      cublasSetStream(_cublasHandle, stream);
      #endif

      #ifdef TENSOR_USE_CUDNN
      CHECK_CUDNN(cudnnCreate(&_cudnnHandle));
      cudnnSetStream(_cudnnHandle, stream);
      #endif

    }

  cudaStream_t& cudaStream(void){ return stream; }

  #ifdef TENSOR_USE_CUBLAS
    cublasHandle_t& cublasHandle(void) { return _cublasHandle; }
  #endif

  #ifdef TENSOR_USE_CUDNN
    cudnnHandle_t& cudnnHandle(void) { return _cudnnHandle; }
  #endif

  void synchronize(void){
    this->_synchronize();
    cudaStreamSynchronize(stream); // No tasks are run or left, might it be
                                   // expressions that have been tasks that
                                   // have been placed in the stream via put
                                   // or we have transfer of mem via cuda::to_host etc
  }

};

#endif
