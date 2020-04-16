#ifndef FUNDAMENTAL_DATACU_INL_H
#define FUNDAMENTAL_DATACU_INL_H

#include <fundamental/cuda/data.cuh>

namespace cuda{

  template<typename T>
    TENSOR_INLINE_HOST T* generate_workspace(size_t wb){
      T* ptr;
      cuda::allocate_bytes(&ptr, wb);
      return ptr;
    }

    template<typename T>
     TENSOR_INLINE_HOST void allocate(T** ptr, size_t N){
      CUDA_CHECK_ERROR_ENFORCE(cudaMalloc((void**)ptr, sizeof(T)*N));
    }

    template<typename T>
    TENSOR_INLINE_HOST void allocate_bytes(T** ptr, size_t bytes){
      CUDA_CHECK_ERROR_ENFORCE(cudaMalloc((void**)ptr, bytes));
    }

    template<typename T>
    TENSOR_INLINE_HOST void deallocate(T** ptr){
      CUDA_CHECK_ERROR_ENFORCE(cudaFree(*ptr));
      (*ptr) = nullptr;
    }

    template<typename T, index_t stream_id>
    TENSOR_INLINE_HOST void to_host(T* host, const T* device, size_t N){
      CUDA_CHECK_ERROR_ENFORCE(cudaMemcpy((void*)host, (void*)device,
                                           sizeof(T)*N,
                                           cudaMemcpyDeviceToHost));
    }

    template<typename T>
    TENSOR_INLINE_HOST void memcpy(T* dest, const T* src, size_t N){
      CUDA_CHECK_ERROR_ENFORCE(cudaMemcpy((void*)dest, (void*)src,
                                           sizeof(T)*N,
                                           cudaMemcpyDeviceToDevice));
    }

    template<typename T, index_t stream_id>
    TENSOR_INLINE_HOST void to_device(T* device, const T* host, size_t N){
      CUDA_CHECK_ERROR_ENFORCE(cudaMemcpy((void*)device, (void*)host,
                                           sizeof(T)*N,
                                           cudaMemcpyHostToDevice));
    }

     template<typename Esrc, typename Edst>
     __global__ void excetute(Esrc src, Edst dst, const index_t Nx, const index_t Ny){

         int idxx = threadIdx.x + blockIdx.x * blockDim.x;
         int idxy = threadIdx.y + blockIdx.y * blockDim.y;

         if ( (idxx < Nx) && (idxy < Ny) )
          dst.Set(idxx, idxy, src.Eval(idxx, idxy));

      }


    template<typename Esrc, typename Edst>
    __global__ void backward(Esrc src, Edst dst, const index_t Nx, const index_t Ny){

          int idxx = threadIdx.x + blockIdx.x * blockDim.x;
          int idxy = threadIdx.y + blockIdx.y * blockDim.y;

          if ( (idxx < Nx) && (idxy < Ny) )
           dst.Backward(idxx, idxy, src.BackwardEval(idxx, idxy));

     }

     template<typename Esrc, typename Edst, index_t stream_id>
     void _excetute(const Esrc& src, const Edst& dst){

        int bzy = src.shape().stride > CUDA_MAX_THREADS  ?
                  CUDA_MAX_THREADS : src.shape().stride;
        int bzx = CUDA_MAX_THREADS / bzy ;

        dim3 blockSize(bzx, bzy);

        index_t gridSizex = (src.shape().N + blockSize.x - 1 ) / blockSize.x;
        index_t gridSizey = (src.shape().stride + blockSize.y  - 1) / blockSize.y;

        dim3 gridSize(gridSizex, gridSizey);

        excetute<<<gridSize, blockSize, 0, STREAM_FORWARD(gpu, stream_id).cudaStream()>>>
              (src, dst, src.shape().N, src.shape().stride);

   }

   template<typename Esrc, typename Edst, index_t stream_id>
   void _backward(const Esrc& src, const Edst& dst){

      int bzy = src.shape().stride > CUDA_MAX_THREADS  ?
                CUDA_MAX_THREADS : src.shape().stride;
      int bzx = CUDA_MAX_THREADS / bzy ;

      dim3 blockSize(bzx, bzy);

      index_t gridSizex = (src.shape().N + blockSize.x - 1 ) / blockSize.x;
      index_t gridSizey = (src.shape().stride + blockSize.y  - 1) / blockSize.y;

      dim3 gridSize(gridSizex, gridSizey);

      backward<<<gridSize, blockSize, 0, STREAM_BACKWARD(gpu, stream_id).cudaStream()>>>
            (src, dst, src.shape().N, src.shape().stride);

 }

}

namespace expression{

  template<typename Esrc, typename Edst, index_t stream_id, typename T,
           index_t exp_type>
   TENSOR_INLINE_HOST void Exceturer<gpu>::excetute(
     const Exp<gpu, stream_id, Esrc, T, exp_type>& src_exp,
     const Exp<gpu, stream_id, Edst, T, type::kLvalue>& dst_exp){

     DEBUG_ASSERT(src_exp.self().shape() == dst_exp.self().shape());

     auto task = [](Esrc src, Edst dst){
      cuda::_excetute<Esrc, Edst, stream_id>(src, dst);
     };

     STREAM_FORWARD(gpu, stream_id).put(task, src_exp.self(), dst_exp.self());

  }

  template<typename Esrc, typename Edst, index_t stream_id, typename T,
           index_t exp_typeSrc, index_t exp_typeDst>
   TENSOR_INLINE_HOST void Exceturer<gpu>::backward(
     const Exp<gpu, stream_id, Esrc, T, exp_typeSrc>& src_exp,
     const Exp<gpu, stream_id, Edst, T, exp_typeDst>& dst_exp){

     DEBUG_ASSERT(src_exp.self().shape() == dst_exp.self().shape());

     auto task = [](Esrc src, Edst dst){
      cuda::_backward<Esrc, Edst, stream_id>(src, dst);
     };

     STREAM_BACKWARD(gpu, stream_id).put(task, src_exp.self(), dst_exp.self());

  }

}


namespace primitives{

  template<>
  struct Primitives<gpu>: public PrimitivesBase{

    struct exp{

      TENSOR_INLINE_CUDA static float Eval(float v){
        return expf(v);
      }

      template<typename T>
      TENSOR_INLINE_CUDA static float Backward(float v){
        return expf (v);
      }

    };

    struct sqrt{

      TENSOR_INLINE static float Eval(float v){
        return sqrtf(v);
      }

      TENSOR_INLINE static float Backward(float v){
        return float(1) / (2 * sqrt::Eval(v));
      }

    };

    struct abs{

      TENSOR_INLINE static float Eval(float v){
        return fabsf(v);
      }

      TENSOR_INLINE static float Backward(float v){
        return v > 0 ? 1 : -1;
      }

    };

    struct ln{

      TENSOR_INLINE_CUDA static float Eval(float v){
        return logf(v);
      }

      TENSOR_INLINE_CUDA static float Backward(float v){
        return float(1) / v ;
      }

    };

    struct tanh{

      TENSOR_INLINE_CUDA static float Eval(float v){
        return tanhf(v);
      }

      TENSOR_INLINE_CUDA static float Backward(float v){
        return float(1) - square::Eval(tanh::Eval(v));
      }

    };


  };

}

#endif
