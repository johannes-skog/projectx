#ifndef FUNDAMENTAL_TENSOR_INL_H
#define FUNDAMENTAL_TENSOR_INL_H

#include <fundamental/tensor.h>
#include <memory.h>

namespace core{

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    Tensor<gpu, stream_id, T> to_cuda(Tensor<cpu, stream_id, T>& src){
       Tensor<gpu, stream_id, T> dst(src.shape());
       dst.allocate(); to_cuda(dst, src);
       return dst;
    }

    template<index_t stream_id, typename T>
    void to_cuda(Tensor<gpu, stream_id, T>& dst,
                 Tensor<cpu, stream_id, T>& src){
       DEBUG_ASSERT( (dst.shape() == src.shape()) );
       DEBUG_ASSERT( (dst.check_allocation() && src.check_allocation()) );
       auto task =  [](T* dst_ptr, T* src_ptr, index_t N){
         cuda::to_device<T, stream_id>(dst_ptr, src_ptr, N);
       };
       STREAM(gpu, stream_id).put(task, dst.ptr(), src.ptr(), src.shape().size());
    }

    template<index_t stream_id, typename T>
    Tensor<cpu, stream_id, T> to_cpu(Tensor<gpu, stream_id, T>& src){
       Tensor<cpu, stream_id, T> dst(src.shape());
       dst.allocate(); to_cpu(dst, src);
       return dst;
    }

    template<index_t stream_id, typename T>
    void to_cpu(Tensor<cpu, stream_id, T>& dst,
               Tensor<gpu, stream_id, T>& src){
       DEBUG_ASSERT( (dst.shape() == src.shape()) );
       DEBUG_ASSERT( (dst.check_allocation() && src.check_allocation()) );
       auto task =  [](T* dst_ptr, T* src_ptr, index_t N){
         cuda::to_host<T, stream_id>(dst_ptr, src_ptr, N);
       };
       STREAM(cpu, stream_id).put(task, dst.ptr(), src.ptr(), src.shape().size());
    }

    template<index_t stream_id, typename T>
    Tensor<gpu, stream_id, T> copy(Tensor<gpu, stream_id, T>& src){
       DEBUG_ASSERT(  (src.check_allocation()) );
       Tensor<gpu, stream_id, T> dst(src.shape());
       dst.allocate();
       auto task =  []( T* dst_ptr, T* src_ptr, index_t N){
         cuda::memcpy<T, stream_id>(dst_ptr, src_ptr, N);
       };
       STREAM(gpu, stream_id).put(task, dst.ptr(), src.ptr(), src.shape().size());
       return dst;
    }

  #endif

  template<typename xpu, index_t stream_id, typename T>
  std::vector<core::Tensor<xpu, stream_id, T>>
  copy(std::vector<core::Tensor<xpu, stream_id, T>> xv){
    std::vector<core::Tensor<xpu, stream_id, T>> y;
    for (auto& x : xv)
      y.push_back(core::copy(x));
    return y;
  }

  template<index_t stream_id, typename T>
  Tensor<cpu, stream_id, T> copy(Tensor<cpu, stream_id, T>& src){
     DEBUG_ASSERT(  (src.check_allocation()) );
     Tensor<cpu, stream_id, T> dst(src.shape());
     dst.allocate();
     auto task =  [](T* dst_ptr, T* src_ptr, index_t N){
        memcpy(dst_ptr, src_ptr, N);
     };
     STREAM(cpu, stream_id).put(task, dst.ptr(), src.ptr(), src.shape().size());
     return dst;
  }

  #ifdef TENSOR_USE_CUDA
    template<index_t stream_id, typename T>
    void store(Storage<T, std::ios::out >& out,
               Tensor<gpu, stream_id, T>& src){
       Tensor<cpu, stream_id, T> src_cpu = to_cpu(src);
       store(out, src_cpu);
       src_cpu.deallocate();
    }
  #endif

  template<index_t stream_id, typename T>
  void store(Storage<T, std::ios::out >& out,
             Tensor<cpu, stream_id, T>& src){
    out.write(src.ptr(), src.shape().size());
  }

  #ifdef TENSOR_USE_CUDA
    template<index_t stream_id, typename T>
    void load(Storage<T, std::ios::in >& in,
              Tensor<gpu, stream_id, T>& dst){
       Tensor<cpu, stream_id, T> dst_cpu = to_cpu(dst);
       load(in, dst_cpu); to_cuda(dst, dst_cpu);
       dst_cpu.deallocate();
    }
  #endif

  template<index_t stream_id, typename T>
  void load(Storage<T, std::ios::in >& in,
             Tensor<cpu, stream_id, T>& dst){
    // TODO check out stream
    in.read(dst.ptr(), dst.shape().size());
  }

}

namespace memory{

  template<index_t stream_id, typename T>
  TENSOR_INLINE_HOST T* Memory<cpu, stream_id, T>::allocate(const size_t N){
    T *ptr;
    auto task = [](T** _ptr, const size_t N){
      (*_ptr) = new T[N];
    };
    STREAM(gpu, stream_id).put(task, &ptr, N);
    return ptr;
  }

  template<index_t stream_id, typename T>
  TENSOR_INLINE_HOST void Memory<cpu, stream_id, T>::deallocate(T* ptr){ delete [] ptr;}

  template<index_t stream_id, typename T>
  TENSOR_INLINE_HOST void Memory<cpu, stream_id, T>::set(T* ptr, T v) { *ptr = v; }

  template<index_t stream_id, typename T>
  TENSOR_INLINE_HOST T Memory<cpu, stream_id, T>::at(T* ptr) { return *ptr; }

  template<index_t stream_id, typename T>
  TENSOR_INLINE_HOST void Memory<cpu, stream_id, T>::memcpy(T* dst, const T* src, size_t N) {
    memcpy(dst, src, sizeof(T)*N);
  }

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST T* Memory<gpu, stream_id, T>::allocate(const size_t N){
      T *ptr;
      auto task = [](T** _ptr, const size_t N){
        cuda::allocate(_ptr, N);
      };
      STREAM(gpu, stream_id).put(task, &ptr, N);
      return ptr;
    }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST T Memory<gpu, stream_id, T>::at(T* ptr) {
      T host; cuda::to_host<T, stream_id>(&host, ptr, 1);
      return host;
    }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST void Memory<gpu, stream_id, T>::deallocate(T* ptr) {
      auto task = [](T* ptr){
        cuda::deallocate(&ptr);
      };
      STREAM(gpu, stream_id).put(task, ptr);
     }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST void Memory<gpu, stream_id, T>::set(T* ptr, T v) {
      cuda::to_device<T, DEFAULT_STREAM>(ptr, &v, 1);
    }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST void Memory<gpu, stream_id, T>::to_device(T* device, const T* host, size_t N) {
      cuda::to_device<T, DEFAULT_STREAM>(device, host, N);
    }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST void Memory<gpu, stream_id, T>::to_host(T* host, const T* device, size_t N) {
      cuda::to_host<T, DEFAULT_STREAM>(host, device, N);
    }

    template<index_t stream_id, typename T>
    TENSOR_INLINE_HOST void Memory<gpu, stream_id, T>::memcpy(T* dst, const T* src, size_t N) {
      cuda::memcpy<T>(dst, src, N);
    }

  #endif

}

#ifdef TENSOR_USE_CUDA

  #include <fundamental/cuda/data-inl.cuh>

#endif


#endif
