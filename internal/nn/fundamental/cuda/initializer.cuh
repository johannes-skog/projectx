#pragma once
#ifndef FUNDAMENTAL_INITIALIZER_CUH
#define FUNDAMENTAL_INITIALIZER_CUH

#include <time.h>

#include <fundamental/initializer.h>
#include <fundamental/cuda/data-inl.cuh>

namespace expression{ namespace initilize {

  template<typename T, index_t stream_id>
  struct Gaussian<gpu, stream_id, T>: public  Exp<gpu, stream_id, Gaussian<gpu, stream_id, T>,
                                     T, type::kRvalue>{

    curandState* states;

    const T mu;
    const T sigma;

    const bool _copy;
    const core::Shape& _shape;

    const index_t stride;

    Gaussian(T mu, T sigma, const core::Shape& _shape):
                                       mu{mu}, sigma{sigma},
                                      _shape{_shape}, states{nullptr},
                                      stride{_shape.stride},
                                      _copy{false}{
      index_t N = _shape.size();
      auto task = [](curandState** states, index_t N){
        cuda::allocate(states, N);
        int blockSize = CUDA_MAX_THREADS;
        index_t gridSize = (N + blockSize - 1) / blockSize;
        initilize_rand<<< gridSize, blockSize, 0, STREAM(gpu, stream_id).cudaStream()>>>
                          (SEED.next(), *states, N);
      };

      STREAM(gpu, stream_id).put(task, &states, N);

    }

    Gaussian(const Gaussian<gpu, stream_id, T>& src):
                          _shape{src._shape}, _copy{true}, states{src.states},
                          mu{src.mu}, sigma{src.sigma}, stride{src.stride}{}

    ~Gaussian(){ if (!_copy) cuda::deallocate(&states); }

    TENSOR_INLINE const core::Shape& shape() const {return _shape;}

    TENSOR_INLINE_CUDA T Eval(index_t s, index_t i) const {
      index_t idx = s*stride + i;
      return curand_normal(&states[idx])*sigma + mu;
    }

  };

  template<typename T, index_t stream_id>
  struct Uniform<gpu, stream_id, T>: public  Exp<gpu, stream_id, Uniform<gpu, stream_id, T>,
                                    T, type::kRvalue>{

    curandState* states;

    const T l;
    const T scale;

    const bool _copy;
    const core::Shape& _shape;

    const index_t stride;

    Uniform(T l, T h, const core::Shape& _shape): l{l}, scale{h-l},_shape{_shape},
                                                  states{nullptr}, _copy{false},
                                                  stride(_shape.stride) {
      assert (scale>0);
      index_t N = _shape.size();
      auto task = [](curandState** states, index_t N){
        cuda::allocate(states, N);
        int blockSize = CUDA_MAX_THREADS;
        index_t gridSize = (N + blockSize - 1) / blockSize;
        initilize_rand<<< gridSize, blockSize, 0, STREAM(gpu, stream_id).cudaStream()>>>
                          (SEED.next(), *states, N);
        };
      STREAM(gpu, stream_id).put(task, &states, N);

    }

    Uniform(const Uniform<gpu, stream_id, T>& src): _shape{src._shape}, _copy{true},
                                        states{src.states}, l{src.l}, scale{src.scale},
                                        stride{src.stride}{}

    ~Uniform(){ if (!_copy) cuda::deallocate(&states); }

    TENSOR_INLINE const core::Shape& shape() const {return _shape;}

    TENSOR_INLINE_CUDA T Eval(index_t s, index_t i) const {
      index_t idx = s*stride + i;
      return curand_uniform(&states[idx])*scale + l;
    }

  };

}
}

#endif
