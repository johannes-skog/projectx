#include <fundamental/tensor.h>
#include <fundamental/cuda/initializer.cuh>

namespace expression{ namespace initilize {

  __global__ void initilize_rand(unsigned int seed, curandState_t* states, index_t N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
      curand_init(seed, blockIdx.x /* seqeunce idx */, 0 /*offset*/,   &states[idx]);
  }

}}
