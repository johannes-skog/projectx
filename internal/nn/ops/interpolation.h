#ifndef OPS_INTERPOLATE_H
#define OPS_INTERPOLATE_H

#include <fundamental/tensor.h>
#include <omp.h>

#ifdef TENSOR_USE_CUDA
  #include <cublas_v2.h>
#endif

namespace operators{

  template<typename T>
  __global__ void _bilinear_interpolation(T* __restrict__ tensorA,
                                          const core::Shape shapeA,
                                          T* __restrict__ tensorB,
                                          const core::Shape shapeB,
                                          float fracX, float fracY){

    const int xB = threadIdx.x + blockDim.x * blockIdx.x; // Width
    const int yB = threadIdx.y + blockDim.y * blockIdx.y; // Height
    const int nB = threadIdx.z + blockDim.z * blockIdx.z; // Batch

    if ( (nB >= shapeB[0]) || (yB >= shapeB[1]) || (xB >= shapeB[2]) )
         return;

    const float xp = float( xB ) * fracX;
    const float yp = float( yB ) * fracY;

    const int x1 = floor( xp );
    const int y1 = floor( yp );

    const int x2 = x1 + 1;
    const int y2 = y1 + 1;

    const float dx = xp - x1;
    const float dy = yp - y1;

    /*
    /* Q1  R1   Q2
    /*
    /*    P
    /*
    /* Q3  R2   Q4
    */

    T* offsetA = tensorA + shapeA.stride * nB;
    T* strideY1 = offsetA + y1 * shapeA[2] * shapeA[3];
    T* strideY2 = offsetA + y2 * shapeA[2] * shapeA[3];

    const int strideX1 = x1 * shapeA[3];
    const int strideX2 = x2 * shapeA[3];

    T *Q1{nullptr}, *Q2{nullptr}, *Q3{nullptr}, *Q4{nullptr};
    float  alpha1{0}, alpha2{0}, alpha3{0}, alpha4{0};
    if ( ( x1 < shapeA[2] ) && ( y1 < shapeA[1] ) ){
      Q1 = strideY1 + strideX1;
      alpha1 = (1 - dy) * (1 - dx);
    }

    if ( ( x2 < shapeA[2] ) && ( y1 < shapeA[1] ) ){
      Q2 = strideY1 + strideX2;
      alpha2 = (1 - dy) * dx;
    }

    if ( ( x1 < shapeA[2] ) && ( y2 < shapeA[1] ) ){
      Q3 = strideY2 + strideX1;
      alpha3 = dy * (1 - dx);
    }

    if ( ( x2  < shapeA[2] ) && ( y2 < shapeA[1] ) ){
      Q4 = strideY2 + strideX2;
      alpha4 = dx * dy;
    }

    T* tensorBptr = tensorB + shapeB.stride * nB + yB * shapeB[2] * shapeB[3] + xB * shapeB[3];
    for (int c = 0; c < shapeB[3]; ++c){
      T v = 0;
      if (Q1) v += alpha1 * Q1[c];
      if (Q2) v += alpha2 * Q2[c];
      if (Q3) v += alpha3 * Q3[c];
      if (Q4) v += alpha4 * Q4[c];
      tensorBptr[c] = v;
    }

  }

  template<index_t stream_id, typename T>
  void bilinear_interpolation(core::Tensor<gpu, stream_id, T> A ,
                              core::Tensor<gpu, stream_id, T> B){

    DEBUG_ASSERT( (A.shape().dim == 4 && B.shape().dim == 4) );
    DEBUG_ASSERT( (A.shape()[0] == B.shape()[0]) );
    DEBUG_ASSERT( (A.shape()[3] == B.shape()[3]) );

    auto task = [](core::Tensor<gpu, stream_id, T> A,
                   core::Tensor<gpu, stream_id, T> B){

         float fracY = (float) A.shape()[1] / (float) B.shape()[1] ;
         float fracX = (float) A.shape()[2] / (float) B.shape()[2] ;

         float ratio = B.shape()[2] / B.shape()[1];

         // max = bx * by
         // bx = ratio * by

         int by = std::sqrt( CUDA_MAX_THREADS /  ratio );

         int bx = ratio * by;

         int bz = CUDA_MAX_THREADS / float(bx * by) < 1 ? 1 :  CUDA_MAX_THREADS / float(bx * by);

         dim3 blockSize(bx, by, bz);

         index_t gridSizex = (B.shape()[2] + blockSize.x  - 1) / blockSize.x;
         index_t gridSizey = (B.shape()[1] + blockSize.y  - 1) / blockSize.y;
         index_t gridSizez = (B.shape()[0] + blockSize.z  - 1) / blockSize.z;

         dim3 gridSize(gridSizex, gridSizey, gridSizez);
         _bilinear_interpolation<<<gridSize, blockSize, 0, STREAM_FORWARD(gpu, stream_id).cudaStream()>>>(
                 A.ptr(), A.shape(), B.ptr(), B.shape(), fracX, fracY);


      };

      STREAM_FORWARD(gpu, stream_id).put(task, A, B);

    }

}


#endif
