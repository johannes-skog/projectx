#ifndef OPS_REDUCERS_H
#define OPS_REDUCERS_H

#include <fundamental/tensor.h>
#include <nn/container.h>

#ifdef TENSOR_USE_CUBLAS
  #include <cublas_v2.h>
#endif

#ifdef TENSOR_USE_OPENBLAS
  #include <cblas.h>
#endif

#include <ops/transform.h>

namespace operators{

  template<typename xpu, index_t stream_id, typename T>
  void asum_backward(nn::TensorContainer<xpu, stream_id, T>& x,
                     nn::TensorContainer<xpu, stream_id, T>& y,
                     int offset, int inc, int n){

     auto bwd = [](nn::TensorContainer<xpu, stream_id, float> x,
                   nn::TensorContainer<xpu, stream_id, float> y,
                   int offset, int inc, int n)
     {

       auto strided_x = expression::transform::stride(x, offset, inc, n);

       core::Shape shape = strided_x.self().shape();

       //auto repeated_y = expression::transform::repeat(shape.size(), yexp);
       core::Constant<xpu, T, stream_id> repeated_y(y.gradient().scalar(),
                                                    core::Shape(x.shape().size()));

       expression::Exceturer<xpu>::backward(repeated_y, strided_x);

     };

     STREAM_BACKWARD(xpu, stream_id).put(bwd, x, y, offset, inc, n);

   }

  template<index_t stream_id>
  void asum(nn::TensorContainer<cpu, stream_id, float>& x,
            nn::TensorContainer<cpu, stream_id, float>& y,
            int offset, int inc, int n){

    //y.conditional_override(core::Shape(1));

    ASSERT_ERROR( ((n+offset) - (x.data().shape().size())) <= 0 , " Out of bound ");

    auto fwd = [](nn::TensorContainer<cpu, stream_id, float> x,
                  nn::TensorContainer<cpu, stream_id, float> y,
                  int offset, int inc, int n)
    {
      float* ptr_offseted = x.data().ptr() + offset;
      float result = cblas_sasum(n, ptr_offseted, inc);
      y.data().set(0, result);
    };

    STREAM_FORWARD(cpu, stream_id).put(fwd, x, y, offset, inc, n);

    if (x.require_grad() && y.require_grad())
      asum_backward(x, y, offset, inc, n);

  }

  #ifdef TENSOR_USE_CUBLAS

    template<index_t stream_id>
    void asum(nn::TensorContainer<gpu, stream_id, float>& x,
              nn::TensorContainer<gpu, stream_id, float>& y,
              int offset, int inc, int n){

      DEBUG_ASSERT(y.data().shape().size() == 1);
      ASSERT_ERROR( ((n+offset) - (x.data().shape().size())) <= 0 , " Out of bound ");

      auto fwd = [](nn::TensorContainer<gpu, stream_id, float> x,
                    nn::TensorContainer<gpu, stream_id, float> y,
                    int offset, int inc, int n)
      {
        auto cublasHandle = STREAM_FORWARD(gpu, stream_id).cublasHandle();
        float result;
        float* ptr_offseted = x.data().ptr() + offset;
        CUDBLAS_CHECK_ERROR_ENFORCE(cublasSasum(cublasHandle, n,
                                                ptr_offseted, inc,
                                                &result));
        y.data().set(0, result);

      };

      STREAM_FORWARD(gpu, stream_id).put(fwd, x, y, offset, inc, n);

      if (x.require_grad() && y.require_grad())
        asum_backward(x, y, offset, inc, n);

    }

  #endif

}

#ifdef TENSOR_USE_CUDNN

  namespace cudnn {

    template<cudnnReduceTensorOp_t OPERATION, index_t stream_id>
    void reduce_cudnn_i(core::Tensor<gpu, stream_id, float>& tensorA,
                        core::Tensor<gpu, stream_id, float>& tensorB,
                        core::Tensor<gpu, stream_id, indices_t>& indicies,
                        float alpha = 1, float beta = 0){

      DEBUG_ASSERT( indicies.shape() == tensorB.shape() );

      cudnnReduceTensorDescriptor_t reduce;
      cudnnCreateReduceTensorDescriptor(&reduce);

      CHECK_CUDNN( cudnnSetReduceTensorDescriptor(
                   reduce,
                   OPERATION,
                   CUDNN_DATA_FLOAT,
                   CUDNN_NOT_PROPAGATE_NAN,
                   CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                   CUDNN_32BIT_INDICES) );

      size_t workspaces_sz;

      auto cudnnhandle = STREAM_FORWARD(gpu, stream_id).cudnnHandle();

      CHECK_CUDNN( cudnnGetReductionWorkspaceSize(cudnnhandle,
                                                  reduce,
                                                  tensorA.descriptor(),
                                                  tensorB.descriptor(),
                                                  &workspaces_sz) );

      auto workspace_ptr = cuda::generate_workspace<void>(workspaces_sz);

      CHECK_CUDNN( cudnnReduceTensor(
                                     cudnnhandle,
                                     reduce, // reduceTensorDesc
                                     indicies.ptr(), // indicies
                                     sizeof(indices_t) * indicies.shape().size(),  // indicesSizeInBytes,
                                     workspace_ptr, // workspace,
                                     workspaces_sz, //workspaceSizeInBytes,
                                     (void*) &alpha,
                                     tensorA.descriptor(),
                                     (void*) tensorA.ptr(),
                                     (void*) &beta,
                                     tensorB.descriptor(),
                                     (void*) tensorB.ptr()) ) ;

      for (int i = 0; i < indicies.shape()[0];  ++i)
        printf(" %f\n", tensorB.at(i) );

    }

    template<cudnnReduceTensorOp_t OPERATION, index_t stream_id>
    void reduce_cudnn(core::Tensor<gpu, stream_id, float>& tensorA,
                      core::Tensor<gpu, stream_id, float>& tensorB,
                      float alpha = 1, float beta = 0){

      cudnnReduceTensorDescriptor_t reduce;
      cudnnCreateReduceTensorDescriptor(&reduce);

      CHECK_CUDNN( cudnnSetReduceTensorDescriptor(
                   reduce,
                   OPERATION,
                   CUDNN_DATA_FLOAT,
                   CUDNN_NOT_PROPAGATE_NAN,
                   CUDNN_REDUCE_TENSOR_NO_INDICES,
                   CUDNN_32BIT_INDICES) );

      size_t workspaces_sz;

      auto cudnnhandle = STREAM_FORWARD(gpu, stream_id).cudnnHandle();

      CHECK_CUDNN( cudnnGetReductionWorkspaceSize(cudnnhandle,
                                                  reduce,
                                                  tensorA.descriptor(),
                                                  tensorB.descriptor(),
                                                  &workspaces_sz) );

      auto workspace_ptr = cuda::generate_workspace<void>(workspaces_sz);

      float dummy;

      CHECK_CUDNN( cudnnReduceTensor(
                                     cudnnhandle,
                                     reduce, // reduceTensorDesc
                                     &dummy, // indicies
                                     0,  // indicesSizeInBytes,
                                     workspace_ptr, // workspace,
                                     workspaces_sz, //workspaceSizeInBytes,
                                     (void*) &alpha,
                                     tensorA.descriptor(),
                                     (void*) tensorA.ptr(),
                                     (void*) &beta,
                                     tensorB.descriptor(),
                                     (void*) tensorB.ptr()) ) ;

    }

  }

#endif

namespace operators{

  #ifdef TENSOR_USE_CUDNN

    template<index_t stream_id>
    void reduce_sum(core::Tensor<gpu, stream_id, float>& tensorA,
                    core::Tensor<gpu, stream_id, float>& tensorB,
                    float alpha = 1, float beta = 0){
      cudnn::reduce_cudnn<CUDNN_REDUCE_TENSOR_ADD, stream_id>
              (tensorA, tensorB, alpha, beta);
    }

    template<index_t stream_id>
    void reduce_max(core::Tensor<gpu, stream_id, float>& tensorA,
                    core::Tensor<gpu, stream_id, float>& tensorB,
                    float alpha = 1, float beta = 0){
      cudnn::reduce_cudnn<CUDNN_REDUCE_TENSOR_MAX, stream_id>
              (tensorA, tensorB, alpha, beta);
    }

    template<index_t stream_id>
    void reduce_min(core::Tensor<gpu, stream_id, float>& tensorA,
                    core::Tensor<gpu, stream_id, float>& tensorB,
                    float alpha = 1, float beta = 0){
      cudnn::reduce_cudnn<CUDNN_REDUCE_TENSOR_MIN, stream_id>
              (tensorA, tensorB, alpha, beta);
    }

    template<index_t stream_id>
    void reduce_max_i(core::Tensor<gpu, stream_id, float>& tensorA,
                      core::Tensor<gpu, stream_id, float>& tensorB,
                      core::Tensor<gpu, stream_id, indices_t>& indices,
                      float alpha = 1, float beta = 0){
      cudnn::reduce_cudnn_i<CUDNN_REDUCE_TENSOR_MAX, stream_id>
              (tensorA, tensorB, indices, alpha, beta);
    }

    template<index_t stream_id>
    void reduce_min_i(core::Tensor<gpu, stream_id, float>& tensorA,
                      core::Tensor<gpu, stream_id, float>& tensorB,
                      core::Tensor<gpu, stream_id, indices_t>& indices,
                      float alpha = 1, float beta = 0){
      cudnn::reduce_cudnn_i<CUDNN_REDUCE_TENSOR_MIN, stream_id>
              (tensorA, tensorB, indices, alpha, beta);
    }


    template<index_t stream_id>
    TENSOR_INLINE_HOST void reduce_min(
        std::shared_ptr<nn::TensorContainer<gpu, stream_id, float>> A,
        std::shared_ptr<nn::TensorContainer<gpu, stream_id, float>> B,
        index_t axis){

        B->conditional_override(A->shape().reduce_axis(axis));

        if (B->require_grad()){
          // Create tensor for incicies
          core::Tensor<gpu, stream_id, float> I(B->shape()); I.allocate();

          auto fwd = [](core::Tensor<gpu, stream_id, float> A,
                        core::Tensor<gpu, stream_id, float> B,
                        core::Tensor<gpu, stream_id, float> I){
            reduce_min_i(A, B, I);
          };

          STREAM_FORWARD(gpu, stream_id).put(fwd, A->data(), B->data(), I);



        }else{

          auto fwd = [](core::Tensor<gpu, stream_id, float> A,
                        core::Tensor<gpu, stream_id, float> B){
            reduce_min(A, B);
          };

          STREAM_FORWARD(gpu, stream_id).put(fwd, A->data(), B->data());

        }

    }


  #endif

}


#endif //OPS_REDUCERS_H
