#ifndef NN_CONV_H
#define NN_CONV_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>

#include <nn/layer.h>

#ifdef TENSOR_USE_CUDNN
  #include <cudnn.h>
#endif

namespace nn{

typedef enum{

  NN_CONV_PADDING_SAME = 0,
  NN_CONV_PADDING_VALID = 1,

} conv_padding;

template<typename xpu, index_t stream_id, typename T>
class Convolution: public Layer<xpu, stream_id, T>{

  protected:

  public:

    Convolution(std::string desc = "Convolution", bool requires_grad = true):
    Layer<xpu, stream_id, T>(desc, GEN_UNIQUE_ID(decltype(this)), requires_grad) {}

};

template<typename xpu, index_t stream_id =  DEFAULT_STREAM, typename T = DEFAULT_TYPE>
class Convolution2dBase: public Convolution<xpu, stream_id, T>{

  protected:

    index_t _batchsize, _input[2];

    const conv_padding padding_type;

    const index_t channels_in, channels_out;

    const index_t kernel[2];

    const index_t stride[2];

    const index_t dilation[2];

  public:

    Convolution2dBase(index_t channels_in, index_t channels_out,
                      index_t kernel_height, index_t kernel_width,
                      index_t stride_vertical, index_t stride_horizontal,
                      conv_padding padding = NN_CONV_PADDING_SAME,
                      index_t dilation_height = 1,
                      index_t dilation_width = 1, bool requires_grad = true):
    _batchsize{-1}, _input{-1, -1}, channels_in{channels_in},
    channels_out{channels_out},
    kernel{kernel_height, kernel_width}, stride{stride_vertical, stride_horizontal},
    padding_type{padding}, dilation{dilation_height, dilation_width},
    Convolution<xpu, stream_id, T>("Convolution2d", requires_grad) {}

};

template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
class Convolution2d: public Convolution2dBase<xpu, stream_id, T>{};

#ifdef TENSOR_USE_CUDNN

  template<index_t stream_id, typename T>
  class Convolution2d<gpu, stream_id, T>: public Convolution2dBase<gpu, stream_id, T>{

    private:

      const cudnnHandle_t cudnn_handle;

      cudnnDataType_t cudnn_datatype;

      cudnnFilterDescriptor_t kernel_descriptor;

      cudnnConvolutionDescriptor_t conv_descriptor;

      cudnnConvolutionFwdAlgo_t algo_descriptor_fw;

      cudnnConvolutionBwdDataAlgo_t algo_descriptor_data_bw;

      cudnnConvolutionBwdFilterAlgo_t algo_descriptor_filter_bw;


      struct Workspace{

        void* ptr;
        size_t size;

        Workspace(): ptr{nullptr}, size{0} {}

        void deallocate(void){
          size = 0;
          if (ptr != nullptr)
            cuda::deallocate(&ptr);
        }

        void allocate(void){
          cuda::allocate_bytes(&ptr, size);
        }

      };

      Workspace fw_workspace;
      Workspace bw_data_workspace;
      Workspace bw_filter_workspace;

      TENSOR_INLINE_HOST size_t dim_transform(size_t x, size_t k, size_t s){

        return ( (x - k + 2 * calculate_padding(x, k, s) )  / s )  + 1;

      }

      TENSOR_INLINE_HOST size_t calculate_padding(size_t x, size_t k, size_t s){

        switch (this->padding_type) {
          case NN_CONV_PADDING_SAME:
            return ( k - s ) / 2.0 ;
          case NN_CONV_PADDING_VALID:
            return 0;
          default:
            return 0;
        }

      }

      void update_descriptor(std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> xc,
                             std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> yc)
      {

        bool update = false;

        const core::Shape& shape = xc->shape();

        if (shape[order::H] != this->_input[0] ||
            shape[order::W] != this->_input[1] ){

            size_t padding_h = calculate_padding(shape[order::H],
                                                 this->kernel[0],
                                                 this->stride[1]);

            size_t padding_w = calculate_padding( shape[order::W],
                                                 this->kernel[1],
                                                 this->stride[1]);

            CHECK_CUDNN( cudnnSetConvolution2dDescriptor(conv_descriptor,
                                         /*pad_height=*/padding_h,
                                         /*pad_width=*/padding_w,
                                         /*vertical_stride=*/this->stride[0],
                                         /*horizontal_stride=*/this->stride[1],
                                         /*dilation_height=*/this->dilation[0],
                                         /*dilation_width=*/this->dilation[1],
                                         /*mode=*/CUDNN_CROSS_CORRELATION,
                                         /*computeType=*/CUDNN_DATA_FLOAT));

            update = true;

            this->_input[0] = shape[order::H];
            this->_input[1] = shape[order::W];

        }

        if(shape[order::N] != this->_batchsize)
          update = true;

        if (update){

          auto x = xc->data();
          auto y = yc->data();

          fw_workspace.deallocate();

          CHECK_CUDNN( cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                           x.descriptor(),
                                                           kernel_descriptor,
                                                           conv_descriptor,
                                                           y.descriptor(),
                                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                           /*memoryLimitInBytes=*/0,
                                                           &algo_descriptor_fw));

          CHECK_CUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                              x.descriptor(),
                                                              kernel_descriptor,
                                                              conv_descriptor,
                                                              y.descriptor(),
                                                              algo_descriptor_fw,
                                                              &fw_workspace.size));
          fw_workspace.allocate();

          if (this->is_grad()){

            bw_data_workspace.deallocate();

            bw_filter_workspace.deallocate();

            CHECK_CUDNN( cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle,
                                         kernel_descriptor,
                                         y.descriptor(),
                                         conv_descriptor,
                                         x.descriptor(),
                                         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                         0,
                                         &algo_descriptor_data_bw));

            CHECK_CUDNN( cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                           kernel_descriptor,
                                           y.descriptor(),
                                           conv_descriptor,
                                           x.descriptor(),
                                           algo_descriptor_data_bw,
                                           &bw_filter_workspace.size));

            bw_filter_workspace.allocate();

            CHECK_CUDNN( cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle,
                                         x.descriptor(),
                                         y.descriptor(),
                                         conv_descriptor,
                                         kernel_descriptor,
                                         CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                         0,
                                         &algo_descriptor_filter_bw));

            CHECK_CUDNN( cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle,
                                              x.descriptor(),
                                              y.descriptor(),
                                              conv_descriptor,
                                              kernel_descriptor,
                                              algo_descriptor_filter_bw,
                                              &bw_filter_workspace.size));

            bw_filter_workspace.allocate();

          }

        }

    }

    public:

      Convolution2d(index_t channels_in, index_t channels_out,
                    index_t kernel_height, index_t kernel_width,
                    index_t stride_vertical, index_t stride_horizontal,
                    conv_padding padding = NN_CONV_PADDING_SAME,
                    index_t dilation_height = 1, index_t dilation_width = 1,
                    bool requires_grad = true):
        Convolution2dBase<gpu, stream_id, T>(channels_in, channels_out,
                    kernel_height, kernel_width, stride_vertical, stride_horizontal,
                    padding, dilation_height, dilation_width, requires_grad),
        cudnn_datatype{cudnn::DataType<T>::get()},
        cudnn_handle{STREAM(gpu, stream_id).cudnnHandle()}
       {

        CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));

        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor));

        CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                               /*dataType=*/cudnn_datatype,
                                               /*format=*/CUDNN_TENSOR_NHWC,
                                               /*out_channels=*/channels_out,
                                               /*in_channels=*/channels_in,
                                               /*kernel_height=*/kernel_height,
                                               /*kernel_width=*/kernel_width));

         auto w = this->register_param("weight");

         w->set(core::Shape(channels_out, kernel_height, kernel_width, channels_in));

      }

    ~Convolution2d(){

      CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));

      CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor));

      fw_workspace.deallocate();
      bw_data_workspace.deallocate();
      bw_filter_workspace.deallocate();

    }

    core::Shape shape_output(const core::Shape& in){

      DEBUG_ASSERT(in.dim == 4);
      DEBUG_ASSERT(in[order::C] == this->channels_in);

      core::Shape y(in);

      y.shape[order::C] = this->channels_out;

      y.shape[order::H] = this->dim_transform(y.shape[order::H],
                                              this->kernel[0],
                                              this->stride[0]);

      y.shape[order::W] = this->dim_transform(y.shape[order::W],
                                              this->kernel[1],
                                              this->stride[1]);

      return y;

    }

    void _forward(std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> xc,
                  std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> yc){

      core::Shape output_shape = this->shape_output(xc->shape());

      yc->conditional_override(output_shape);

      this->update_descriptor(xc, yc);

      auto fwd = [this](std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> xc,
                        std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> yc)
      {

        auto x = xc->data();

        auto y = yc->data();

        const T alpha = 1, beta = 0;

        auto w = this->param("weight")->data();

        CHECK_CUDNN( cudnnConvolutionForward(cudnn_handle,
                                            (void*) &alpha,
                                            x.descriptor(),
                                            x.ptr(),
                                            kernel_descriptor,
                                            w.ptr(),
                                            conv_descriptor,
                                            algo_descriptor_fw,
                                            fw_workspace.ptr,
                                            fw_workspace.size,
                                            (void*) &beta,
                                            y.descriptor(),
                                            y.ptr()));

      };

      STREAM_FORWARD(gpu, stream_id).put(fwd, xc, yc);


      if (this->is_grad()){

        auto bwd = [this](std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> xc,
                          std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> yc)
        {

          auto x = xc->data();
          auto dx = xc->gradient();
          auto dy = yc->gradient();
          auto w = this->param("weight")->data();
          auto dw = this->param("weight")->gradient();

          T alpha = 1, beta = this->param("weight")->accumulate_grad();

          CHECK_CUDNN( cudnnConvolutionBackwardFilter(cudnn_handle,
                                                      (void*) &alpha,
                                                      x.descriptor(),
                                                      x.ptr(),
                                                      dy.descriptor(),
                                                      dy.ptr(),
                                                      this->conv_descriptor,
                                                      algo_descriptor_filter_bw,
                                                      bw_filter_workspace.ptr,
                                                      bw_filter_workspace.size,
                                                      (void*) &beta,
                                                      kernel_descriptor,
                                                      dw.ptr()) );

          beta = xc->accumulate_grad();

          CHECK_CUDNN( cudnnConvolutionBackwardData(cudnn_handle,
                                                    (void*) &alpha,
                                                    kernel_descriptor,
                                                    w.ptr(),
                                                    dy.descriptor(),
                                                    dy.ptr(),
                                                    this->conv_descriptor,
                                                    algo_descriptor_data_bw,
                                                    bw_data_workspace.ptr,
                                                    bw_filter_workspace.size,
                                                    (void*) &beta,
                                                    dx.descriptor(),
                                                    dx.ptr()) );

        };

        STREAM_BACKWARD(gpu, stream_id).put(bwd, xc, yc);

      }

    }

    void _backward(std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> xc,
                   std::shared_ptr<nn::TensorContainer<gpu, stream_id, T>> yc){}

  };

#endif

template<index_t stream_id, typename T>
class Convolution2d<cpu, stream_id, T>: public Convolution2dBase<cpu, stream_id, T>{

  public:

    Convolution2d(index_t channels_in, index_t channels_out,
                      index_t kernel_height, index_t kernel_width,
                      index_t stride_vertical, index_t stride_horizontal,
                      conv_padding padding = NN_CONV_PADDING_SAME,
                      index_t dilation_height = 1, index_t dilation_width = 1,
                      bool requires_grad = true):
      Convolution2dBase<gpu, stream_id, T>(channels_in, channels_out,
                                           kernel_height, kernel_width,
                                           stride_vertical, stride_horizontal,
                                           padding, dilation_height)
     {
     }


};


}


#endif
