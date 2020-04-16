#ifndef NN_LAYER_H
#define NN_LAYER_H

#include <fundamental/expression.h>
#include <fundamental/shape.h>
#include <fundamental/tensor.h>
#include <fundamental/macro.h>
#include <fundamental/initializer.h>

#include <nn/container.h>

#include <stdexcept>
#include <map>
#include <vector>

namespace nn{

template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
class Layer{

  using SharedTensor = std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>;

   const bool _requires_grad;

   bool _accumulate_grad;

 protected:

  std::string desc;

  std::map<std::string, SharedTensor> _params;

  SharedTensor register_param(std::string s, bool grad = true, bool accumulate = true){
    //DEBUG_ASSERT( _params.find(s) == _params.end());
    std::vector<std::string> sv = {desc + ":" + std::to_string(id)};
    if (grad) sv.push_back(TRAINABLE);
    auto scope = SCOPE.withv(sv);
    auto t = ContainerFactory<xpu, stream_id, T>(s, grad, accumulate);
    _params.emplace(s, t);
    return t;
  }

  public:

    unique_id id;

    Layer() = default;

    Layer(std::string desc, unique_id _id, bool _requires_grad = true):
          _requires_grad{_requires_grad}, _accumulate_grad{false},
          desc{desc}, id{_id}{}

    virtual ~Layer(){}

    virtual core::Shape shape_output(const core::Shape& in){
      ASSERT_ERROR(0, "Not implemented");
      return in;
    }

    virtual void _forward(SharedTensor xc,
                          SharedTensor yc){
      ASSERT_ERROR(0, "Not implemented");
    }

    virtual void _forward_inplace(SharedTensor xc){
      ASSERT_ERROR(0, "Not implemented");
    }

    virtual void _backward(SharedTensor xc,
                           SharedTensor yc){
      ASSERT_ERROR(0, "Not implemented");
    }

    virtual void _backward_inplace(SharedTensor xc){
      ASSERT_ERROR(0, "Not implemented");
    }

    virtual std::vector<core::Shape> shape_output(std::vector<core::Shape> in){
      DEBUG_ASSERT(in.size() == 1);
      return {shape_output(in[0])};
    }

    void forward(std::vector<SharedTensor> xcv,
                 std::vector<SharedTensor> ycv){
      DEBUG_ASSERT(xcv.size() == 1 && ycv.size() == 1);
      this->forward(xcv[0], ycv[0]);
    }

    virtual void backward(std::vector<SharedTensor> xcv,
                          std::vector<SharedTensor> ycv){
      DEBUG_ASSERT(xcv.size()  == 1 && ycv.size()  == 1);
      backward(xcv[0], ycv[0]);
    }

    virtual void backward_inplace(std::vector<SharedTensor> xcv){
      DEBUG_ASSERT(xcv.size()  == 1);
      backward_inplace(xcv[0]);
    }

    void forward(SharedTensor xc,
                 SharedTensor yc){
       this->_forward(xc, yc);
    }

    void forward_inplace(std::vector<SharedTensor> xcv){
      DEBUG_ASSERT(xcv.size() == 1);
      forward_inplace(xcv[0]);
    }

    void forward_inplace(SharedTensor xc){
      _forward_inplace(xc);
    }

    void backward(SharedTensor xc,
                  SharedTensor yc){}

    void backward_inplace(SharedTensor xc){

    }

    SharedTensor param(std::string s) {
       DEBUG_ASSERT( _params.find(s) != _params.end());
       return _params.at(s);
    }

    std::map<std::string, SharedTensor>& params() {
       return _params;
    }

    bool is_grad(void) const { return _requires_grad; }

    bool accumulate_grad(void) const { return _accumulate_grad; }

    bool accumulate_grad(bool b){
      if (_accumulate_grad && !b) this->zero_grad();
      _accumulate_grad = b;
      return _accumulate_grad;
    }


};

}

#endif
