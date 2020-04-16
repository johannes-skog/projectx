#ifndef NN_CONTAINER_H
#define NN_CONTAINER_H

#include <fundamental/tensor.h>
#include <fundamental/initializer.h>

namespace nn{

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  class TensorContainer: public expression::Exp<xpu, stream_id, TensorContainer<xpu, stream_id, T>, T,
                                expression::type::kLvalue>{

      const bool _require_grad;

      const bool _trainable;

      const T _alpha;

      core::Shape _shape;

      core::Blob<xpu, stream_id, T> dblob;

      core::Blob<xpu, stream_id, T> gblob;

      std::vector< std::function<void(void)> > cleanup;

      std::string desc;

      void _set_shape(const core::Shape& s){
        _shape = s;
      }

    public:

      TensorContainer(bool __require_grad = true,
                      bool __trainable = false,
                      bool __accumulate_grad = true):
        _require_grad{__require_grad}, _trainable{__trainable},
        _alpha{(T)__accumulate_grad}, _shape{0} {};

      TensorContainer(core::Shape __shape,
                      bool __require_grad = true,
                      bool __trainable = false,
                      bool __accumulate_grad = true):
        _require_grad{__require_grad}, _trainable{__trainable},
        _alpha{(T)__accumulate_grad}, _shape{0}{
        conditional_override(__shape);
      };

      TensorContainer(const TensorContainer& src):
        _require_grad{src._require_grad},
        _alpha{src._alpha},
        _shape{src._shape}, dblob{src.dblob},
         gblob{src.gblob}, cleanup{}, // We should not copy the cleanup, otherwise we will delete
         // the org tensorCont.
         _trainable{src._trainable} {}

      ~TensorContainer(){
        for (std::function<void(void)>& f : cleanup) f();
      }

      void add_cleanup(std::function<void(void)> f){
        cleanup.push_back(f);
      }

      void set_shape(const core::Shape& s){
        DEBUG_ASSERT (shape().size() == s.size());
        _shape = s;
       }

      template<typename E, int ET>
      TensorContainer<xpu, stream_id, T>&
      operator=(const expression::Exp<xpu, stream_id, E, T, ET>& e){

        conditional_override(e.self().shape());

        expression::Exceturer<xpu>::excetute(e, *this);

        if (_require_grad)
         expression::Exceturer<xpu>::backward(*this, e);

        return *this;

      }

      void set(core::Shape s){

        DEBUG_ASSERT(!dblob.check_data_allocation());

        if ( _require_grad )
          DEBUG_ASSERT(!gblob.check_data_allocation());

        conditional_override(s);

      }

      T accumulate_grad() {return _alpha;}

      void conditional_override(core::Shape s){

        index_t N = s.size();

        if ( dblob.size() != N  && N > 0){
          dblob.update(N); dblob.allocate();
          _set_shape(s);
          if ( _require_grad ){
            gblob.update(N); gblob.allocate();
            auto t = gradient();
            expression::initilize::zeros(t);
          }
        }

      }

      core::Tensor<xpu, stream_id, T> data(){
        return core::Tensor<xpu, stream_id, T>(dblob, _shape);
      }

      core::Tensor<xpu, stream_id, T> gradient(){
        return core::Tensor<xpu, stream_id, T>(gblob, _shape);
      }

      core::Blob<xpu, stream_id, T>& data_blob(){
        return dblob;
      }

      core::Blob<xpu, stream_id, T>& gradient_blob(){
        return gblob;
      }

      bool require_grad(void){
        return _require_grad;
      }

      const core::Shape& shape(void) const{
       return _shape;
      }

      TENSOR_INLINE T Eval(index_t s, index_t i) const{
        return dblob.Eval(s*_shape.stride + i);
      }

      TENSOR_INLINE void Set(index_t s, index_t i, T v){
        dblob.Set(s*_shape.stride + i, v);
      }

      TENSOR_INLINE T BackwardEval(index_t s, index_t i){
        DEBUG_ASSERT(_require_grad);
        return gblob.Eval(s*_shape.stride + i);
      }

      TENSOR_INLINE void Backward(index_t s, index_t i, T v){
        DEBUG_ASSERT(_require_grad);
        gblob.Set(s*_shape.stride + i, BackwardEval(s, i) * _alpha + v);
      }

  };

  template<typename xpu, index_t stream_id, typename T>
  class _Factory{

    using Content = nn::TensorContainer<xpu, stream_id, T>;

    using Identifier = std::string;
    using SharedContent = std::shared_ptr<Content>;

    std::map<Identifier, Content*> collection;

    bool _exists(Identifier id){
      if (collection.find(id) != collection.end())
       return true;
      else return false;
    }

    Content* _get(Identifier id) {
      ASSERT_ERROR(this->_exists(id), id);
      return this->collection.at(id);
    }

    void _remove(Identifier id){
      ASSERT_ERROR(this->collection.find(id) != this->collection.end(), id);
      this->collection.erase(id);
    }

    Identifier _pad(Identifier specific){
      return SCOPE.full() + TAG_SEPERATOR + specific;
    }

    public:

      _Factory(){};

      template <typename... Args>
      SharedContent create(Identifier specific, Args&&... args){
        SharedContent content =
                      std::make_shared<Content>(std::forward<Args>(args)...);
        this->add(specific, content);
        return content;
      }

      void add(Identifier specific, SharedContent content){
        Identifier id;
        if (content->require_grad())
          id =  DIFFERENTIABLE + TAG_SEPERATOR + specific;
        else
          id = specific;
        Identifier id_full = _pad(id);
        DEBUG_ASSERT(!this->_exists(id_full));
        content->add_cleanup([this, id_full](){  this->_remove(id_full);  });
        collection.insert({id_full, content.get()});
      }

      bool exists(Identifier id, bool pad = true){
        if (pad) id = _pad(id);
        return _exists(id);
      }

      Content* get(Identifier id, bool pad = true){
        if (pad) id = _pad(id);
        return _get(id);
      }

      std::map<Identifier, Content*> get_all(){
        return collection;
      }

      void remove_all(void){
         collection.clear();
       }

      std::map<Identifier, Content*> search(Identifier id){
        std::regex re(id);
        std::map<Identifier, Content*> allowed;
        for (auto const& c: this->collection)
          if (std::regex_search(c.first, re)) allowed.insert({c.first, c.second});
        return allowed;
      }

  };

  //#define ContainerContext(x, s, t) Context<_Factory<x, s, t > >::Instance()

  //#define ContainerFactoryF(x, s) Context<_Factory<x, s, DEFAULT_TYPE > >::Instance()

  template<typename xpu, index_t stream_id = DEFAULT_STREAM,
           typename T = DEFAULT_TYPE, typename... Args>
  std::shared_ptr<TensorContainer<xpu, stream_id, T>>
                  ContainerFactory(std::string s, Args&&... values){
    return Context<_Factory<xpu, stream_id, T > >::Instance().create(s,
                   std::forward<Args>(values)...);
  }

  template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
  _Factory<xpu, stream_id, T >& ContainerContext(){
    return Context<_Factory<xpu, stream_id, T > >::Instance();
  }

  template<typename xpu, index_t stream_id, typename T>
  std::shared_ptr<TensorContainer<xpu, stream_id, T>>
  view(std::shared_ptr<TensorContainer<xpu, stream_id, T>> src, const core::Shape s){

    DEBUG_ASSERT (s.size() == src->shape().size());

    auto tc = ContainerFactory<xpu, stream_id, T>("view", *(src.get()));
    tc->set_shape(s);

    return tc;

  }

  template<typename xpu, index_t stream_id, typename T>
  TensorContainer<xpu, stream_id, T>
  view(TensorContainer<xpu, stream_id, T>& src, const core::Shape s){

    DEBUG_ASSERT (s.size() == src.shape().size());

    TensorContainer<xpu, stream_id, T> tc(src);
    tc.set_shape(s);
    return tc;

  }

}


#endif
