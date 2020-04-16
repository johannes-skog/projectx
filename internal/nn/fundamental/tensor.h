#ifndef FUNDAMENTAL_TENSOR_H
#define FUNDAMENTAL_TENSOR_H

#include <fundamental/expression.h>
#include <fundamental/macro.h>
#include <fundamental/shape.h>
#include <fundamental/storage.h>
#include <string>

namespace memory{

  template<typename xpu, index_t stream_id, typename T>
  class Memory{
    TENSOR_INLINE_HOST static T* allocate(const size_t);
    TENSOR_INLINE_HOST static void deallocate(T*);
    TENSOR_INLINE_HOST static void set(T*, T);
  };

  template<index_t stream_id, typename T>
  struct Memory<cpu, stream_id, T>{
    TENSOR_INLINE_HOST static T* allocate(const size_t);
    TENSOR_INLINE_HOST static void deallocate(T*);
    TENSOR_INLINE_HOST static void set(T*, T);
    TENSOR_INLINE_HOST static T at(T*);
    TENSOR_INLINE_HOST static void memcpy(T*, const T*, size_t N);
  };

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    struct Memory<gpu, stream_id, T>{
      TENSOR_INLINE_HOST static T* allocate(const size_t);
      TENSOR_INLINE_HOST static void deallocate(T*);
      TENSOR_INLINE_HOST static void set(T*, T);
      TENSOR_INLINE_HOST static T at(T*);
      TENSOR_INLINE_HOST static void memcpy(T*, const T*, size_t N);
      TENSOR_INLINE_HOST static void to_device(T*, const T*, size_t N);
      TENSOR_INLINE_HOST static void to_host(T*, const T*, size_t N);
    };

  #endif

  template<typename T, typename xpu, index_t stream_id>
  TENSOR_INLINE_HOST void set(T*, T);

}

namespace core{

    template<typename xpu, typename T>
    class Descriptor{

      public:

        Descriptor(){}

        Descriptor(const core::Shape&){}

        void get_descriptor(void) const{}

        void deallocate(void){}

        void operator=(const Descriptor&){}

    };

    template <typename xpu, typename T = DEFAULT_TYPE, index_t stream_id = DEFAULT_STREAM>
    class Scalar: public expression::Exp<xpu, stream_id, Scalar<xpu, T, stream_id>, T,
                         expression::type::kscalar>{

      const Shape _shape{1};

      public:

        T v;

        Scalar(T v): v{v} {}

        TENSOR_INLINE T Eval(index_t, index_t) const{
          return v;
        }

        TENSOR_INLINE void Backward(index_t, index_t, T){}

        TENSOR_INLINE const Shape& shape(void) const {return _shape;}

    };

    template <typename xpu, typename T, index_t stream_id = DEFAULT_STREAM>
    class Constant: public expression::Exp<xpu, stream_id, Constant<xpu, T, stream_id>, T,
                           expression::type::kRvalue>{

      public:

        T v;

        const Shape _shape;

        Constant(T v, const Shape& s): v{v}, _shape{s} {}

        TENSOR_INLINE T Eval(index_t, index_t) const{
          return v;
        }

        TENSOR_INLINE T BackwardEval(index_t, index_t){
          return v;
        }

        TENSOR_INLINE void Backward(index_t, index_t, T){}

        TENSOR_INLINE const Shape& shape(void) const {return _shape;}

    };

    template<typename xpu, index_t stream_id = DEFAULT_STREAM, typename T = DEFAULT_TYPE>
    class Blob{

       T* dataptr;

       index_t* counter;

       index_t N;

       void new_count(index_t v = 0){
         counter = new index_t;
         (*counter) = v;
       }

       const bool owner;

      public:

        Blob(index_t N, T* _dataptr, bool _owner = true): owner{_owner},
                                                          dataptr{_dataptr},
                                                          counter{nullptr}, N{N}{
          new_count(1);
        }

        Blob(index_t N): owner{true}, dataptr{nullptr}, counter{nullptr}, N{N}{}

        Blob(void): owner{true}, dataptr{nullptr}, counter{nullptr}, N{0} {}

        Blob(const Blob& src): dataptr{src.dataptr}, N{src.N},
                               counter{src.counter}, owner{src.owner}{
          increment();
        }

        ~Blob(void){
           decrement();
        }

        TENSOR_INLINE index_t size(void) const{
          return this->N;
        }

        index_t count(void) const{
          DEBUG_ASSERT(POINTER_CHECK(counter));
          return (*counter);
        }

        void update(const Blob& src){
          decrement();
          this->dataptr = src.dataptr;
          this->N = src.N;
          this->counter = src.counter;
          increment();
        }

        void update(index_t _N){
          decrement();
          dataptr = nullptr; counter = nullptr;
          this->N = _N;
        }

        void copy(const Blob& src){
          DEBUG_ASSERT(size() == src.size());
          allocate();
          memory::Memory<xpu, stream_id, T>::memcpy(ptr(), src.ptr(), size());
        }

        void allocate(void){
          decrement();
          dataptr = memory::Memory<xpu, stream_id, T>::allocate(size());
          new_count(1);
        }

        bool check_data_allocation(){
          return POINTER_CHECK(dataptr);
        }

        void deallocate(void){

          if (POINTER_CHECK(dataptr) && this->owner){
            memory::Memory<xpu, stream_id, T>::deallocate(dataptr);
            dataptr = nullptr; N = 0;
          }

          if (POINTER_CHECK(counter)){
            delete counter;
            counter = nullptr;
          }

          N = 0;

        }

        void increment(void){
          if (POINTER_CHECK(counter))
            (*counter) += 1;
        }

        void decrement(void){
          if (POINTER_CHECK(counter)){
            (*counter) -= 1;
            if ( count() <= 0)
              deallocate();
          }
        }

        TENSOR_INLINE T* ptr(void){
          return dataptr;
        }

        TENSOR_INLINE const T* ptr(void) const{
          return dataptr;
        }

        TENSOR_INLINE T Eval(index_t i) const{
          DEBUG_ASSERT(i < N);
          return dataptr[i];
        }

        TENSOR_INLINE void Set(index_t i, T v){
          DEBUG_ASSERT(i < size());
          dataptr[i] = v;
        }

        T at(index_t i) const{
          return memory::Memory<xpu, stream_id, T>::at(dataptr + i);
        }

        void set(index_t i, T v){
          DEBUG_ASSERT(i < size());
          memory::Memory<xpu, stream_id, T>::set(dataptr + i, v);
        }

    };

    template<typename xpu, index_t stream_id = DEFAULT_STREAM,
             typename T = DEFAULT_TYPE>
    class Tensor: public expression::Exp<xpu, stream_id,
                core::Tensor<xpu, stream_id, T>, T, expression::type::kLvalue> {

    TENSOR_INLINE_HOST void initilize(void){
      blob.update(shape().size());
    }

    Shape _shape;

    public:

      Descriptor<xpu, T> _descriptor;

      Blob<xpu, stream_id, T> blob;

      std::string tags;

      Tensor(): _shape{0}, _descriptor{} {}

      MULTI_ARG_INDEX Tensor(T* ptr, Args&&... args):
        _shape{std::forward<Args>(args)...},
        _descriptor{_shape}{
        initilize();
      }

      MULTI_ARG_INDEX Tensor(Args&&... args):
        _shape{std::forward<Args>(args)...},
        _descriptor{_shape}{
        initilize();
       }

      Tensor(const core::Blob<xpu, stream_id, T>& _blob, const core::Shape& _shape):
        _shape{_shape},
        blob{_blob},
        _descriptor{_shape}{}

      Tensor(const Tensor& src):
        _shape{src._shape},
        blob{src.blob},
        _descriptor{src._descriptor}{}

      void update(const Tensor& src){
        _shape = src._shape;
        blob.update(src.blob);
        _descriptor = src._descriptor;
      }

      Tensor(const Tensor& src, const Shape& s):
        _shape{s},
        blob{src.blob},
        _descriptor{src._descriptor}{
        DEBUG_ASSERT(s.size() == src.shape().size());
      }

      Tensor(const Shape& s):
        _shape{s},
        _descriptor{s}{
        initilize();
      }

      Tensor(T* ptr, const Shape& s):
        blob{s.size(), ptr, false},
        _shape{s},
        _descriptor{s}{}

      TENSOR_INLINE_HOST ~Tensor(void) {}

      template<typename E, index_t exp_type>
      TENSOR_INLINE_HOST Tensor<xpu, stream_id, T>&
          operator=(const expression::Exp<xpu, stream_id, E, T, exp_type>& src){
        expression::Exceturer<xpu>::excetute(src, *this);
        return *this;
      }

      TENSOR_INLINE_HOST void copy_data(const Tensor& src){
        blob.copy(src.blob);
      }

      TENSOR_XXXX view(const core::Shape s) const{
        DEBUG_ASSERT(s.size() == shape().size());
        TENSOR_XXXX tensor(*this, s);
        return tensor;
      }

      TENSOR_XXXX flatten() const{
        TENSOR_XXXX tensor(*this, this->shape().flatten());
        return tensor;
      }

      void tag(std::string s){
         tags.append( TAG_SEPERATOR + s );
       }

      void tag(unique_id id){
         tags.append( TAG_SEPERATOR + std::to_string(id) );
      }

      decltype(auto) descriptor(void){
        return _descriptor.get_descriptor();
      }

      TENSOR_INLINE T* ptr(void){
        return blob.ptr();
      }

      TENSOR_INLINE const T* ptr(void) const{
        return blob.ptr();
      }

      TENSOR_INLINE T* begin(void){
        return ptr();
      }

      TENSOR_INLINE T* end(void){
        return (ptr() + shape().size() - 1);
      }

      TENSOR_INLINE const Shape& shape(void) const{
        return _shape;
      }

      TENSOR_INLINE const T scalar(void){
        DEBUG_ASSERT( this->shape().size() == 1 );
        return this->at(0);
      }

      TENSOR_INLINE_HOST T at(std::initializer_list<index_t> values){
        index_t values_a[SHAPE_MAX_DIM];
        // TODO, send the initializer_list directly to shape
        std::copy(values.begin(), values.end(), values_a);
        return at(values_a);
      }

      TENSOR_INLINE_HOST T at(index_t i){
        return blob.at(i);
      }

      TENSOR_INLINE_HOST T at(index_t values[SHAPE_MAX_DIM]){
        index_t s, i;
        this->shape().index(s, i, values);
        return at(shape().stride * s +  i);
      }

      MULTI_ARG_INDEX TENSOR_INLINE_HOST T at(Args&&... values){
        index_t s, i;
        this->shape().index(s, i, std::forward<Args>(values)...);
        return at(shape().stride * s +  i);
      }

      TENSOR_INLINE T Eval(index_t s, index_t i) const{
        return blob.Eval(s*shape().stride + i);
      }

      TENSOR_INLINE void Set(index_t i, T v){
        blob.Set(i, v);
      }

      TENSOR_INLINE void Set(index_t s, index_t i, T v){
        Set(s*shape().stride + i, v);
      }

      TENSOR_INLINE void Backward(index_t, index_t, T){}

      TENSOR_INLINE_HOST void set(index_t i,  T v){
         blob.set(i, v);
      }

      TENSOR_INLINE_HOST void set(std::initializer_list<index_t> values,T v){
        // TODO, send the initializer_list directly to shape
        index_t values_a[SHAPE_MAX_DIM];
        std::copy(values.begin(), values.end(), values_a);
        index_t s, i;
        this->shape().index(s, i, values_a);
        blob.set(s*shape().stride +  i, v);
      }

      TENSOR_INLINE_HOST void deallocate(void){
        blob.deallocate();
      }

      TENSOR_INLINE_HOST void allocate(void){
        blob.allocate();
      }

      TENSOR_INLINE_HOST bool check_allocation(void){
        if (POINTER_CHECK(blob.ptr())) return true;
        return false;
      }

      TENSOR_INLINE_HOST void write(Storage<T, std::ios::out>& out){
         store(out, *this);
      }

      TENSOR_INLINE_HOST void write(std::string filename){
        Storage<T, std::ios::out> out; out.open(filename);
        store(out, *this);
      }

      TENSOR_INLINE_HOST void read(Storage<T, std::ios::in>& in){
         load(in, *this);
      }

      TENSOR_INLINE_HOST void read(std::string filename){
         Storage<T, std::ios::in> in; in.open(filename);
         DEBUG_ASSERT( (in.size() == shape().size()*sizeof(T)) );
         load(in, *this);
      }

  };

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    Tensor<gpu, stream_id, T> to_cuda(Tensor<cpu, stream_id, T>&);

    template<index_t stream_id, typename T>
    void to_cuda(Tensor<gpu, stream_id, T>&,
                 Tensor<cpu, stream_id, T>&);

    template<index_t stream_id, typename T>
    Tensor<cpu, stream_id, T> to_cpu(Tensor<gpu, stream_id, T>&);

    template<index_t stream_id, typename T>
    void to_cpu(Tensor<cpu, stream_id, T>&, Tensor<gpu, stream_id, T>&);

    template<index_t stream_id, typename T>
    Tensor<gpu, stream_id, T> copy(Tensor<gpu, stream_id, T>&);

  #endif

  template<index_t stream_id, typename T>
  Tensor<cpu, stream_id, T> copy(Tensor<cpu, stream_id, T>&);

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    void store(Storage<T, std::ios::out >&, Tensor<gpu, stream_id, T>&);

  #endif

  template<typename xpu, index_t stream_id, typename T>
  std::vector<core::Tensor<xpu, stream_id, T>>
  copy(std::vector<core::Tensor<xpu, stream_id, T>>);

  template<index_t stream_id, typename T>
  void store(Storage<T, std::ios::out >&, Tensor<cpu, stream_id, T>&);

  #ifdef TENSOR_USE_CUDA

    template<index_t stream_id, typename T>
    void load(Storage<T,  std::ios::in >&, Tensor<gpu, stream_id, T>&);

  #endif

  template<index_t stream_id, typename T>
  void load(Storage<T, std::ios::in >&,
            Tensor<cpu, stream_id, T>&);

  template<typename xpu, index_t stream_id, typename T>
  std::vector<core::Tensor<xpu, stream_id, T>>
  split(core::Tensor<xpu, stream_id, T>& x){

    core::Shape tshape = x.shape().offset_shape(1);

    size_t stride = tshape.size() - 1;

    std::vector<core::Tensor<xpu, stream_id, T>> vt;

    int N = x.shape()[0];

    vt.reserve(N);

    for (int i = 0; i < N; ++i)
      vt.emplace_back(x.ptr() + i * stride, tshape);

    return vt;

  }

}

#ifdef TENSOR_USE_CUDA

  #include <fundamental/cuda/data.cuh>

#endif

#endif
