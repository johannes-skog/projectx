#ifndef FUNDAMENTAL_MACRO_H
#define FUNDAMENTAL_MACRO_H

#include <assert.h>
#include "string.h"
#include <regex>
#include <unordered_map>
#include <queue>

using index_t = int;

using indices_t = int32_t;

using unique_id = unsigned long;

#define DEBUG(x) x

#define DEBUG_ASSERT(x) assert(x)

//#define DEBUG_ASSERT(x)

#define CUDA_MAX_THREADS 512

#define LOG_LEVEL 5

#define SHAPE_MAX_DIM 6

using DEFAULT_TYPE = float;

static const index_t DEFAULT_STREAM = 0;

static const std::string TAG_SEPERATOR = "/";

static const std::string TRAINABLE = "Trainable";

static const std::string DIFFERENTIABLE = "Differentiable";

struct ERROR{
  static const int level = 1;
  static const std::string description_lvl;
};

struct WARNING{
  static const int level = 2;
  static const std::string description_lvl;
};

struct INFO{
  static const int level = 3;
  static const std::string description_lvl;
};

#define CUDA_CHECK_ERROR_ENFORCE(ee) {                   \
 if(ee != cudaSuccess) {                \
   printf("CUDA ERROR %s:%d:  %s \n",                   \
          __FILE__,__LINE__, cudaGetErrorString( ee ));   \
   std::exit(EXIT_FAILURE);                             \
 }                                                      \
}while(0)

#define CUDBLAS_CHECK_ERROR_ENFORCE(ee) {                   \
 if(ee != 0) {                \
   printf("CUDA ERROR %s:%d:  %d \n",                   \
          __FILE__,__LINE__, ee);   \
   std::exit(EXIT_FAILURE);                             \
 }                                                      \
}while(0)

#define POINTER_CHECK_ENFORCE(ptr){                    \
  if (ptr == nullptr && ptr == NULL){                  \
    exit(0);                                           \
  }                                                    \
}

#define CHECK_CUDNN(e)                               \
  {                                                          \
    cudnnStatus_t status = (e);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      printf("CUDNN ERROR %s:%d: '%s' %d \n",              \
             __FILE__, __LINE__, cudnnGetErrorString(status), e); \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define POINTER_CHECK(ptr)                             \
          (ptr != nullptr && ptr != NULL)              \

#define LOG_INFO(ss) {                                 \
    printf("%s::%s \n", "INFO", ss);                   \
}while(0)


#define ASSERT_ERROR(e, ss) {                             \
    if (e == 0){                                       \
    printf("%s::%s:%d::%s \n", "ERROR", __FILE__,      \
           __LINE__, ss);                              \
    exit(0);                                           \
    }                                                  \
}while(0)


#define MULTI_ARG_FIX(x) template <typename... Args,   \
      typename = typename std::enable_if<              \
          sizeof...(Args) == x>::type>                 \


#define MULTI_ARG_INDEX template<typename... Args,     \
                        std::enable_if_t<all_same<index_t, \
                        Args...>::value>* = nullptr>   \

#define MULTI_ARG template <typename... Args>

#define EQUAL_ENFORCE(x, y ){                          \
  if (x !=y){                                          \
     LOG_ERROR("Not equal")                            \
     exit(0);                                          \
  }                                                    \
}while(0)

#define TENSOR_INLINE  __forceinline__ __attribute__((always_inline)) \
                       __device__ __host__

#define TENSOR_INLINE_CUDA  __forceinline__ __attribute__((always_inline)) \
                      __device__

#define TENSOR_INLINE_HOST  __forceinline__ __attribute__((always_inline)) \

struct cpu{
  static const std::string description_ctx;
  static const bool _cpu = true;
};

struct gpu{
  static const std::string description_ctx;
  static const bool _cpu = false;
};


template<typename ctx>
class Context{ // For contexts, ensure that we only have one instance
  static ctx* instance;
  Context(){};
  public:
    static ctx& Instance(){
       if (!instance)
       instance = new ctx;
       return *instance;
    }
};

template<typename ctx>
ctx* Context<ctx>::instance = nullptr;
#define CONTEXT(x) Context<x>::Instance()

template<typename Content>
class UniqueName{
  unique_id id;
  public:
    UniqueName(void): id{0} {}
    unique_id next(void){id++; return id;}
};

class Scope{

  class _Scope{

    unique_id id;

    std::function<void(unique_id)> cleanup;

    public:

      _Scope(unique_id _id, std::function<void(unique_id)> _cleanup):
             cleanup{_cleanup}, id{_id}{}

      ~_Scope(){ cleanup(id); }

  };

  public:

    std::vector<unique_id> ids;
    std::map<unique_id, std::string> desc;

    std::function<void(unique_id)> cleanup;

    Scope(){

      cleanup = [this](unique_id id){
        auto it = std::find(this->ids.begin(), this->ids.end(), id);
        DEBUG_ASSERT(it != this->ids.end());
        DEBUG_ASSERT(this->desc.find(id) != this->desc.end());
        this->ids.erase(it);
        this->desc.erase(id);
      };

    }

    _Scope with(std::string s){
      unique_id id = CONTEXT(UniqueName< _Scope >).next();
      ids.push_back(id); desc.insert({id, s});
      return _Scope(id, cleanup);
    }

    std::shared_ptr<std::vector<_Scope>> withv(std::vector<std::string> v){
      auto vs = std::make_shared<std::vector<_Scope>>();
      vs->reserve(v.size());
      for (std::string s : v){
        unique_id id = CONTEXT(UniqueName<_Scope>).next();
        ids.push_back(id); desc.insert({id, s});
        vs->emplace_back(id, cleanup);
      }
      return vs;
    }

    std::string full(){
      std::string s = "";
      for (const unique_id id: ids)
        s += TAG_SEPERATOR + desc[id];
      return s;
    }

};

#define SCOPE  Context< Scope >::Instance()


template<typename Content>
class Counter{

  std::map<unique_id, index_t> counter;

  public:

    Counter(void) {}

    index_t count(unique_id id){
      if (counter.find(id) == counter.end()) return 0;
      return counter[id];
    }

    void increment(unique_id id){
      if (counter.find(id) == counter.end()){
        counter[id] = 1;
      } else{
        counter[id] =  counter[id] + 1;
      }
    }

    void decrement(unique_id id){
      if (counter.find(id) != counter.end()){
        counter[id] = counter[id] - 1;
        if (counter[id] <= 0) counter.erase(id);
      }
    }

};

template<typename Content, typename Key>
class CollectionOrganized{

  using ContentMap = std::map<Key, Content*>;

  ContentMap collection;

  public:

    CollectionOrganized(){};

    template <typename... Args>
    Content* Create(Key rel, Args&&... args){
      assert(collection.find(rel) == collection.end());
      Content* content =  new Content(std::forward<Args>(args)...);
      collection[rel] = content;
      return content;
    }

    void Add(Key rel, Content& content){
      assert(collection.find(rel) == collection.end());
      collection[rel] = &content;
    }

    void Add(Key rel, Content* content){
      assert(collection.find(rel) == collection.end());
      collection[rel] = content;
    }

    ContentMap GetAll(void){ return collection; }

    Content* Get(Key rel){
      if (collection.find(rel) == collection.end()) return nullptr;
      return collection[rel];
    }

    void Remove(Key rel){
      if (collection.find(rel) != collection.end())
        collection.erase(rel);
     }

};

template<typename T>
class Seed{

  T seed;

  T inc;

  public:

    Seed(): seed{time(NULL)}, inc{1} {}

    void freeze(){
      inc = 0;
    }

    void set(T _seed){
      seed = _seed;
    }

    T next(){
      seed +=inc;
      return seed;
    }

};


template<typename xpu, index_t stream_id>
struct TensorUnique {};

#define SEED Context<Seed<long int>>::Instance()

#define TENSOR_XXXX core::Tensor<xpu, stream_id, T>

#define TENSOR_XXXX core::Tensor<xpu, stream_id, T>

#define GEN_UNIQUE_ID(x) \
        Context<UniqueName< x > >::Instance().next()

#define TENSOR_GEN_UNIQUE_ID(x, s) \
        Context<UniqueName<TensorUnique< x, s > >>::Instance().next()

#define COUNTER(x, s) \
        Context<Counter< TensorUnique< x, s > > >::Instance()

namespace detail{
  template<bool...> struct bool_pack;
  template<bool... bs>
  //if any are false, they'll be shifted in the second version, so types won't match
  using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
}

template <typename... Ts>
using all_true = detail::all_true<Ts::value...>;

template <typename T, typename... Ts>
using all_same = all_true<std::is_same<T,Ts>...>;

#include <fundamental/stream.h>

#endif
