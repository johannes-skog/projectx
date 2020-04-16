#ifndef NN_ACTIVATION_FUNCS_H
#define NN_ACTIVATION_FUNCS_H

#include <fundamental/macro.h>

namespace nn{ namespace activation{

struct ReLu{

  template<typename xpu>
  struct Forward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v > 0 ? (v) : 0;
    }
  };

  template<typename xpu>
  struct Backward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v > 1 ? v : 0;
    }
  };

};

struct Tanh{

  template<typename xpu>
  struct Forward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v > 0 ? v : 0;
    }
  };

  template<typename xpu>
  struct Backward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v > 1 ? v : 0;
    }
  };

};

struct Linear{

  template<typename xpu>
  struct Forward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v;
    }
  };

  template<typename xpu>
  struct Backward{
    template<typename T>
    TENSOR_INLINE static T Map(T v) {
      return v;
    }
  };

};

}

}




#endif
