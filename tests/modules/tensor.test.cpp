#include <catch.hpp>

#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/expression-inl.h>
#include <fundamental/tensor-inl.h>

#include <fundamental/linalg/matmul.h>

#include <fundamental/initializer.h>

#include <ops/transform.h>

using namespace core;
using namespace expression;
using namespace initilize;
using namespace transform;

#ifdef TENSOR_USE_CUDA

  TEST_CASE("Tensor - transfered to GPU/CUDA and back", "[tensor]"){

    float val = 1;
    bool tensor_transfer = true;
    Tensor<cpu> tensor_cpu(54, 2, 2, 3);
    tensor_cpu.allocate();

    for (auto p = tensor_cpu.begin(); p < tensor_cpu.end(); ++p)
      *p = val;

    Tensor<gpu> tensor_gpu = to_cuda(tensor_cpu);
    Tensor<cpu> tensor_cpu2 = to_cpu(tensor_gpu);

    for (auto p = tensor_cpu2.begin(); p < tensor_cpu2.end(); ++p)
      if ((*p) != val) tensor_transfer = false;

    REQUIRE(tensor_transfer);
  }

#endif

TEST_CASE("Tensor - intilized ", "[tensor]"){

  Shape shape(510, 10, 10, 1);
  float val = 1;

  SECTION("CPU"){ using DEVICE = cpu;
    bool tensor_intilize = true;
    Tensor<DEVICE> tensor(shape);
    tensor.allocate();
    tensor = Initilizer< Constant<DEVICE, float> >(shape, val);
    for (auto p = tensor.begin(); p < tensor.end(); ++p)
      if ((*p) != val) tensor_intilize = false;
    REQUIRE(tensor_intilize);
  }

  #ifdef TENSOR_USE_CUDA
    SECTION("GPU"){  using DEVICE = gpu;
      bool tensor_intilize = true;
      Tensor<gpu> tensor(shape);
      tensor.allocate();
      tensor = Initilizer< Constant<DEVICE, float> >(shape, val);
      Tensor<cpu> tensor_cpu = to_cpu(tensor);
      for (auto p = tensor_cpu.begin(); p < tensor_cpu.end(); ++p)
        if ((*p) != val) tensor_intilize = false;
      REQUIRE(tensor_intilize);
    }
  #endif

}

TEST_CASE("Tensor - Gaussian ", "[tensor]"){

  Shape shape(510, 100, 10, 1);
  float val = 1;
  float threshold = val*0.2;

  SECTION("CPU"){ using DEVICE = cpu;
    Tensor<DEVICE> tensor(shape);
    tensor.allocate();
    tensor = Initilizer<Gaussian<DEVICE, DEFAULT_STREAM, float> >(shape, val, 2); // mu, std
    double mean = 0;
    for (auto p = tensor.begin(); p < tensor.end(); ++p)
      mean += (*p);
    mean /= shape.size();
    REQUIRE(std::abs(mean - val) < threshold);
  }

  #ifdef TENSOR_USE_CUDA
    SECTION("GPU"){ using DEVICE = gpu;
      Tensor<DEVICE> tensor_gpu(shape);
      tensor_gpu.allocate();
      tensor_gpu = Initilizer<Gaussian<DEVICE, DEFAULT_STREAM, float> >(shape, val, 2); // mu, std
      Tensor<cpu> tensor_cpu = to_cpu(tensor_gpu);
      double mean = 0;
      for (auto p = tensor_cpu.begin(); p < tensor_cpu.end(); ++p)
        mean += (*p);
      mean /= shape.size();
      REQUIRE(std::abs(mean - val) < threshold);
    }
  #endif

}

TEST_CASE("Tensor - Multi Expression ", "[tensor]"){

  float a = 1.1;
  float b = 2.102;
  float c = 31.21;
  float d = 31.21;
  float s = 1.2;


  Shape shape(510, 10, 10, 1);

  float res = s + (a + b - c * d) / a;;
  float tol = 0.01;

  SECTION("CPU"){ using DEVICE = cpu;

    bool pass = true;

    Tensor<DEVICE> tensorA(shape); tensorA.allocate();
    Tensor<DEVICE> tensorB(shape); tensorB.allocate();
    Tensor<DEVICE> tensorC(shape); tensorC.allocate();
    Tensor<DEVICE> tensorD(shape); tensorD.allocate();

    tensorA = Initilizer< Constant<DEVICE, float> >(shape, a);
    tensorB = Initilizer< Constant<DEVICE, float> >(shape, b);
    tensorC = Initilizer< Constant<DEVICE, float> >(shape, c);
    tensorD = Initilizer< Constant<DEVICE, float> >(shape, d);

    Scalar<DEVICE> scalar(s);

    auto tensorRes = ( scalar + (tensorA + tensorB - tensorC*tensorD) / tensorA).eval();

    for (auto p = tensorRes.begin(); p < tensorRes.end(); ++p)
      if ( std::abs((*p) -res) > tol)  pass = false;

    REQUIRE(pass);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      bool pass = true;

      Tensor<DEVICE> tensorA(shape); tensorA.allocate();
      Tensor<DEVICE> tensorB(shape); tensorB.allocate();
      Tensor<DEVICE> tensorC(shape); tensorC.allocate();
      Tensor<DEVICE> tensorD(shape); tensorD.allocate();

      tensorA = Initilizer< Constant<DEVICE, float> >(shape, a);
      tensorB = Initilizer< Constant<DEVICE, float> >(shape, b);
      tensorC = Initilizer< Constant<DEVICE, float> >(shape, c);
      tensorD = Initilizer< Constant<DEVICE, float> >(shape, d);

      Scalar<DEVICE> scalar(s);

      auto tensorRes = (scalar + (tensorA + tensorB - tensorC*tensorD) / tensorA).eval();

      Tensor<cpu> tensor_cpu = to_cpu(tensorRes);

      for (auto p = tensor_cpu.begin(); p < tensor_cpu.end(); ++p)
        if ( std::abs((*p) -res) > tol)  pass = false;

      REQUIRE(pass);

    }

  #endif

}

TEST_CASE("Tensor - Single Expression  ", "[tensor]"){

  float a = 1.1;

  Shape shape(510, 10, 10, 1);

  float tol = 0.01;

  struct Test{
    TENSOR_INLINE static float Eval(float v) { return 2.2;}
  };

  float res = 2.2;

  SECTION("CPU"){ using DEVICE = cpu;

    bool pass = true;

    Tensor<DEVICE> tensorA(shape); tensorA.allocate();

    tensorA = Initilizer< Constant<DEVICE, float> >(shape, a);

    auto tensorRes = F<Test>(tensorA).eval();

    for (auto p = tensorRes.begin(); p < tensorRes.end(); ++p)
      if ( std::abs((*p) -res) > tol)  pass = false;

    REQUIRE(pass);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu; }

  #endif

}

TEST_CASE("Tensor - save/load ", "[tensor]"){

  int dim = 4;

  index_t _shape [SHAPE_MAX_DIM] = {510, 10, 19, 12};
  index_t _shape2 [SHAPE_MAX_DIM] = {10, 10, 19, 12};

  Shape shape(_shape, dim);
  Shape shape2(_shape2, dim);

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensor(shape); tensor.allocate();
    Tensor<DEVICE> tensor_loaded(shape); tensor_loaded.allocate();

    tensor = Initilizer<Constant<DEVICE, float> >(shape, 0);

    tensor.set({150, 2, 9, 6}, 1);

    tensor.write("test.dat");
    tensor_loaded.read("test.dat");

    index_t result [SHAPE_MAX_DIM] = {150, 2, 9, 6};

    index_t deduced_shape[SHAPE_MAX_DIM];
    for (int i = 0; i < shape.size();  ++i){
      if (tensor_loaded.ptr()[i] != 0){
        tensor_loaded.shape().deduce_shape(i, deduced_shape);
        for (int ii = 0; ii < dim; ++ii){
          REQUIRE (result[ii] == deduced_shape[ii]);
          found = true;
        }
      }
    }

    REQUIRE (found);

  }

  #ifdef TENSOR_USE_CUDA
    SECTION("GPU"){  }
  #endif

}

TEST_CASE("Tensor - cast ", "[tensor]"){

  int dim = 4;

  index_t _shape [SHAPE_MAX_DIM] = {510, 10, 19, 12};

  Shape shape(_shape, dim);

  SECTION("CPU"){ using DEVICE = cpu;

    Tensor<DEVICE, 0, float> tensor(shape); tensor.allocate();
    tensor = Initilizer<Constant<DEVICE, float> >(shape, 2.6);
    Tensor<DEVICE, 0, uint8_t> tensor8(shape); tensor8.allocate();

    tensor8 = cast<uint8_t>(tensor);

    for (int i = 0; i < shape.size();  ++i)
      REQUIRE (tensor8.ptr()[i] == 2);

  }

  #ifdef TENSOR_USE_CUDA
    SECTION("GPU"){ using DEVICE = gpu;

      Tensor<DEVICE, 0, float> tensor(shape); tensor.allocate();
      tensor = Initilizer<Constant<DEVICE, float> >(shape, 2.6);
      Tensor<DEVICE, 0, uint8_t> tensor8(shape); tensor8.allocate();

      tensor8 = cast<uint8_t>(tensor);

      Tensor<cpu, 0, uint8_t> tensor_cpu = to_cpu(tensor8);

      for (int i = 0; i < shape.size();  ++i)
        REQUIRE (tensor_cpu.ptr()[i] == 2);

    }

  #endif

}
