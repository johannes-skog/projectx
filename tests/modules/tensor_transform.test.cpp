#include <catch.hpp>

#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/expression-inl.h>
#include <fundamental/tensor-inl.h>

#include <fundamental/initializer.h>

#include <ops/transform.h>

using namespace core;
using namespace expression;
using namespace initilize;
using namespace transform;

TEST_CASE("Tensor - Permute ", "[tensor]"){

  int dim = 4;
  index_t _shape [SHAPE_MAX_DIM] = {510, 10, 19, 2};
  index_t _shape_permuted [SHAPE_MAX_DIM] = {19, 2, 510, 10};

  std::vector<index_t> values = {2, 3, 0, 1};

  Shape shape(_shape, dim);
  Shape shape_permuted(_shape_permuted, dim);

  index_t deduced_shape[SHAPE_MAX_DIM];

  index_t result [SHAPE_MAX_DIM] = {2, 1, 55, 3};


  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);

    tensor.set( {55, 3, 2, 1}, 1);

    auto tensor_transformed = operators::permute(tensor, values);

    for (int i = 0; i < shape.size();  ++i){
      if (tensor_transformed.at(i) == 1){
        tensor_transformed.shape().deduce_shape(i, deduced_shape);
        for (int ii = 0; ii < dim; ++ii){
          REQUIRE (result[ii] == deduced_shape[ii]);
          found = true;
        }
      }
    }

    REQUIRE (found);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      bool found = false;

      Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);
      tensor.set({55, 3, 2, 1}, 1);

      auto tensor_transformed_g = operators::permute(tensor, values);

      auto tensor_transformed = core::to_cpu(tensor_transformed_g);
      for (int i = 0; i < shape.size();  ++i){
        if (tensor_transformed.at(i) == 1){
          tensor_transformed.shape().deduce_shape(i, deduced_shape);
          for (int ii = 0; ii < dim; ++ii){
            REQUIRE (result[ii] == deduced_shape[ii]);
            found = true;
          }
        }
      }

      REQUIRE (found);

     }

  #endif

}

TEST_CASE("Tensor - Concat ", "[tensor]"){

  int dim = 4;

  Shape shape1(510, 10, 19, 2);
  Shape shape2(510, 10, 19, 10);
  Shape shape(510, 10, 19, 12);

  index_t deduced_shape[SHAPE_MAX_DIM];
  index_t result [SHAPE_MAX_DIM] = {1, 9, 2, 7};

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensor1(shape1); tensor1.allocate(); initilize::zeros(tensor1);
    Tensor<DEVICE> tensor2(shape2); tensor2.allocate(); initilize::zeros(tensor2);
    tensor2.set({1, 9, 2, 5}, 1);

    auto tensor = operators::concat(tensor1, tensor2, 3);

    for (int i = 0; i < shape.size();  ++i){
      if (tensor.at(i)  == 1){
        tensor.shape().deduce_shape(i, deduced_shape);
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


TEST_CASE("Tensor - Slice Map ", "[tensor]"){

  int dim = 4;

  Shape shape(510, 10, 19, 300);

  std::vector<index_t> slice = {51, 52, 8, 9, 10, 11, 255, 256};

  index_t deduced_shape[SHAPE_MAX_DIM];
  index_t result [SHAPE_MAX_DIM] = {51, 8, 10, 255};

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);

    operators::slice_map<operators::Test<float>>(tensor, slice);

    for (int i = 0; i < tensor.shape().size();  ++i){
      if (tensor.at(i)  > 0){
        tensor.shape().deduce_shape(i, deduced_shape);
        for (int ii = 0; ii < dim; ++ii){
          REQUIRE (result[ii] == deduced_shape[ii]);
          found = true;
        }
      }
    }

    REQUIRE (found);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      bool found = false;

      Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);

      operators::slice_map<operators::Test<float>>(tensor, slice);

      Tensor<cpu> tensor_cpu = to_cpu(tensor);
      for (int i = 0; i < tensor_cpu.shape().size();  ++i){
        if (tensor_cpu.at(i)  > 0){
          tensor_cpu.shape().deduce_shape(i, deduced_shape);
          for (int ii = 0; ii < dim; ++ii){
            REQUIRE (result[ii] == deduced_shape[ii]);
            found = true;
          }
        }
      }

      REQUIRE (found);

    }

  #endif

}


TEST_CASE("Tensor - Slice ", "[tensor]"){

  Shape shape(510, 10, 19, 300);

  std::vector<index_t> slice = {51, 52,  8, 9, 10, 11, 255, 256};

  index_t deduced_shape[SHAPE_MAX_DIM];
  index_t result [SHAPE_MAX_DIM] = {0, 0, 0, 0};

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);

    tensor.set({51, 8, 10, 255}, 1);

    auto tensor2 = operators::slice(tensor, slice);

    for (int i = 0; i < tensor2.shape().size();  ++i){
      if (tensor2.at(i)  > 0){
        tensor2.shape().deduce_shape(i, deduced_shape);
        for (int ii = 0; ii < tensor2.shape().dim; ++ii){
          REQUIRE (result[ii] == deduced_shape[ii]);
          found = true;
        }
      }
    }

    REQUIRE (found);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu; }

  #endif

}


TEST_CASE("Tensor - Stride ", "[tensor]"){

  Shape shape(1000);

  index_t stride = 3;
  index_t offset = 700;

  SECTION("GPU"){ using DEVICE = gpu;

    bool found = false;

    Tensor<DEVICE> tensor(shape); tensor.allocate(); initilize::zeros(tensor);

    index_t half = shape.size()/2;

    for (int i = 0; i < half; ++i)
      tensor.set(i*2, 1);

    auto tensor2 = expression::transform::stride(tensor, offset, stride, shape.size()-offset).eval();

    float sum = 0;


    for (int i = 0; i < tensor2.shape().size();  ++i){
      sum += tensor2.at(i);
    }

    REQUIRE (sum == 50);

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu; }

  #endif

}
