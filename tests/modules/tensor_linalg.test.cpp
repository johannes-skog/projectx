#include <catch.hpp>

#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/expression-inl.h>
#include <fundamental/tensor-inl.h>

#include <fundamental/linalg/linalg_base.h>

#include <fundamental/initializer.h>

#include <ops/transform.h>

using namespace core;
using namespace expression;
using namespace initilize;
using namespace transform;


TEST_CASE("Tensor - Matmul ", "[tensor]"){

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensorA(4, 2); tensorA.allocate(); ones(tensorA);

    tensorA.set({0, 0}, 1.2215);
    tensorA.set({2, 1}, 3.1451);
    tensorA.set({3, 0}, 4.21);
    tensorA.set({3, 1}, 122.1);

    Tensor<DEVICE> tensorB(2, 4); tensorB.allocate(); ones(tensorB);

    tensorB.set({0, 1}, 24.20);
    tensorB.set({0, 3}, 2.29);
    tensorB.set({1, 1}, 7.2156);
    tensorB.set({1, 2}, 2.11);

    auto tensorC = linalg::matmul(tensorA, tensorB).eval();

  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu; }

  #endif

}



TEST_CASE("Tensor - Transpose Matmul ", "[tensor]"){

  int dim = 2;

  index_t _shapeA [SHAPE_MAX_DIM] = {4, 2};

  index_t _shapeB [SHAPE_MAX_DIM] = {4, 2};

  index_t _shapeC [SHAPE_MAX_DIM] = {4, 4};

  Shape shapeA(_shapeA, dim);
  Shape shapeB(_shapeB, dim);
  Shape shapeC(_shapeC, dim);

  SECTION("CPU"){ using DEVICE = cpu;

    bool found = false;

    Tensor<DEVICE> tensorA(shapeA); tensorA.allocate(); ones(tensorA);

    tensorA.set({0, 0}, 1.2215);
    tensorA.set({2, 1}, 3.1451);
    tensorA.set({3, 0}, 4.21);
    tensorA.set({3, 1}, 122.1);

    Tensor<DEVICE> tensorB(shapeB); tensorB.allocate(); ones(tensorB);

    tensorB.set({1, 0}, 24.20);
    tensorB.set({3, 0}, 2.29);
    tensorB.set({1, 1}, 7.2156);
    tensorB.set({2, 1}, 2.11);

    Tensor<DEVICE> tensorC(shapeC); tensorC.allocate(); zeros(tensorC);

    tensorC = linalg::matmul(tensorA, linalg::transpose(tensorB));


  }

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu; }

  #endif

}
