#include <catch.hpp>
#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/expression-inl.h>
#include <fundamental/tensor-inl.h>

#include <fundamental/transform/transform_base.h>

#include <fundamental/image/image.h>

#include <ops/interpolation.h>

using namespace core;
using namespace expression;
using namespace transform;

TEST_CASE("Image - load ", "[image]"){

  SECTION("CPU"){ using DEVICE = cpu;

    Tensor<DEVICE, 0, uint8_t> image_tensor = image::load("/home/johannes/test.jpg");

    Tensor<DEVICE, 0, float> image_float(image_tensor.shape());

    image_float.allocate();

    image_float = cast<float>(image_tensor);

    core::Shape scaled(image_float.shape());
    scaled.shape[2] *= 2;
    scaled.shape[1] *= 1;

    Tensor<gpu> image_scaled(scaled);
    image_scaled.allocate();

    auto image_gpu = to_cuda(image_float);

    operators::bilinear_interpolation(image_gpu, image_scaled);

    auto image_scaled_cpu = to_cpu(image_scaled);

    Tensor<DEVICE, 0, float> image_offset(image_float.shape());
    image_offset.allocate();

    Constant<DEVICE, float> offset(2, image_float.shape());

    image_float = image_float * offset;

    int e = image::savej(image_scaled_cpu, "testaoutput.jpg");

    REQUIRE(e == 1);

  }

  #ifdef TENSOR_USE_CUDA
    SECTION("GPU"){  using DEVICE = gpu;
    }
  #endif

}
