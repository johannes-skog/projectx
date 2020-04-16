#include <catch.hpp>

#include <fundamental/macro.h>
#include <fundamental/shape.h>

#include <fundamental/expression-inl.h>
#include <fundamental/tensor-inl.h>
#include <fundamental/initializer.h>

#include <nn/conv.h>
#include <nn/bias.h>
#include <nn/activation.h>
#include <nn/fully.h>
#include <nn/optimizer.h>
#include <nn/gradient_check.h>

#include <nn/container.h>

#include <ops/transform.h>
#include <ops/reducers.h>
#include <omp.h>
#include <chrono>

#include <nn/initializer.h>

#include <dloader/loader.h>


using namespace core;
using namespace expression;
using namespace nn;
using namespace initilize;


TEST_CASE("Layer - GradientChecking Sliafafce ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

    using META = meta::ImageFull < meta::MetaClass> ;

    ImageLoader<META , DEVICE> loader;

    meta::Distriubution<int, unique_id> dist;

    dist.add_class(2, "Hoist");

    dist.add_class(9, "Hoista");

    dist.add_class(90, "Hoisaft");

    loader.observeration("/home/johannes/test.jpg").add_meta(9).link(dist);

    loader.observeration("/home/johannes/test.jpg").add_meta(90).link(dist);

    loader.observeration("/home/johannes/test.jpg").add_meta(2).link(dist);

    auto sampler = [](std::vector<META>& d, int N){

      std::vector<META> s;
      s.reserve(N);

      for (int i = 0; i < N; ++i)
        s.push_back(d[i]);

      return s;

    };

    auto _process = [](META& m, core::Tensor<DEVICE, DEFAULT_STREAM, DEFAULT_TYPE>& y,
                       core::Tensor<DEVICE, DEFAULT_STREAM, DEFAULT_TYPE>& l){

      int res = 0;
      int N = 10000;

      auto x = image::load<DEFAULT_STREAM>(m.filename());

      for (int i = 0; i < N; ++i)
        res += x.at(i);

      printf("CPU uint8 %d\n", res);

      auto y8 = core::to_cuda(x).view(x.shape().offset_shape(1));

      y = expression::cast<DEFAULT_TYPE>(y8);

      y = primitives::tanh(y);

      float resf = 0;

      for (int i = 0; i < N; ++i)
        resf += y.at(i);

      l.set(0, m.regions[0].metas[0].cid);

      printf("GPU float %f\n", resf);

      printf("l %f\n", l.at(0));

    };

    loader.set_sampler(sampler).set_process(_process);

    {
      auto y = ContainerFactory<DEVICE>("Hejsan");
      bool exists = ContainerContext<DEVICE>().exists("Hejsan");
      printf("HEJAN %d\n" , exists);
    }

    bool exists = ContainerContext<DEVICE>().exists("Hejsan");

    printf("HEJAN %d\n" , exists);

    auto y = ContainerFactory<DEVICE>("Y");

    auto labels = ContainerFactory<DEVICE>("Labels");

    STREAM(DEVICE, DEFAULT_STREAM).eager();

    y->conditional_override(core::Shape(3, 355, 500, 3));

    labels->conditional_override(core::Shape(3));

    loader.run(y, labels);

    STREAM(DEVICE, DEFAULT_STREAM).synchronize();

    auto scope1 = SCOPE.with("test1");

    std::cout << SCOPE.full()  << std::endl;

    {
      auto scope2 = SCOPE.with("test2");
      std::cout << SCOPE.full()  << std::endl;
    }

    std::cout << SCOPE.full()  << std::endl;

    auto scope3 = SCOPE.with("test3");

    std::cout << SCOPE.full()  << std::endl;

    Optimizer<DEVICE> opt(1);

    opt.zero_grad();

    opt.step();

    nn::initilizer::zeros<DEVICE>("Labels");

    }

  #endif

}
