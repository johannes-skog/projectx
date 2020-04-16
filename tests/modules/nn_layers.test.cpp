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

using namespace core;
using namespace expression;
using namespace nn;
using namespace initilize;

TEST_CASE("Layer - Test ", "[layer]"){

  Shape shape(100, 1000);

  STREAM_BACKWARD(gpu, DEFAULT_STREAM).record(false);
  STREAM_BACKWARD(cpu, DEFAULT_STREAM).record(false);

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto input  = std::make_shared<TensorContainer<DEVICE>>();
      auto output = std::make_shared<TensorContainer<DEVICE>>();

      input->set(shape);

      auto d = input->data();

      initilize::ones(d);

      nn::FullyConnected<DEVICE> fully(1000, 101);

      nn::FullyConnected<DEVICE> fully2(1000, 100);

      auto cweight = fully.param("weight")->data();

      initilize::constant(cweight, (float)1);

      fully.forward(input, output);

      auto tensor_transformed = output->data();

      auto start = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < 10000; ++i){
         fully.forward(input, output);
      }

      auto finish = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> elapsed = finish - start;
      std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    }

  #endif

}


TEST_CASE("Layer - FullyConnected ", "[layer]"){

  Shape shape(2, 3);

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto input  = ContainerFactory<DEVICE, 0, float>("Input");
      auto output = ContainerFactory<DEVICE, 0, float>("Output");

      input->set(shape);

      auto tensor = input->data();

      initilize::ones(tensor);

      tensor.set({0, 0}, 1);
      tensor.set({0, 1}, 3);
      tensor.set({0, 2}, 6);

      tensor.set({1, 0}, 9);
      tensor.set({1, 1}, 3);
      tensor.set({1, 2}, 9);

      nn::FullyConnected<DEVICE> fully(3, 4);

      nn::Bias<DEVICE> bias(1, 4);

      auto cweight = fully.param("weight")->data();

      auto cbias = bias.param("weight")->data();

      float biasf = 0.2;

      initilize::constant(cweight, (float)0);
      initilize::constant(cbias, (float)biasf);

      cweight.set({0, 0}, 1);
      cweight.set({0, 1}, 3);
      cweight.set({0, 2}, 4);

      cweight.set({1, 0}, 5);
      cweight.set({1, 1}, 9);
      cweight.set({1, 2}, 1);

      cweight.set({2, 0}, 1);
      cweight.set({2, 1}, 5);
      cweight.set({2, 2}, 3);

      cweight.set({3, 0}, 4);
      cweight.set({3, 1}, 4);
      cweight.set({3, 2}, 6);

      fully.forward(input, output);

      auto tensor_transformed = output->data();

      bias.forward_inplace(output);

      auto y = output->data();

      REQUIRE (y.at(0, 0) == 34 + biasf);
      REQUIRE (y.at(0, 1) == 38 + biasf);
      REQUIRE (y.at(0, 2) == 34 + biasf);
      REQUIRE (y.at(0, 3) == 52 + biasf);

      REQUIRE (y.at(1, 0) == 54 + biasf);
      REQUIRE (y.at(1, 2) == 51 + biasf);
      REQUIRE (y.at(1, 1) == 81 + biasf);
      REQUIRE (y.at(1, 3) == 102 + biasf);

    }

  #endif

}


TEST_CASE("Layer - GradientChecking Bias", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x = ContainerFactory<DEVICE, 0, float>("x", core::Shape(20, 20, 10));

      nn::GradientCheck<DEVICE> checker;

      nn::Bias<DEVICE, 0, float> layer(2, 10);

      auto w = layer.param("weight");

      auto fwd = [&x, &layer](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        layer.forward(x, y);
      };

      checker.register_var("x", x, -2, 2);
      checker.register_var("w", w, -2, 2);

      auto result = checker.perturbation(fwd, 10, 100, 0.1);

      REQUIRE (result.at("x") < 0.04);
      REQUIRE (result.at("w") < 0.04);

    }

  #endif

}


TEST_CASE("Layer - GradientChecking Fully", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x = ContainerFactory<DEVICE, 0, float>("x", core::Shape(200, 5));

      nn::GradientCheck<DEVICE> checker;

      nn::FullyConnected<DEVICE, 0, float> layer(5, 50);

      auto w = layer.param("weight");

      auto fwd = [&x, &layer](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        layer.forward(x, y);
      };

      checker.register_var("x", x, -1, 1);
      checker.register_var("w", w, -1, 1);

      auto result = checker.perturbation(fwd, 10, 100, 0.01);

      REQUIRE (result.at("x") < 0.04);
      REQUIRE (result.at("w") < 0.04);

    }

  #endif

}



TEST_CASE("Layer - GradientChecking Conv", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x = ContainerFactory<DEVICE, 0, float>("x", core::Shape(5, 5, 10, 3));

      nn::GradientCheck<DEVICE> checker;

      nn::Convolution2d<DEVICE, 0, float> layer(3, 5, 3, 3, 1, 1);

      auto w = layer.param("weight");

      auto fwd = [&x, &layer](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        layer.forward(x, y);
      };

      checker.register_var("x", x, -1, 1);
      checker.register_var("w", w, -1, 1);

      auto result = checker.perturbation(fwd);

      REQUIRE (result.at("x") < 0.04);
      REQUIRE (result.at("w") < 0.04);

    }

  #endif

}


TEST_CASE("Layer - GradientChecking Expression ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(5, 20));

      auto x2 = ContainerFactory<DEVICE, 0, float>("x2", core::Shape(5, 20));

      nn::GradientCheck<DEVICE> checker;

      auto fwd = [&x1, &x2](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        (*y) = primitives::square(*x1) * core::Scalar<DEVICE>(5) + (*x1) * (*x2);
      };

      checker.register_var("x1", x1, -1, 1);
      checker.register_var("x2", x2, -1, 1);

      auto result = checker.perturbation(fwd, 10, 100, 0.005);

      REQUIRE (result.at("x1") < 0.04);
      REQUIRE (result.at("x2") < 0.04);

    }

  #endif

}


TEST_CASE("Layer - GradientChecking Concat ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(5, 22));

      auto x2 = ContainerFactory<DEVICE, 0, float>("x2", core::Shape(5, 10));

      nn::GradientCheck<DEVICE> checker;

      auto fwd = [&x1, &x2](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        (*y) = expression::transform::concat(*x1, *x2, 1);
      };

      checker.register_var("x1", x1, -1, 1);
      checker.register_var("x2", x2, -1, 1);

      auto result = checker.perturbation(fwd, 10, 100, 0.005);

      REQUIRE (result.at("x1") < 0.04);
      REQUIRE (result.at("x2") < 0.04);

    }

  #endif

}

TEST_CASE("Layer - GradientChecking Test ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(5, 22));

      auto fwd = [](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        //printf("%s \n", "From lambda" );
        //printf("INSIDE LAMBDA %d \n", STREAM_HANDLER(DEVICE, 0).dir );
      };

      STREAM(DEVICE, 0).synchronize();

      {

        auto c = STREAM_CONTEXT(DEVICE, 0, STREAM_BACKWARD_DIR);
        //printf("INSIDE  %d \n", STREAM_HANDLER(DEVICE, 0).dir );

        STREAM(DEVICE, 0).record(true);

        STREAM(DEVICE, 0).eager();

        //printf("Put  %d \n", STREAM_HANDLER(DEVICE, 0).dir );

        STREAM(DEVICE, 0).put(fwd, x1);
        STREAM(DEVICE, 0).put(fwd, x1);

      }

      STREAM(DEVICE, 0).synchronize();

      //printf("OUTSIDE  %d \n", STREAM_HANDLER(DEVICE, 0).dir );

    }

  #endif

}

TEST_CASE("Layer - GradientChecking Permute ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(6, 5, 3, 7));

      nn::GradientCheck<DEVICE> checker;

      auto fwd = [&x1](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {
        (*y) = (expression::transform::permute(*x1, {1, 3, 0, 2}));
      };

      checker.register_var("x1", x1, -1, 1);

      auto result = checker.perturbation(fwd, 10, 100, 0.005);

      REQUIRE (result.at("x1") < 0.04);

    }

  #endif

}

TEST_CASE("Layer - GradientChecking Slice ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(6, 22, 20, 6));

      nn::GradientCheck<DEVICE> checker;

      auto fwd = [&x1](std::shared_ptr<nn::TensorContainer<DEVICE>> y)
      {

        std::vector<index_t> slice = {0, 3, 0, 11, 0, 10, 0, 3};

        (*y) = (expression::transform::slice(*x1, slice));

      };

      checker.register_var("x1", x1, -1, 1);

      auto result = checker.perturbation(fwd, 10, 100, 0.005);

      REQUIRE (result.at("x1") < 0.04);

    }

  #endif

}


TEST_CASE("Layer - GradientChecking Reducer ", "[layer]"){

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      auto x1 = ContainerFactory<DEVICE, 0, float>("x1", core::Shape(2, 100));

      auto x2 = ContainerFactory<DEVICE, 0, float>("x2", core::Shape(2, 1));

      auto indices = ContainerFactory<DEVICE, 0, int32_t>("indices", core::Shape(2, 1));

      auto d1 = x1->data();
      auto d2 = x2->data();
      auto di = indices->data();

      expression::initilize::zeros(d1);
      expression::initilize::zeros(d2);

      d1.set({0, 10}, -2);
      d1.set({1, 19}, -2);

      operators::reduce_min_i(d1, d2, di);

    }

  #endif

}

/*

#include <fundamental/image/image.h>

TEST_CASE("Layer - TA ", "[layer]"){

  Shape shape(2, 10, 100, 2, 100);

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      GradientCheck<DEVICE> checker;

      nn::Container<DEVICE> x1;
      nn::Container<DEVICE> x2;

      nn::Container<DEVICE> y;

      core::Shape shape(2, 20);

      x1.set(NN_TENSOR_DATA, shape);
      x2.set(NN_TENSOR_DATA, shape);

      y.set(NN_TENSOR_DATA, shape);

      std::map<std::string, nn::Container<DEVICE> > vars;

      vars.insert({"x1", x1});
      vars.insert({"x2", x2});

      auto fwd = [&x1, &x2]()
      {

        nn::Container<DEVICE> y;

        y = expression::F<expression::op::square>(x1.exp()) * core::Scalar<DEVICE>(5) +
                   x2.exp()*core::Scalar<DEVICE>(10);

        return y;

      };

      auto initilizer = [&x1, &x2](int i)
      {

        expression::initilize::uniform(x1.data(), float(-1), float(1));
        expression::initilize::uniform(x2.data(), float(-1), float(1));

        expression::initilize::zeros(x1.gradient());
        expression::initilize::zeros(x2.gradient());

      };

      //auto result = checker.perturbation(fwd, vars, initilizer);

      //REQUIRE (result.at("x1") < 0.04);

      //REQUIRE (result.at("x2") < 0.04);

    }

  #endif

}

#include <fundamental/image/image.h>

TEST_CASE("Layer - Conv ", "[layer]"){

  Shape shape(2, 100, 200, 3);

  #ifdef TENSOR_USE_CUDA

    SECTION("GPU"){ using DEVICE = gpu;

      Container<DEVICE> input;
      Container<DEVICE> output;

      Tensor<cpu, 0, uint8_t> image_tensor = image::load("/home/johannes/test.jpg");

      auto image_tensor_gpu = to_cuda(image_tensor);

      Tensor<DEVICE, 0, float> image_float(image_tensor_gpu.shape());
      image_float.allocate();

      image_float = cast<float>(image_tensor_gpu);

      nn::Convolution2d<DEVICE> conv(3, 3, 3, 3, 1, 1);

      nn::Bias<DEVICE> bias(3, 3);

      auto& cbias = bias.param("weight").data();

      initilize::constant(cbias, (float)5);

      auto& w = conv.param("weight").data();

      zeros(w);

      w.set({0, 1, 1, 0}, 1.0);
      w.set({1, 1, 1, 1}, 1.0);
      w.set({2, 1, 1, 2}, 1.0);

      input.set(NN_TENSOR_DATA, image_float);

      conv.forward(input, output);

      bias.forward_inplace(output);

      //activation::forward_inplace<activation::ReLu>(y);

      auto image_cpu= to_cpu(output.data());

      int e = image::savej(image_cpu, "testaoutput.jpg");


    }

  #endif

}

*/
