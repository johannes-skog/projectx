#ifndef NN_GRADIENT_CHECK_H
#define NN_GRADIENT_CHECK_H

#include <limits>       // std::numeric_limits

#include <fundamental/tensor.h>
#include <fundamental/initializer.h>

#include <ops/reducers.h>
#include <nn/container.h>
#include <algorithm>

namespace nn{

  template<typename xpu, index_t stream_id, typename T>
  class GradientCheckBase{

    using SharedTensor = std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>;

    std::shared_ptr<std::mt19937> gen;

    std::vector<std::function<void(void)>> initilizer;

    std::map<std::string, SharedTensor > vars;

    int uniform_int(T l, T h) const{

      std::uniform_int_distribution<> dist(0, h);

      return dist(*gen);

    }

    protected:

      GradientCheckBase(): gen{std::make_shared<std::mt19937>(std::random_device{}())}{}

      T center_difference(T pos, T neg, T epsilon) const{
          return ((pos - neg) / (2 * epsilon));
      }

    public:


      void register_var(std::string s, SharedTensor x, T l = -1, T h = 1){

        this->vars.insert({s, x });

        auto init = [x, l, h](){

          auto xd = x->data();
          auto xg = x->gradient();

          expression::initilize::uniform(xd, T(l), T(h));
          expression::initilize::zeros(xg);

        };

        initilizer.push_back(init);

      }

      void initilize(){

        for (auto init : initilizer){
          init();
        }

      }

      void loss(SharedTensor y, SharedTensor l){

        auto scope = SCOPE.withv({"Loss"});

        auto sum = ContainerFactory<xpu, stream_id, T>("Sum", core::Shape(1));
        auto y_squared = ContainerFactory<xpu, stream_id, T>("Squared", core::Shape(1));

        *y_squared = primitives::square(*y);

        operators::asum(*y_squared, *sum, 0, 1, y->shape().size());

        (*l) = (*sum) * core::Scalar<xpu, T, stream_id>(0.5);

      }

      std::map<std::string,  T>
      perturbation(std::function<void(SharedTensor)> fwd,
                   int N1 = 10, int N2 = 100, T epsilon = 0.1){

        std::map<std::string,  T> result;

        for (auto & it : vars)
          result[it.first] = std::numeric_limits<T>::min();

        auto scope = SCOPE.withv({"GradientCheck", "Pertubation"});

        auto l = ContainerFactory<xpu, stream_id, T>("Loss", core::Shape(1));
        auto lp = ContainerFactory<xpu, stream_id, T>("Loss_pos", core::Shape(1));
        auto ln = ContainerFactory<xpu, stream_id, T>("Loss_neg", core::Shape(1));

        auto y = ContainerFactory<xpu, stream_id, T>("y");

        for (int i = 0; i < N1; ++i){

          STREAM_BACKWARD(xpu, stream_id).record(true).lazy().hold_on();

          // TODO, will not work with lazy, backward pass calls other
          // FORWARD functions, it there are lazy excetuted, then
          // it will not work...
          STREAM_FORWARD(xpu, stream_id).record(true).eager();

          initilize();

          fwd(y);

          this->loss(y, l);

          l->gradient().set(0, 1);

          auto grad = y->gradient();
          expression::initilize::zeros(grad);

          STREAM_FORWARD(xpu, stream_id).synchronize();

          //printf("%f\n", y->gradient().at(0));

          STREAM_BACKWARD(xpu, stream_id).hold_off().synchronize().record(false);

          //printf("%f\n", y->gradient().at(0));

          for (int ii = 0; ii < N2; ++ii){

            for (auto & it : vars){

              STREAM_FORWARD(xpu, stream_id).synchronize();

              auto& var = it.second;

              int idx = this->uniform_int(0, var->shape().size() -1);

              T x_idx_value = var->data().at(idx);

              T dvar_analytic = var->gradient().at(idx);

              var->data().set(idx, (x_idx_value + epsilon));
              fwd(y);   this->loss(y, lp);

              STREAM_FORWARD(xpu, stream_id).synchronize();

              var->data().set(idx, (x_idx_value - epsilon));
              fwd(y);   this->loss(y, ln);

              STREAM_FORWARD(xpu, stream_id).synchronize();

              T dvar_numeric  = this->center_difference(lp->data().at(0),
                                                        ln->data().at(0),
                                                        epsilon);

              T rel = std::abs(dvar_numeric-dvar_analytic) /
                      (abs(dvar_numeric) + abs(dvar_analytic) + 1e-8);


              if (rel > .3){

                printf("%s REL %f\n", it.first.c_str(), rel);
                printf("%s Numeric %f\n", it.first.c_str(), dvar_numeric);
                printf("%s Analytics %f\n", it.first.c_str(), dvar_analytic);

              }

              var->data().set(idx, x_idx_value); // Revert to the the org value

              if (rel > result.at(it.first))
                result.at(it.first) = rel;

            }

          }

        }

        return result;

      }

  };


  template<typename xpu, index_t stream_id = DEFAULT_STREAM,
           typename T = DEFAULT_TYPE>
  class GradientCheck: public GradientCheckBase<xpu, stream_id, T>{

    using SharedTensor = std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>;

    public:

      GradientCheck():
        GradientCheckBase<xpu, stream_id, T>(){}

  };

}

#endif
