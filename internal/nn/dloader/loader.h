#ifndef DLOADER_LOADER_H
#define DLOADER_LOADER_H

#include <fundamental/image/image.h>
#include <fundamental/tensor.h>
#include <nn/container.h>

namespace meta{

  template <typename TD, typename T>
  struct Distriubution{

    struct Data{

      std::vector<T> el;

      const char* desc;

      Data(const char* _desc): desc{_desc} {}

      size_t count(void) { return el.size(); }

      void add(T e){ el.push_back(e); }

    };

    std::map<TD, Data> odata;

    Distriubution(){}

    template <typename... Args>
    void add_class(TD idx, Args&&... args){
      DEBUG_ASSERT(odata.find(idx) == odata.end());
      odata.emplace(idx, std::forward<Args>(args)...);
    }

    void add_observeration(TD idx, T dataref){
      DEBUG_ASSERT(odata.find(idx) != odata.end());
      odata.at(idx).add(dataref);
    }

  };

  template<typename RT>
  struct ImageBase{

    std::vector<RT> regions;

    const char* _file;

    unique_id id;

    int w, h;

    ImageBase(unique_id _id, const char* __file):
              id{_id}, _file{__file}, w{-1}, h{-1} {}

    ImageBase(unique_id _id, const char* __file, int _w, int _h):
              id{_id}, _file{__file}, w{_w}, h{_h} {}


    template <typename... Args>
    RT& add_region(Args&&... args){
      regions.emplace_back(id, std::forward<Args>(args)...);
      return regions.back();
    }

    std::string filename(){
      return _file;
    }


  };

  template<typename MT>
  struct RegionBase{

    std::vector<MT> metas;
    unique_id id;

    RegionBase(unique_id _id): id{_id} {}

    template <typename... Args>
    MT& add_meta(Args&&... args){
      metas.emplace_back(this->id, std::forward<Args>(args)...);
      return metas.back();
    }

  };

  struct MetaBase{

    const char* s;
    unique_id id;

    MetaBase(unique_id _id): id{_id}{}

  };

  template<typename MT>
  struct RegionFull: public RegionBase<MT>{

    RegionFull(unique_id _id): RegionBase<MT>(_id){}

  };

  template<typename MT>
  struct RegionBox: public RegionBase<MT>{

    int w, h;
    int xc, yc;

    RegionBox(unique_id _id, int _h, int _w, int _xc, int _yc):
              h{_h}, w{_w}, xc{_xc}, yc{_yc}, RegionBase<MT>(_id){}

  };

  struct MetaClass: public MetaBase{

    const int cid;

    MetaClass(unique_id _id, int _cid): cid{_cid}, MetaBase(_id){}

    void link(Distriubution<int, unique_id> dist){
      dist.add_observeration(cid, this->id);
    }

  };

  template<typename RT>
  struct ImageDetect: public ImageBase<RT>{

    ImageDetect(unique_id _id, const char* __file): ImageBase<RT>(_id, __file) {}

  };

  template<typename MT>
  struct ImageFull: public ImageBase<RegionFull<MT>>{

    ImageFull(unique_id _id, const char* __file):
              ImageBase<RegionFull<MT>>(_id, __file) {}

    template <typename... Args>
    MT& add_meta(Args&&... args){
      DEBUG_ASSERT(this->regions.size() == 0);
      return this->add_region().add_meta(std::forward<Args>(args)...);
    }

  };

}

template<typename META, typename xpu, index_t stream_id = DEFAULT_STREAM,
         typename T = DEFAULT_TYPE>
class ImageLoader{

  using SharedTensor = std::shared_ptr<nn::TensorContainer<xpu, stream_id, T>>;

  // Get N sampled META
  using SAMPLE = std::function<std::vector<META>(std::vector<META>&, int N)>;

  // Process the loaded image, cast to T etch resize
  using PROCESS = std::function<void(META&, core::Tensor<xpu, stream_id, T>&,
                                     core::Tensor<xpu, stream_id, T>&)>;

  std::vector<META> data;

  SAMPLE sampler;

  PROCESS process;

  public:

    ImageLoader(){}

    ImageLoader(SAMPLE _sampler, PROCESS _process):
                sampler{_sampler}, process{_process} {}

    ImageLoader& set_sampler(SAMPLE _sampler){
      STREAM(xpu, stream_id).synchronize();
      sampler = _sampler;
      return *this;
    }

    ImageLoader& set_process(PROCESS _process){
      STREAM(xpu, stream_id).synchronize();
      process = _process;
      return *this;
    }

    template <typename... Args>
    META& observeration(Args&&... args){
      data.emplace_back((unique_id) data.size(), std::forward<Args>(args)...);
      return data.back();
    }

    ImageLoader& run(SharedTensor x, SharedTensor labels){

      auto _process = [this](SharedTensor x, SharedTensor labels){

        int N = x->shape()[0];

        core::Tensor<xpu, stream_id, T> xt = x->data();
        core::Tensor<xpu, stream_id, T> lt = x->data();

        std::vector<core::Tensor<xpu, stream_id, T>> xy = core::split(xt);
        std::vector<core::Tensor<xpu, stream_id, T>> ly = core::split(lt);

        // Sample N meta entries
        std::vector<META> samples = this->sampler(this->data, N);

        for (int i = 0; i < N; ++i)
          this->process(samples[i], xy[i], ly[i]);

      };

      STREAM(xpu, stream_id).put(_process, x, labels);

      return *this;

    }

};

#endif
