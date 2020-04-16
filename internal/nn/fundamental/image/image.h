#ifndef FUNDAMENTAL_IMAGE_H
#define FUNDAMENTAL_IMAGE_H

#include "stb_image.h"
#include "stb_image_write.h"
#include <fundamental/macro.h>
#include <fundamental/tensor.h>
#include <stdexcept>
#include <fundamental/expression.h>

namespace image{

  template<index_t stream_id = DEFAULT_STREAM>
  core::Tensor<cpu, stream_id, float> loadf(const std::string filename){

    int w, h, c;

    float *data = stbi_loadf(filename.c_str(), &w, &h, &c, 0);

    core::Shape shape(1, h, w, c);

    core::Tensor<cpu, stream_id, float> tensor(data, shape);

    return tensor;

  }

  template<index_t stream_id = DEFAULT_STREAM>
  core::Tensor<cpu, stream_id, uint8_t> load(const std::string filename){

    int w, h, c;

    unsigned char *data = stbi_load(filename.c_str(), &w, &h, &c, 0);

    core::Shape shape(1, h, w, c);

    core::Tensor<cpu, stream_id, uint8_t> tensor(data, shape);

    return tensor;

  }

  template<index_t stream_id>
  int savej(const core::Tensor<cpu, stream_id, uint8_t> tensor, const std::string filename,
            const int quality = 90){

    const core::Shape& shape = tensor.shape();

    int h, w, c;

    if (shape.dim == 4){
      h = shape[1];
      w = shape[2];
      c = shape[3];
    } else if (shape.dim == 3){
      h = shape[0];
      w = shape[1];
      c = shape[2];
    } else if (shape.dim == 2){
      h = shape[0];
      w = shape[1];
      c = 1;
    } else{
      throw std::invalid_argument("Tensor is of non acceptable size");
    }

    int e = stbi_write_jpg(filename.c_str(), w, h, c,  (void*) tensor.ptr(),
                           quality);

    return e;

  }

  template<index_t stream_id>
  int savej(const core::Tensor<cpu, stream_id, float> tensor, const std::string filename,
            const int quality = 90){

     core::Tensor<cpu, stream_id, uint8_t> tensor_i8(tensor.shape());
     tensor_i8.allocate();
     tensor_i8 = expression::cast<uint8_t>(tensor);
     int e  = savej(tensor_i8, filename, quality);
     tensor_i8.deallocate();
     return e;

  }

}

#endif
