#ifndef FUNDAMENTAL_SHAPE_H
#define FUNDAMENTAL_SHAPE_H

typedef enum{

  N = 0,
  H = 1,
  W = 2,
  C = 3

} order;


namespace core{

class Shape{

private:

public:

  index_t shape [SHAPE_MAX_DIM];
  index_t dim; // number of dims
  index_t N; // number of elements
  index_t stride; // stride [batch, x, y, z], stride = x * y * z

  TENSOR_INLINE Shape() {
    for(int i = 0; i < SHAPE_MAX_DIM; ++i) shape[i] = -1;
    this->N = 0; this->stride = 0;
  }

  MULTI_ARG_INDEX  TENSOR_INLINE Shape(Args... values): shape{ values... },
                                                        dim{sizeof...(values)}{
    initilize();
  }

  TENSOR_INLINE Shape(const Shape& src): dim{src.dim}{
    for(int i = 0; i < SHAPE_MAX_DIM; ++i) shape[i] = src.shape[i];
    initilize();
  }

  TENSOR_INLINE Shape(index_t values[SHAPE_MAX_DIM], index_t dim): dim{dim}{
    for (int i = 0; i < dim; ++i) shape[i] = values[i];
    initilize();
  }

  TENSOR_INLINE void initilize(void){
    this->N = ndim(0);
    if (this->dim > 1){
     this->stride = _size(1);
     return;
    }
    this->stride = 1;
  }

  TENSOR_INLINE index_t size(order d) const{
    if (d > dim ) return 0;
    else return shape[d];
  }

  TENSOR_INLINE index_t _size(index_t offset) const{
    if (this->dim <= (offset)) return 0;
    index_t _size = this->shape[offset];
    for (int i = offset + 1; i < this->dim; ++i)
      _size *= this->shape[i];
    return _size;
  }

  TENSOR_INLINE bool is_scalar(void) const{
    if (this->N  == 1 && this->stride == 1) return true;
    return false;
  }

  TENSOR_INLINE index_t operator[](const int idx) const { return this->shape[idx]; }

  TENSOR_INLINE index_t operator[](order d){
    if (d > dim ) return 0;
    else return shape[d];
  }

  TENSOR_INLINE void operator=(const core::Shape& src){
     dim = src.dim;
     for(int i = 0; i < SHAPE_MAX_DIM; ++i) shape[i] = src.shape[i];
     initilize();
  }

  TENSOR_INLINE Shape flatten(void) const{ return Shape(this->size()); }

  TENSOR_INLINE Shape collapse_to_dim(index_t _dim) const{
     DEBUG_ASSERT( (this->dim >= _dim) );
     index_t newShape [SHAPE_MAX_DIM];
     for (int i = 0; i < _dim; ++i) newShape[i] = this->shape[i];
     for (int ii = _dim; ii < this->dim; ++ii)
      newShape[_dim-1] = newShape[_dim-1] * this->shape[ii];
     return Shape(newShape, _dim);
  }

  TENSOR_INLINE Shape reduce_axis(index_t _axis) const{
     DEBUG_ASSERT( (this->dim >= _axis) & _axis > 0 );
     index_t newShape [SHAPE_MAX_DIM];
     for (int i = 0; i < this->dim; ++i) newShape[i] = this->shape[i];
     newShape[_axis] = 1;
     return Shape(newShape, this->dim);
  }

  TENSOR_INLINE Shape offset_shape(index_t _offset) const{
     int rest_dim = (this->dim - _offset);
     DEBUG_ASSERT( rest_dim > 0 );
     index_t newShape [SHAPE_MAX_DIM];
     for (int i = 0; i < rest_dim; ++i) newShape[i] = this->shape[i+_offset];
     return Shape(newShape, rest_dim);
  }

  TENSOR_INLINE void flatten_index(index_t& s, index_t& i, index_t& flatt) const{
    DEBUG_ASSERT(s < this->N && i < stride);
    flatt = s*stride + i;
  }

  TENSOR_INLINE index_t index(index_t values[]) const{
    DEBUG( for (int ii = 0; ii < this->dim; ++ii) DEBUG_ASSERT(values[ii] < this->shape[ii]) );
    index_t i = 0;
    for (int ii = 0; ii < this->dim -1; ++ii)
        i += values[ii] * this->_size(ii+1);
    i += values[this->dim - 1];
    return i;
  }

  TENSOR_INLINE void index(index_t& s, index_t& i, index_t values[SHAPE_MAX_DIM]) const{
    DEBUG( for (int ii = 0; ii < this->dim; ++ii) DEBUG_ASSERT(values[ii] < this->shape[ii]) );
    s = values[0]; i = 0;
    for (int ii = 1; ii < this->dim -1; ++ii) i += values[ii] * this->_size(ii+1);
    i += values[this->dim - 1];
  }

  MULTI_ARG_INDEX TENSOR_INLINE void index(index_t& s, index_t& i, Args&&... _values) const{
    index_t values [] = {_values...};
    DEBUG_ASSERT( ( (sizeof...(_values) == this->dim)) );
    this->index(s, i, values);
  }

  TENSOR_INLINE void deduce_shape(index_t s, index_t i, index_t deduced[SHAPE_MAX_DIM]) const{
    deduced[0] = s; index_t remainder = i;
    for (int idx = 1; idx < this->dim - 1; ++idx){
      index_t prod = _size(idx+1);
      deduced[idx] = (index_t) remainder / prod;
      remainder -= deduced[idx] * prod;
      DEBUG_ASSERT( (deduced[idx] >= 0 && deduced[idx] < this->shape[idx]) );
    }
    deduced[this->dim -1] = remainder;
  }

  TENSOR_INLINE void deduce_shape(index_t i, index_t deduced[SHAPE_MAX_DIM]) const{
    index_t remainder = i;
    for (int idx = 0; idx < this->dim -1; ++idx){
      index_t prod = _size(idx+1);
      deduced[idx] = (index_t) remainder/ prod;
      remainder -= deduced[idx] * prod;
      DEBUG_ASSERT( (deduced[idx] >= 0 && deduced[idx] < this->shape[idx]) );
    }
    deduced[this->dim -1] = remainder;
  }

  TENSOR_INLINE Shape reshape(index_t values[SHAPE_MAX_DIM], index_t n) const{
    DEBUG_ASSERT(  (n < SHAPE_MAX_DIM) );
    index_t newShape [SHAPE_MAX_DIM];
    for (int i = 0; i < n; ++i) newShape[i] = values[i];
    index_t unknown = -1; index_t total = 1; bool first = true;
    for (int i = 0; i < n; ++i){
      if (newShape[i] < 0){ DEBUG_ASSERT( ( unknown < 0) ); // Otherwise we have two unknowns
        unknown = i;
      }else{
        if (first){ total = newShape[i]; first = false; }
        else{ total *= newShape[i];}
      }
    }
    if (unknown >= 0){
      double remainder = this->size()/total;
      DEBUG_ASSERT( ( (int(remainder) - remainder) == 0) ); // Even divisiable with total
      newShape[unknown] = int(remainder);
    }
    return Shape(newShape, n);
  }

  MULTI_ARG TENSOR_INLINE Shape reshape(Args ... values) const{
    index_t newShape [SHAPE_MAX_DIM]{values ...};
    index_t n =  sizeof...(values);
    return this->reshape(newShape, n);
  }

  TENSOR_INLINE index_t size(index_t offset = 0) const { return _size(offset); }

  TENSOR_INLINE index_t ndim(index_t _dim) const {
    if(this->dim < _dim) return 0;
    return this->shape[_dim];
  }

  TENSOR_INLINE bool operator==(const Shape& s) const {
    if (this->dim != s.dim) return false;
    for (int i = 0; i < this->dim; ++i) // ignore the check if the dims are minus
      if (s.shape[i] != this->shape[i] && ( s.shape[i] > 0 &&  this->shape[i] >0 )) return false;
    return true;
  }

};





}



#endif
