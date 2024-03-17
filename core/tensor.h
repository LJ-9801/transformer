#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <memory>
#include <array>

#include "common/kernels.h"

using namespace std;

#define GET_SIZE(shape) std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>())

typedef vector<uint32_t> shape_t;
typedef vector<uint32_t> stride_t;

static inline stride_t calculate_stride(const shape_t& shape);

template <typename T>
struct accessor;

template <typename T>
struct Tensor
{
    public:
    Tensor(T* data, shape_t shape) {
        this->_data = data;
        this->_shape = shape;
        this->_size = GET_SIZE(shape);
        this->_stride = calculate_stride(shape);
    }

    Tensor(shape_t shape) {
        this->_size = GET_SIZE(shape); 
        this->_data = this->alloc(this->_size); 
        this->_shape = shape;
        this->_stride = calculate_stride(shape);
    }

    Tensor(const Tensor<T>& other){
      if(other.empty()){
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
        this->_stride = {};
        return;
      }

      this->_data = this->alloc(other.size()); 
      this->_shape = other.shape();
      this->_stride = other.stride();
      this->_size = other.size();
      this->copy(other._data);
    }

    Tensor() {
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
        this->_stride = {};
    }

    ~Tensor() {
      this->clean(); 
    }

    T* data() const {
        return this->_data;
    }

    void view(shape_t shape){
        assert(GET_SIZE(shape) == this->_size && 
              "The new shape must have the same size as the old shape");
        this->_shape = shape;
    }

    shape_t shape() const {
        return this->_shape;
    }

    stride_t stride() const {
        return this->_stride;
    }

    size_t size() const {
        return this->_size;
    }

    uint16_t ndim() const {
        return this->_shape.size();
    }

    bool empty() const {
        return this->_data == nullptr;
    }

    // TODO this is so inefficient as we are using
    // vector to store intermediate indexes
    void transpose(int16_t axis1, int16_t axis2){ 
      if(this->empty()) return;

      if(axis1 < 0) axis1 = this->_shape.size() + axis1;
      if(axis2 < 0) axis2 = this->_shape.size() + axis2;

      shape_t new_shape = this->_shape;
      std::swap(new_shape[axis1], new_shape[axis2]);
      stride_t new_stride = calculate_stride(new_shape);
      T* new_data = this->alloc(this->size()); 

      tranpose_tensor(this->_data, &new_data, 
                      this->_shape.data(), new_shape.data(),
                      this->_stride.data(), new_stride.data(),
                      this->size(), this->_shape.size(),
                      axis1, axis2);

      this->clean(); 
      this->_data = new_data;
      this->_shape = new_shape;
      this->_stride = new_stride;
    }

    Tensor<T>& operator/=(const T& scalar){
        if(this->empty()){
            return *this;
        }

        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] /= scalar;
        }

        return *this;
    }

    Tensor<T> operator+(const Tensor<T>& other){
        if(this->empty() || other.empty()){
            return Tensor<T>();
        }

        // TODO: this needs to check broadcasting rules
        if(this->shape() != other.shape()){
            throw runtime_error("The two tensors must have the same shape");
        }
        Tensor<T> out = Tensor<T>(this->shape());
        
        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            out._data[i] = this->_data[i] + other._data[i];
        }

        return out;
    }

    Tensor<T> operator-(const T& scalar){
        if(this->empty()){
            return *this;
        }

        Tensor<T> out = Tensor<T>(*this);
        out -= scalar;
        return out;
    }

    Tensor<T>& operator*= (const T& scalar){
        if(this->empty()){
            return *this;
        }

        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] *= scalar;
        }

        return *this;
    }

    Tensor<T>& operator-=(const T& scalar){
        if(this->empty()){
            return *this;
        }

        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] -= scalar;
        }

        return *this;
    }


    Tensor<T>& operator=(const Tensor<T>& other){
      if(this == &other){
        return *this;
      }

      if(this->_data != nullptr){
        this->clean();
      }

      if(other.empty()){
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
        return *this;
      }

      this->_data = this->alloc(other.size()); 
      this->_shape = other.shape();
      this->_size = other.size();
      this->copy(other._data); 
      
      return *this;
    }

    T operator[](uint64_t index) const {
        return this->_data[index];
    }

    T& at(const shape_t& index){
        uint32_t idx = index_from_shape(index.data(), this->_shape.data(), this->ndim());
        return this->_data[idx];
    }

    Tensor<T>& fill_one(){
        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] = 1;
        }

        return *this;
    }

    Tensor<T>& arange(){
        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] = i;
        }

        return *this;
    }

    template <typename U>
    friend struct accessor;

    private:

    // TODO: tile and parallize memcpy
    void copy(const T* data){
      #pragma omp parallel for
      for(int i = 0; i < _size; i++){
        this->_data[i] = data[i];
      }
    }

    void clean(){
      if(this->_data != nullptr){
        delete[] this->_data;
        this->_data = nullptr;
      }
    }

    T* alloc(size_t size){
      return new T[size];
    }

    T* _data;
    shape_t _shape;
    stride_t _stride;
    size_t _size;
};

// accessor for tensor
template <typename T>
struct accessor {
  static T* const_ptr(const Tensor<T>& t) { return t._data;}
  static T* get(Tensor<T>& t) { return t._data;}
};

static inline stride_t calculate_stride(const shape_t& shape){
  stride_t stride(shape.size(), 1);
  for(int i = shape.size() - 2; i >= 0; i--){
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

std::ostream& operator<<(std::ostream& os, const shape_t& shape){
  os << "[";
  for(int i = 0; i < shape.size(); i++){
    os << shape[i];
    if(i != shape.size() - 1){
      os << ", ";
    }
  }
  os << "]";
  return os;
}
#endif

