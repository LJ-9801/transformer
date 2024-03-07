#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <assert.h>
#include <pthread.h>
#include <iostream>
#include <memory>

#include "kernels.h"

using namespace std;

#define GET_SIZE(shape) std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>())

typedef vector<uint32_t> shape_t;
typedef vector<int32_t> axis_t;
typedef std::unique_ptr<uint32_t[]> stride_t;

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
    }

    Tensor(shape_t shape) {
        this->_size = GET_SIZE(shape); 
        this->_data = new T[this->size()];
        this->_shape = shape;
    }

    Tensor(const Tensor<T>& other){
      if(other.empty()){
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
        return;
      }

      this->_data = new T[other._size];
      this->_shape = other.shape();
      this->_size = other.size();
      this->copy(other._data);
    }

    Tensor() {
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
    }

    ~Tensor() {
      if(this->_data != nullptr){
        delete[] this->_data;
        this->_data = nullptr;
      }
    }

    T* data() const {
        return this->_data;
    }

    void view(shape_t shape){
        size_t new_size = GET_SIZE(shape); 
        assert(new_size == this->_size && "The new shape must have the same size as the old shape");
        this->_shape = shape;
    }

    shape_t shape() const {
        return this->_shape;
    }

    size_t size() const {
        return this->_size;
    }

    bool empty() const {
        return this->_data == nullptr;
    }

    // TODO this is so inefficient as we are using
    // vector to store intermediate indexes
    void transpose(axis_t axes){ 
      if(this->empty()){
        return;
      }

      assert(axes.size() == 2 && "The axes must have two elements");
      if(axes[0] < 0){
        axes[0] = this->_shape.size() + axes[0];
      }

      if(axes[1] < 0){
        axes[1] = this->_shape.size() + axes[1];
      }

      shape_t new_shape = this->_shape;
      std::swap(new_shape[axes[0]], new_shape[axes[1]]);

      // calculate stride
      std::vector<uint32_t> old_stride(this->_shape.size(), 1);
      std::vector<uint32_t> new_stride(new_shape.size(), 1);
      std::vector<uint32_t> old_position(this->_shape.size(), 0);

      for(int i = this->_shape.size() - 2; i >= 0; i--){
        old_stride[i] = old_stride[i + 1] * this->_shape[i + 1];
      }

      for(int i = new_shape.size() - 2; i >= 0; i--){
        new_stride[i] = new_stride[i + 1] * new_shape[i + 1];
      }

      T* new_data = new T[this->size()];

      #pragma omp parallel for
      for(int i = 0; i < this->size(); i++){
        // calculate old position
        for(int j = 0; j < old_position.size(); j++){
          old_position[j] = (i / old_stride[j]) % this->_shape[j];
        }

        // calculate new position
        std::swap(old_position[axes[0]], old_position[axes[1]]);

        int new_index = 0;
        for(int j = 0; j < old_position.size(); j++){
          new_index += old_position[j] * new_stride[j];
        }
        new_data[new_index] = this->_data[i];
      }

      delete[] this->_data;
      this->_data = new_data;
      this->_shape = new_shape;
    }

    Tensor<T>& operator/=(const uint32_t& scalar){
        if(this->empty()){
            return *this;
        }

        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] /= scalar;
        }

        return *this;
    }


    Tensor<T>& operator=(const Tensor<T>& other){
      if(this == &other){
        return *this;
      }

      if(this->_data != nullptr){
        // @todo return this to the pool 
        // instead of deleting it
        delete[] this->_data;
      }

      if(other.empty()){
        this->_data = nullptr;
        this->_shape = {};
        this->_size = 0;
        return *this;
      }

      this->_data = new T[other._size];
      this->_shape = other.shape();
      this->_size = other.size();
      this->copy(other._data); 
      
      return *this;
    }

    void fill_one(){
        #pragma omp parallel for
        for(int i = 0; i < _size; i++){
            this->_data[i] = 1;
        }
    }

    template <typename U>
    friend struct accessor;

    private:

    void copy(const T* data){
      #pragma omp parallel for
      for(int i = 0; i < _size; i++){
        this->_data[i] = data[i];
      }
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
#endif

