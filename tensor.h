#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <numeric>
#include <assert.h>
#include <pthread.h>
#include <iostream>

#include "kernels.h"

using namespace std;


typedef vector<uint32_t> shape_t;
typedef vector<int32_t> axis_t;

template <typename T>
struct accessor;

template <typename T>
struct Tensor
{

    public:
    Tensor(T* data, shape_t shape) {
        this->_data = data;
        this->_shape = shape;
        this->_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    }

    Tensor(shape_t shape) {
        this->_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
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
        this->_shape = shape;
        size_t new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        assert(new_size == this->_size && "The new shape must have the same size as the old shape");
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

    void transpose(axis_t axes){
        // TODO
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
    size_t _size;

};

// accessor for tensor
template <typename T>
struct accessor {
  static T* const_ptr(const Tensor<T>& t) { return t._data;}
  static T* get(Tensor<T>& t) { return t._data;}
};
#endif

