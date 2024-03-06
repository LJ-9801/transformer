#include <vector>
#include <numeric>
#include <assert.h>
#include <pthread.h>
#include <iostream>

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


// @todo: needs to be optimized
template <typename T>
void gemm(const T* a, const T* b, const T* c, T* d, int M, int N, int K)
{
  #pragma omp parallel for
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      T sum = 0;
      #pragma unroll
      for(int k = 0; k < K; k++){
        sum += a[i * K + k] * b[k * N + j];
      }
      if(c != nullptr)
        d[i * N + j] = sum + c[i * N + j];
      else
        d[i * N + j] = sum;
    }
  }
}

/*
  here we are just doing a simple batch matmul
  where tensor A has to be bigger then tensor B

  the last two dimension of A and B has to follow
  the matmul rule
*/
template <typename T>
Tensor<T> batch_matmul(const Tensor<T>* a, const Tensor<T>* b, const Tensor<T>* c)
{
  if(a->empty() || b->empty()){
    return Tensor<T>();
  }

  uint32_t M = a->shape()[a->shape().size() - 2];
  uint32_t Ka = a->shape()[a->shape().size() - 1];
  uint32_t Kb = b->shape()[b->shape().size() - 2];
  uint32_t N = b->shape()[b->shape().size() - 1];

  assert(Ka == Kb && "The last two dimensions of A and B must be the same");
  assert(a->shape().size() == 4 && b->shape().size() == 2 && "A has to be a 4D tensor and B has to be a 2D tensor");

  // here we are assuming size A > size B, 
  // fow now let's just consider A has dim 4 and B has dim 2
  uint32_t total_size = std::accumulate(a->shape().begin(), a->shape().end(), 1, std::multiplies<uint32_t>());
  uint32_t nBatch = a->shape()[0] * a->shape()[1];

  T* a_ptr = accessor<T>::const_ptr(*a);
  T* b_ptr = accessor<T>::const_ptr(*b);
  T* c_ptr = nullptr;
  if(c != nullptr){
    c_ptr = accessor<T>::const_ptr(*c);
  }

  Tensor<T> out = Tensor<T>({a->shape()[0], a->shape()[1], M, N});
  T* out_ptr = accessor<T>::get(out);

  #pragma omp parallel for
  for(int i = 0; i < nBatch; i++){
    a_ptr = a_ptr + i * M * Ka; 
    out_ptr = out_ptr + i * M * N; 
    gemm(a_ptr, b_ptr, c_ptr, out_ptr, M, N, Ka);
  }

  return out;
}

template <typename T>
Tensor<T> softmax(Tensor<T>* t, int axis)
{ 
  if(t->empty() || t == nullptr){
    return Tensor<T>();
  }


  return Tensor<T>();
}