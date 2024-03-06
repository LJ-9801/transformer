#include <vector>
#include <numeric>
#include <assert.h>
#include <pthread.h>
#include <iostream>

using namespace std;


typedef vector<uint32_t> shape_t;
typedef vector<int32_t> axis_t;

template <typename T>
struct Tensor
{
    T* data;
    shape_t shape;
    size_t size;

    Tensor(T* data, shape_t shape) {
        this->data = data;
        this->shape = shape;
        this->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    }

    Tensor(shape_t shape) {
        this->data = new T[std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>())];
        this->shape = shape;
        this->size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    }

    Tensor() {
        this->data = nullptr;
        this->shape = {};
        this->size = 0;
    }

    ~Tensor() {
        delete[] this->data;
    }

    bool empty() const {
        return this->data == nullptr;
    }

    void transpose(axis_t axes){
        // TODO
    }

    Tensor<T> operator=(const Tensor<T>& other){
        Tensor<T> out = Tensor<T>(other.shape);
        
        #pragma omp parallel for
        for(int i = 0; i < size; i++){
            out.data[i] = other.data[i];
        }

        return out;
    }

    Tensor<T> operator/(const uint32_t& scalar){
        if(this->empty()){
            return Tensor<T>();
        }

        Tensor<T> out = Tensor<T>(this->shape);

        #pragma omp parallel for
        for(int i = 0; i < size; i++){
            out.data[i] = this->data[i] / scalar;
        }

        return out;
    }

    void fill_one(){

        #pragma omp parallel for
        for(int i = 0; i < size; i++){
            this->data[i] = 1;
        }
    }
};


// @todo: needs to be optimized
template <typename T>
void gemm(T** a, T** b, T** c, int M, int N, int K)
{
  #pragma omp parallel for
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      T sum = 0;
      for(int k = 0; k < K; k++){
        sum += (*a)[i * K + k] * (*b)[k * N + j];
      }
      (*c)[i * N + j] = sum;
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

  uint32_t M = a->shape[a->shape.size() - 2];
  uint32_t Ka = a->shape[a->shape.size() - 1];
  uint32_t Kb = b->shape[b->shape.size() - 2];
  uint32_t N = b->shape[b->shape.size() - 1];

  assert(Ka == Kb && "The last two dimensions of A and B must be the same");
  assert(a->shape.size() == 4 && b->shape.size() == 2 && "A has to be a 4D tensor and B has to be a 2D tensor");

  // here we are assuming size A > size B, 
  // fow now let's just consider A has dim 4 and B has dim 2
  uint32_t total_size = std::accumulate(a->shape.begin(), a->shape.end(), 1, std::multiplies<uint32_t>());
  uint32_t nBatch = a->shape[0] * a->shape[1];

  T* b_ptr = b->data;

  Tensor<T> out = Tensor<T>({a->shape[0], a->shape[1], M, N});

  #pragma omp parallel for
  for(int i = 0; i < nBatch; i++){
    T* a_ptr = a->data + i * M * Ka;
    T* c_ptr = out.data + i * M * N;
    gemm(&a_ptr, &b_ptr, &c_ptr, M, N, Ka);
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