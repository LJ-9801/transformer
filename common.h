#include <vector>

using namespace std;


typedef vector<uint32_t> shape_t;
typedef vector<int32_t> axis_t;

template <typename T>
struct Tensor
{
    vector<T> data;
    shape_t shape;

    Tensor(vector<T> data, shape_t shape)
    {
        this->data = data;
        this->shape = shape;
    }

    Tensor()
    {
        this->data = {};
        this->shape = {};
    }

    void transpose(axis_t axes){
        // TODO
    }

    Tensor<T> operator/(const uint32_t& scalar){
        // TODO
        return Tensor<T>();
    }
};

template <typename T>
Tensor<T> batch_matmul(const Tensor<T>* a, const Tensor<T>* b, const Tensor<T>* c)
{
  return Tensor<T>();
}

template <typename T>
Tensor<T> softmax(Tensor<T>* t, int axis)
{
  return Tensor<T>();
}