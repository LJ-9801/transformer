#include <vector>

typedef std::vector<uint32_t> shape_t;

template <typename T>
struct Tensor
{
    std::vector<T> data;
    shape_t shape;

    Tensor(std::vector<T> data, shape_t shape)
    {
        this->data = data;
        this->shape = shape;
    }

    Tensor()
    {
        this->data = {};
        this->shape = {};
    }
};
