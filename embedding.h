#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "core/tensor.h"

class Embedding
{
  public:
    Embedding(uint32_t vocab_size, uint32_t d_model):
      _vocab_size(vocab_size), _d_model(d_model)
    {
      _embedding = Tensor<float>({vocab_size, d_model});
    }

    void generate_weights(){
    }

  Tensor<float> forward(Tensor<uint32_t> x) {
    Tensor<float> out = Tensor<float>({x.shape()[0], x.shape()[1], _d_model});

    for(uint32_t i = 0; i < x.shape()[0]; i++){
      for(uint32_t j = 0; j < x.shape()[1]; j++){
        for(uint32_t k = 0; k < _d_model; k++){
          out.at({i, j, k}) = _embedding.at({x.at({i, j}), k});
        }
      }
    }
    return out;
  }


  private:
    uint32_t _vocab_size;
    uint32_t _d_model;
    Tensor<float> _embedding;
};


#endif