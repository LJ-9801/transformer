#ifndef POSITIONALEMBEDDING_H
#define POSITIONALEMBEDDING_H
#include "core/tensor.h"

class PositionalEmbedding
{
  public:
    PositionalEmbedding(uint32_t max_seq_len, uint32_t d_model):
      _max_seq_len(max_seq_len), _d_model(d_model)
    {
      _positional_embedding = Tensor<float>({1, max_seq_len, d_model});

      #pragma omp parallel for
      for (uint32_t pos = 0; pos < max_seq_len; ++pos) {
        for (uint32_t i = 0; i < d_model; ++i) {
          if (i % 2 == 0) {
            _positional_embedding.at({0, pos, i}) = sin(pos / pow(10000, (2 * i) / d_model));
          } else {
            _positional_embedding.at({0, pos, i}) = cos(pos / pow(10000, (2 * i) / d_model));
          }
        }
      }
    }

    Tensor<float> forward(Tensor<float> x) {
      x *= sqrt(_d_model);
      uint32_t seq_len = x.shape()[1];

      #pragma omp parallel for
      for(uint32_t i = 0; i < x.shape()[0]; i++){
        for(uint32_t j = 0; j < x.shape()[2]; j++){
          for(uint32_t k = 0; k < seq_len; k++){
            x.at({i, k, j}) += _positional_embedding.at({0, k, j});
          }
        }
      }

      return x;
    }



  private:
    uint32_t _max_seq_len;
    uint32_t _d_model;
    Tensor<float> _positional_embedding;
};


#endif