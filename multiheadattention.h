#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#include "core/operations.h"
#include "core/linear.h"


class Multiheadattention{
  public:

  Multiheadattention(uint32_t embed_dim, uint32_t n_heads):
    _embed_dim(embed_dim), 
    _n_heads(n_heads), 
    _single_head_dim(embed_dim / n_heads),
    q_linear(_single_head_dim, _single_head_dim, false),
    k_linear(_single_head_dim, _single_head_dim, false),
    v_linear(_single_head_dim, _single_head_dim, false),
    out_linear(_n_heads*_single_head_dim, _embed_dim) {}

  void generate_weights(){

    this->q_linear.generate_weights();
    this->k_linear.generate_weights();
    this->v_linear.generate_weights();
    this->out_linear.generate_weights();
  }

  void load_weights(std::pair<Tensor<float>, Tensor<float>> q,
                    std::pair<Tensor<float>, Tensor<float>> k,
                    std::pair<Tensor<float>, Tensor<float>> v){

    this->q_linear.load_weights(q.first, q.second);
    this->k_linear.load_weights(k.first, k.second);
    this->v_linear.load_weights(v.first, v.second);
  }

  Tensor<float> forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
    uint32_t batch_size = key.shape()[0];
    uint32_t seq_len = key.shape()[1];
    uint32_t seq_length_query = query.shape()[1];

    key.view({batch_size, seq_len, _n_heads, _single_head_dim});
    query.view({batch_size, seq_length_query, _n_heads, _single_head_dim});
    value.view({batch_size, seq_len, _n_heads, _single_head_dim});

    auto k_tmp = k_linear.forward(key);
    auto q_tmp = q_linear.forward(query);
    auto v_tmp = v_linear.forward(value);

    // this is parallelizable
    k_tmp.transpose(1, 2);
    q_tmp.transpose(1, 2);
    v_tmp.transpose(1, 2);

    k_tmp.transpose(-1, -2);

    auto product = batch_matmul<float>(q_tmp, k_tmp);

    product /= sqrt(_single_head_dim);

    auto scores = softmax<float>(product, -1);

    scores = batch_matmul<float>(scores, v_tmp);

    scores.transpose(1, 2);
    scores.view({batch_size, seq_length_query, _n_heads * _single_head_dim});

    auto output = out_linear.forward(scores);

    return output;
  }
  
  private:

  uint32_t _embed_dim;
  uint32_t _n_heads;
  uint32_t _single_head_dim;

  Linear q_linear;
  Linear k_linear;
  Linear v_linear;
  Linear out_linear;
};
#endif