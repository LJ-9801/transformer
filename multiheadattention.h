#include "common.h"


class Multiheadattention{
  public:

  Multiheadattention(uint32_t embed_dim, uint32_t n_heads):
    _embed_dim(embed_dim), 
    _n_heads(n_heads), 
    _single_head_dim(embed_dim / n_heads),
    q(),k(), v(), out_weight(), out_bias() {}

  void generate_weights(){
    this->q = Tensor<float>({_single_head_dim, _single_head_dim});
    this->k = Tensor<float>({_single_head_dim, _single_head_dim});
    this->v = Tensor<float>({_single_head_dim, _single_head_dim});
    this->out_weight = Tensor<float>({_n_heads * _single_head_dim, _embed_dim});
    this->out_bias = Tensor<float>({_embed_dim});

    this->q.fill_one();
    this->k.fill_one();
    this->v.fill_one();
    this->out_weight.fill_one();
    this->out_bias.fill_one();
  }

  void load_weights(Tensor<float> q, Tensor<float> k, Tensor<float> v){
    this->q = q;
    this->k = k;
    this->v = v;
  }

  Tensor<float> forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
    uint32_t batch_size = key.shape()[0];
    uint32_t seq_len = key.shape()[1];
    uint32_t seq_length_query = query.shape()[1];

    key.view({batch_size, seq_len, _n_heads, _single_head_dim});
    query.view({batch_size, seq_length_query, _n_heads, _single_head_dim});
    value.view({batch_size, seq_len, _n_heads, _single_head_dim});

    // this is parallelizable
    auto k_tmp = batch_matmul<float>(&key, &this->k, nullptr);
    auto q_tmp = batch_matmul<float>(&query, &this->q, nullptr);
    auto v_tmp = batch_matmul<float>(&value, &this->v, nullptr);

    // this is parallelizable
    k_tmp.transpose({1, 2});
    q_tmp.transpose({1, 2});
    v_tmp.transpose({1, 2});

    k_tmp.transpose({-1, -2});

    auto product = batch_matmul<float>(&q_tmp, &k_tmp, nullptr);

    product /= sqrt(_single_head_dim);

    auto scores = softmax<float>(&product, -1);

    scores = batch_matmul<float>(&scores, &v_tmp, nullptr);

    scores.transpose({1, 2});

    scores.view({batch_size, seq_length_query, _n_heads * _single_head_dim});

    auto output = batch_matmul<float>(&scores, &this->out_weight, &this->out_bias);

    return output;
  }
  
  private:



  uint32_t _embed_dim;
  uint32_t _n_heads;
  uint32_t _single_head_dim;

  Tensor<float> q;
  Tensor<float> k;
  Tensor<float> v;
  Tensor<float> out_weight;
  Tensor<float> out_bias;
};