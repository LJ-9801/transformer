#include "common.h"


class Multiheadattention{
  public:

  Multiheadattention(uint32_t embed_dim, uint32_t n_heads):
    _embed_dim(embed_dim), 
    _n_heads(n_heads), 
    _single_head_dim(embed_dim / n_heads),
    q(),k(), v(), out_weight(), out_bias() {}

  void generate_weights(){
    std::vector<float> q_data(1.0, _single_head_dim * _single_head_dim);
    std::vector<float> k_data(1.0, _single_head_dim * _single_head_dim);
    std::vector<float> v_data(1.0, _single_head_dim * _single_head_dim);
    std::vector<float> out_data(1.0, _n_heads * _single_head_dim * _embed_dim);
    std::vector<float> out_bias(1.0, _embed_dim);

    this->q = Tensor<float>(q_data, {_single_head_dim, _single_head_dim});
    this->k = Tensor<float>(k_data, {_single_head_dim, _single_head_dim});
    this->v = Tensor<float>(v_data, {_single_head_dim, _single_head_dim});
    this->out_weight = Tensor<float>(out_data, {_n_heads * _single_head_dim, _embed_dim});
    this->out_bias = Tensor<float>(out_bias, {_embed_dim});
  }

  void load_weights(Tensor<float> q, Tensor<float> k, Tensor<float> v){
    this->q = q;
    this->k = k;
    this->v = v;
  }

  Tensor<float> forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
    uint32_t batch_size = key.shape[0];
    uint32_t seq_len = key.shape[1];
    uint32_t seq_length_query = query.shape[1];

    key.shape = {batch_size, seq_len, _n_heads, _single_head_dim};
    query.shape = {batch_size, seq_length_query, _n_heads, _single_head_dim};
    value.shape = {batch_size, seq_len, _n_heads, _single_head_dim};

    auto k_tmp = batch_matmul<float>(&key, &this->k, nullptr);
    auto q_tmp = batch_matmul<float>(&query, &this->q, nullptr);
    auto v_tmp = batch_matmul<float>(&value, &this->v, nullptr);

    k_tmp.transpose({1, 2});
    q_tmp.transpose({1, 2});
    v_tmp.transpose({1, 2});

    k_tmp.transpose({-1, -2});

    auto product = batch_matmul<float>(&q_tmp, &k_tmp, nullptr);

    product = product / sqrt(_single_head_dim);

    auto scores = softmax<float>(&product, -1);

    scores = batch_matmul<float>(&scores, &v_tmp, nullptr);

    scores.transpose({1, 2});

    scores.shape = {batch_size, seq_length_query, _n_heads * _single_head_dim};

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