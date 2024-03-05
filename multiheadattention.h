#include "common.h"


class Multiheadattention{
  public:

  Multiheadattention(uint32_t embed_dim, uint32_t n_heads):
    _embed_dim(embed_dim), 
    _n_heads(n_heads), 
    _single_head_dim(embed_dim / n_heads),
    q(),k(), v() {}

  void generate_weights(){
    std::vector<float> q_data(1.0, _embed_dim * _embed_dim);
    std::vector<float> k_data(1.0, _embed_dim * _embed_dim);
    std::vector<float> v_data(1.0, _embed_dim * _embed_dim);

    this->q = Tensor<float>(q_data, {_embed_dim, _embed_dim});
    this->k = Tensor<float>(k_data, {_embed_dim, _embed_dim});
    this->v = Tensor<float>(v_data, {_embed_dim, _embed_dim});
  }

  void load_weights(Tensor<float> q, Tensor<float> k, Tensor<float> v){
    this->q = q;
    this->k = k;
    this->v = v;
  }

  void forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
    uint32_t batch_size = key.shape[0];
    uint32_t seq_len = key.shape[1];
    uint32_t seq_length_query = query.shape[1];

    key.shape = {batch_size, seq_len, _n_heads, _single_head_dim};
    query.shape = {batch_size, seq_length_query, _n_heads, _single_head_dim};
    value.shape = {batch_size, seq_len, _n_heads, _single_head_dim};
  
  }
  
  private:

  uint32_t _embed_dim;
  uint32_t _n_heads;
  uint32_t _single_head_dim;

  Tensor<float> q;
  Tensor<float> k;
  Tensor<float> v;
};