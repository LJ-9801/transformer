#ifndef TRANSFORMERENCODER_H
#define TRANSFORMERENCODER_H
#include "transformerblock.h"
#include "embedding.h"
#include "positionalEmbedding.h"

class TransformerEncoder{
  public:
  TransformerEncoder(uint32_t seq_len, uint32_t vocab_size, uint32_t embed_dim, 
                    uint32_t n_layers = 2, uint32_t n_head = 8, uint32_t expansion_factor = 4):
                    _seq_len(seq_len), _vocab_size(vocab_size), _embed_dim(embed_dim),
                    _n_layers(n_layers), _n_heads(n_head), _expansion_factor(expansion_factor),
                    embedding(vocab_size, embed_dim), positionalEmbedding(seq_len, embed_dim)
  {
    for(int i = 0; i < _n_layers; i++){
      blocks.push_back(transformerblock(embed_dim, expansion_factor, n_head)); 
    }
  }

  void generate_weights(){
    for(int i = 0; i < _n_layers; i++){
      blocks[i].generate_weights();
    }
  }

  void load_weights(std::vector<std::pair<Tensor<float>, Tensor<float>>> q,
                    std::vector<std::pair<Tensor<float>, Tensor<float>>> k,
                    std::vector<std::pair<Tensor<float>, Tensor<float>>> v){
  
  }

  Tensor<float> forward(Tensor<uint32_t> x){
    auto emb = embedding.forward(x);
    auto pos_emb = positionalEmbedding.forward(emb);
    for(int i = 0; i < _n_layers; i++){
      pos_emb = blocks[i].forward(pos_emb, pos_emb, pos_emb);
    }
    return pos_emb; 
  }

  private:
  uint32_t _embed_dim;
  uint32_t _n_heads;
  uint32_t _n_layers;
  uint32_t _expansion_factor;
  uint32_t _seq_len;
  uint32_t _vocab_size; 
  std::vector<transformerblock> blocks;
  Embedding embedding;
  PositionalEmbedding positionalEmbedding; 
};

#endif