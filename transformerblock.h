#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H
#include "multiheadattention.h"
#include "core/layernorm.h"
#include "core/activations.h"


class transformerblock
{
  public:
    transformerblock(uint32_t embed_dim, uint32_t expansion_factor, uint32_t num_heads):
      _mattention(embed_dim, num_heads),
      _norm1({embed_dim}),
      _norm2({embed_dim}),
      _linear1(embed_dim, embed_dim * expansion_factor),
      _linear2(embed_dim * expansion_factor, embed_dim){}

    Tensor<float> forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
      auto out = _mattention.forward(key, query, value);
      auto res = out + value;
      auto norm1_out = dropout<float, false>(_norm1.forward(res), 0.2);
      auto linear1_out = _linear1.forward(norm1_out);
      auto relu_out = ReLU<float, false>(linear1_out);
      auto linear2_out = _linear2.forward(relu_out);
      auto feed_fwd_residual_out = linear2_out + norm1_out;
      auto norm2_out = dropout<float, false>(_norm2.forward(feed_fwd_residual_out), 0.2);

      return norm2_out;

    }



private:
    Multiheadattention _mattention;
    LayerNorm _norm1;
    LayerNorm _norm2;
    Linear<true> _linear1;
    Linear<true> _linear2;

};


#endif