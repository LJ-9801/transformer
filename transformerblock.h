#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H
#include "multiheadattention.h"
#include "core/layernorm.h"
#include "core/activations.h"


class transformerblock
{
  public:
    transformerblock(uint32_t embed_dim, uint32_t expansion_factor, uint32_t num_heads):
      mattention(embed_dim, num_heads),
      norm1({embed_dim}),
      norm2({embed_dim}),
      linear1(embed_dim, embed_dim * expansion_factor),
      linear2(embed_dim * expansion_factor, embed_dim){}

    Tensor<float> forward(Tensor<float> key, Tensor<float> query, Tensor<float> value){
      auto out = mattention.forward(key, query, value);
      auto res = out + value;
      auto norm1_out = dropout<float, false>(norm1.forward(res), 0.2);
      auto linear1_out = linear1.forward(norm1_out);
      auto relu_out = ReLU<float, false>(linear1_out);
      auto linear2_out = linear2.forward(relu_out);
      auto feed_fwd_residual_out = linear2_out + norm1_out;
      auto norm2_out = dropout<float, false>(norm2.forward(feed_fwd_residual_out), 0.2);

      return norm2_out;

    }



private:
    Multiheadattention mattention;
    LayerNorm norm1;
    LayerNorm norm2;
    Linear linear1;
    Linear linear2;

};


#endif