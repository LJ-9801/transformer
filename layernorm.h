#ifndef LAYERNORM_H
#define LAYERNORM_H
#include "kernels.h"
#include "tensor.h"

class LayerNorm{

  public:
  LayerNorm(shape_t normalized_shape, float eps = 1e-5, 
                                      bool elementwise_affine = true,
                                      bool bias = true):
            
            _normalized_shape(normalized_shape),
            _eps(eps),
            _affine(elementwise_affine),
            _isbias(bias) {}

  void generate_weight(){
    this->_weight = _affine ? 
                    Tensor<float>(this->_normalized_shape).fill_one() :
                    Tensor<float>();

    this->_bias = _isbias ?
                  Tensor<float>(this->_normalized_shape).fill_one() :
                  Tensor<float>();

  }

  Tensor<float> forward(Tensor<float> input){

  }


  private:

  shape_t _normalized_shape;
  float _eps;
  bool _affine;
  bool _isbias;

  Tensor<float> _weight;
  Tensor<float> _bias;

};
#endif