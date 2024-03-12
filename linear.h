#ifndef LINEAR_H
#define LINEAR_H
#include "operations.h"

class Linear{
  public:
  Linear(uint32_t in_features, uint32_t out_features, bool bias = true):
    _in_features(in_features), 
    _out_features(out_features), 
    _isbias(bias), 
    _weight(Tensor<float>({out_features, in_features})),
    _bias(_isbias ? Tensor<float>({out_features}) : Tensor<float>()) 
    {
      this->generate_weights();
    }

  void generate_weights(){
    this->_weight.fill_one();
    if(this->_isbias){
      this->_bias.fill_one();
    } 
  }

  void load_weights(Tensor<float> weight, Tensor<float> bias){
    this->_weight = weight;
    this->_bias = bias;
  }

  Tensor<float> forward(Tensor<float> input){
    Tensor<float> out = batch_matmul<float>(&input, &this->_weight);
    return out;
  }

  private:
  uint32_t _in_features;
  uint32_t _out_features;
  bool _isbias;
  Tensor<float> _weight;
  Tensor<float> _bias;
};
#endif