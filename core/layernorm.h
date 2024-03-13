#ifndef LAYERNORM_H
#define LAYERNORM_H
#include "common/kernels.h"
#include "tensor.h"

class LayerNorm{

  public:
  LayerNorm(shape_t normalized_shape, float eps = 1e-5, 
                                      bool elementwise_affine = true,
                                      bool bias = true):
            
            _normalized_shape(normalized_shape),
            _norm_size(multiply_accumulate(normalized_shape.begin(), 
                                           normalized_shape.end())),
            _eps(eps),
            _affine(elementwise_affine),
            _isbias(bias) 
  {
    this->generate_weight();
  }

  void generate_weight(){
    this->_weight = _affine ? 
                    Tensor<float>(this->_normalized_shape).fill_one() :
                    Tensor<float>();

    this->_bias = _isbias ?
                  Tensor<float>(this->_normalized_shape).fill_one() :
                  Tensor<float>();

  }

  Tensor<float> forward(Tensor<float> input){

    uint32_t n_batches = input.size() / _norm_size;

    Tensor<float> output = Tensor<float>(input.shape());

    #pragma omp parallel for
    for(unsigned int i = 0; i < n_batches; i++){
      
      float m = mean(i, input.data());
      float v = var(i, input.data(), m);
      
      #pragma unroll
      for(unsigned int j = 0; j < _norm_size; j++){
        uint32_t pos[_normalized_shape.size()];

        for(unsigned int k = 0; k < _normalized_shape.size(); k++){
          pos[k] = (j / multiply_accumulate(
                           _normalized_shape.begin(),
                           _normalized_shape.begin() + k)) % _normalized_shape[k];
        }

        int idx = index_from_shape(pos, _normalized_shape.data(), 
                                            _normalized_shape.size());

        float* out_ptr = accessor<float>::get(output);
        
        out_ptr[i * _norm_size + j] = 
          _weight[idx] * (input[i * _norm_size + j] - m) /
          std::sqrt(v + _eps) + _bias[idx];
      }
    }
    return output;
  }


  private:

  float mean(uint32_t i, float* input){
    float mean = 0;
    #pragma unroll
    for(unsigned int j = 0; j < _norm_size; j++){
      mean += input[i * _norm_size + j];
    }
    mean /= this->_norm_size;
    return mean;
  }

  float var(uint32_t i, float* input, float m){
    float variance = 0;
    #pragma unroll
    for(unsigned int j = 0; j < _norm_size; j++){
      variance += pow(input[i * _norm_size + j] - m, 2);
    }
    variance /= this->_norm_size;
    return variance;
  }

  shape_t _normalized_shape;
  size_t _norm_size;
  float _eps;
  bool _affine;
  bool _isbias;

  Tensor<float> _weight;
  Tensor<float> _bias;

};
#endif