#ifndef LINEAR_H
#define LINEAR_H
#include "common/kernels.h"

template <bool IsBias>
class Linear{
  public:
  Linear(uint32_t in_features, uint32_t out_features):
    _in_features(in_features), 
    _out_features(out_features), 
    _weight(Tensor<float>({out_features, in_features})),
    _bias(IsBias ? Tensor<float>({out_features}) : Tensor<float>()) 
    {
      this->generate_weights();
    }

  void generate_weights(){
    this->_weight = this->_weight.fill_one();
    if(IsBias){
      this->_bias = this->_bias.fill_one();
    } 
  }

  void load_weights(Tensor<float> weight, Tensor<float> bias){
    this->_weight = weight;
    if(IsBias){
      this->_bias = bias;
    } 
  }

  bool isbias() const {
    return IsBias;
  }

  Tensor<float> forward(Tensor<float>& input){
    assert(input.ndim() > 1 && "Input tensor must have at least 2 dimensions");
    shape_t shape = shape_t();
    for(size_t i = 0; i < input.shape().size() - 1; i++){
      shape.push_back(input.shape()[i]);
    }
    shape.push_back(this->_out_features);

    Tensor<float> out = Tensor<float>(shape);
    uint32_t batch_size = multiply_accumulate(shape.begin(), shape.end() - 2);
    uint32_t slices = out.size() / batch_size;

    uint32_t M = input.shape()[input.ndim() - 2];
    uint32_t N = this->_out_features;
    uint32_t K = this->_in_features;

    #pragma omp parallel for
    for(int i = 0; i < batch_size; i++){
      
      gemm_nt<float>(
              accessor<float>::const_ptr(input) + i * M * K, 
              accessor<float>::const_ptr(this->_weight),
              IsBias ? nullptr : accessor<float>::const_ptr(this->_bias),
              accessor<float>::get(out) + i * M * N, 
              M, N, K);

     /*
      float bias_value = accessor<float>::get(this->_bias) == nullptr ? 0 : accessor<float>::get(this->_bias)[i * slices];
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0, 
                  accessor<float>::const_ptr(input) + i * M * K, K, 
                  accessor<float>::const_ptr(this->_weight), K, 
                  bias_value,  
                  accessor<float>::get(out) + i * M * N, N);
      */
      // @todo: this could be fused into the gemm_nt?
      /*
      if(this->_isbias){
        #pragma unrroll 
        for(int j = 0; j < slices; j++){
          float* out_ptr = accessor<float>::get(out) + i * slices + j;
          float* bias_ptr = accessor<float>::const_ptr(this->_bias);
          *out_ptr += *bias_ptr; 
        }
      }
      */
    }
                                         
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