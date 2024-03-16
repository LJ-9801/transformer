#include "multiheadattention.h"
#include "core/layernorm.h"
#include "core/linear.h"
#include "transformerblock.h"

int main(){

  auto key = Tensor<float>({4, 2, 4}).arange();
  auto value = Tensor<float>({4, 2, 4}).arange();
  auto query = Tensor<float>({4, 2, 4}).arange();

  auto tmp = key - 10; 
  
  auto tb = transformerblock(4, 2, 4);

  auto out = tb.forward(key, query, value); 

  return 0;
}