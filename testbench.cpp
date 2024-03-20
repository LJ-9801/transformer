#include <time.h>
#include <chrono>

#include "core/layernorm.h"
#include "core/linear.h"
#include "transformerblock.h"
#include "positionalEmbedding.h"
#include "embedding.h"
#include "transformerEncoder.h"

int main(){

  auto key = Tensor<uint32_t>({512, 10, 64}).fill_one();
  //auto value = Tensor<float>({512, 10, 64}).fill_one();
  //auto query = Tensor<float>({512, 10, 64}).fill_one();

  auto tmp = key - 10; 
  
  auto tencoder = TransformerEncoder(10, 512, 64);

  tencoder.generate_weights();

    auto start = std::chrono::high_resolution_clock::now();
    auto out = tencoder.forward(key);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() *  1000 << " ms\n";


  return 0;
}