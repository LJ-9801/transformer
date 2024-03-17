#include <time.h>
#include <chrono>

#include "core/layernorm.h"
#include "core/linear.h"
#include "transformerblock.h"

int main(){

  auto key = Tensor<float>({512, 10, 64}).fill_one();
  auto value = Tensor<float>({512, 10, 64}).fill_one();
  auto query = Tensor<float>({512, 10, 64}).fill_one();

  auto tmp = key - 10; 
  
  auto tb = transformerblock(64, 4, 8);

    auto start = std::chrono::high_resolution_clock::now();
    auto out = tb.forward(key, query, value);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() *  1000 << " ms\n";


  return 0;
}