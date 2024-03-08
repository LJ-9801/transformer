#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <omp.h>

// Helper function to compute the index in the 1D array for a given set of indices in the N-dimensional tensor
int computeIndex(const std::vector<int>& indices, const std::vector<int>& shape) {
    int index = 0;
    int multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        index += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return index;
}

// Softmax function for an N-dimensional tensor represented as a 1D array
void softmax(std::vector<double>& tensor, const std::vector<int>& shape, int dim) {
    int totalElements = 1;
    for (int s : shape) {
        totalElements *= s;
    }

    int dimSize = shape[dim];
    int innerStride = 1;
    for (int i = dim + 1; i < shape.size(); ++i) {
        innerStride *= shape[i];
    }
    int outerStride = innerStride * dimSize;

    #pragma omp parallel for
    for (int outer = 0; outer < totalElements / outerStride; ++outer) {
        #pragma unroll
        for (int inner = 0; inner < innerStride; ++inner) {
            // Find the maximum value for numerical stability
            double maxVal = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < dimSize; ++i) {
                int index = outer * outerStride + i * innerStride + inner;
                maxVal = std::max(maxVal, tensor[index]);
            }

            // Compute the sum of exponentials
            double sumExp = 0.0;
            for (int i = 0; i < dimSize; ++i) {
                int index = outer * outerStride + i * innerStride + inner;
                tensor[index] = std::exp(tensor[index] - maxVal); // Subtract maxVal for numerical stability
                sumExp += tensor[index];
            }

            // Normalize to get the softmax probabilities
            for (int i = 0; i < dimSize; ++i) {
                int index = outer * outerStride + i * innerStride + inner;
                tensor[index] /= sumExp;
            }
        }
    }
}


int main() {
    // Example usage
    std::vector<double> tensor = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    std::vector<int> shape = {2, 2, 2, 2}; 
    int dim = 1; // Apply softmax along the second dimension

    softmax(tensor, shape, dim);

    // Print the result
    for (double val : tensor) {
        std::cout << val << std::endl;
    }
    std::cout << std::endl;


    return 0;
}
