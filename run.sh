# /usr/bin/bash

echo "compiling"
g++ testbench.cpp -O3 -fopenmp -lopenblas
echo "running"
./a.out
