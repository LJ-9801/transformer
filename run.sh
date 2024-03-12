# /usr/bin/bash

echo "compiling"
g++ testbench.cpp -O3 -fopenmp
echo "running"
./a.out
