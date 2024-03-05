# /usr/bin/bash

echo "compiling"
clang++ testbench.cpp -O3
echo "running"
./a.out