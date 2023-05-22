#/bin/bash
# use -O3 for optimized code
g++ -std=c++11 -O3 ./dyn_lie/classification_A_k.cpp -o ./dyn_lie/classifify_A_k.out
echo "Running algebra $1 up to $2"
./dyn_lie/classifify_A_k.out $1 $2 0
./dyn_lie/classifify_A_k.out $1 $2 1


