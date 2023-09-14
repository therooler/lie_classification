#/bin/bash
# use -O3 for optimized code
g++ -std=c++11 -O3 ./dyn_lie/classification_associative.cpp -o ./dyn_lie/classify_associative.out

for N in 2 3 4 5
do
    ./dyn_lie/classify_associative.out $N 0 0
    ./dyn_lie/classify_associative.out $N 1 0
    ./dyn_lie/classify_associative.out $N 0 1
    ./dyn_lie/classify_associative.out $N 1 1
done

