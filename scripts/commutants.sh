#/bin/bash
# use -O3 for optimized code
g++ -std=c++11 -O3 ./dyn_lie/classification_commutants.cpp -o ./dyn_lie/classify_commutants.out

for N in 2 3 4 5 6 7 8
do
    ./dyn_lie/classify_commutants.out $N 0
    ./dyn_lie/classify_commutants.out $N 1
done

