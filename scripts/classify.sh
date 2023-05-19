#/bin/bash
# use -O3 for optimized code
g++ -std=c++11 -O3 ./dyn_lie/classification_su4_raw.cpp -o ./dyn_lie/classify_su4_raw.out
g++ -std=c++11 -O3 ./dyn_lie/classification_sun.cpp -o ./dyn_lie/classify_sun.out

# Get the large set of su4 algebras
./dyn_lie/classify_su4_raw.out 0
./dyn_lie/classify_su4_raw.out 1

for N in 2 3 4 5 6 7 8
do
    ./dyn_lie/classify_sun.out $N 0
    ./dyn_lie/classify_sun.out $N 1
done

