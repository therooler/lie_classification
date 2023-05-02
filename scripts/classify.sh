#/bin/bash
g++ -std=c++11 ./dyn_lie/classification_su4_raw.cpp -o ./dyn_lie/classify_su4_raw.out
g++ -std=c++11 ./dyn_lie/classification_sun.cpp -o ./dyn_lie/classify.out

# Get the large set of su4 algebras
./dyn_lie/classify_su4_raw.out 0
./dyn_lie/classify_su4_raw.out 1

for N in 2 3 4 5 6 7
do
    ./dyn_lie/classify.out $N 0
    ./dyn_lie/classify.out $N 1
done

