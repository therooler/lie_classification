#/bin/bash
# use -O3 for optimized code
g++ -std=c++11 -O3 ./dyn_lie/classification_su4_raw.cpp -o ./dyn_lie/classify_su4_raw.out
g++ -std=c++11 -O3 ./dyn_lie/classification_sun.cpp -o ./dyn_lie/classify_sun.out

# Get the large set of su4 algebras
./dyn_lie/classify_su4_raw.out 0 0
./dyn_lie/classify_su4_raw.out 1 0
./dyn_lie/classify_su4_raw.out 0 1 
./dyn_lie/classify_su4_raw.out 1 1
for N in 2 3 4 5 6 7
do
    # # Open boundary without I
    ./dyn_lie/classify_sun.out $N 0 0
    # Open boundary with I
    ./dyn_lie/classify_sun.out $N 1 0
    # Closed boundary without I
    ./dyn_lie/classify_sun.out $N 0 1
    # Closed boundary with I
    ./dyn_lie/classify_sun.out $N 1 1
done

