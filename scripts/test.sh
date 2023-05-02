#/bin/bash
g++ -std=c++11 ./tests/test_pauli.cpp -o ./tests/test_pauli.out
g++ -std=c++11 ./tests/test_pauli_set.cpp -o ./tests/test_pauli_set.out
g++ -std=c++11 ./tests/test_frustration_graph.cpp -o ./tests/test_frustration_graph.out

./tests/test_pauli.out
./tests/test_pauli_set.out
./tests/test_frustration_graph.out
