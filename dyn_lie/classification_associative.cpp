#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <sstream>
#include "pauli.hh"
#include "pauli_set.hh"
#include "dynamical_lie.hh"
#include "frustration_graph.hh"
#include "dynamical_lie.hh"

int main(int argc, char **argv)
{
    if (!(argc == 3))
    {
        throw std::invalid_argument("Expected 2 arguments for classification: `N` and `add_I`");
    }

    std::istringstream iss_N(argv[1]);
    std::istringstream iss_add_I(argv[2]);
    int N;
    iss_N >> N;
    bool add_I;
    iss_add_I >> add_I;

    get_associative_algebra(N, add_I);
    

    return 0;
}