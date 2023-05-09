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
    if (!(argc == 4))
    {
        throw std::invalid_argument("Expected 3 arguments for classification: `k`, `N` and `add_I`");
    }

    std::istringstream iss_k(argv[1]);
    std::istringstream iss_N(argv[2]);
    std::istringstream iss_add_I(argv[3]);
    int k;
    iss_k >> k;
    int N;
    iss_N >> N;
    bool add_I;
    iss_add_I >> add_I;

    get_dynamical_lie_algebra_A_k(k, N, add_I);
    

    return 0;
}