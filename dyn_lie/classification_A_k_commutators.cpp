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
    if (!(argc == 5))
    {
        throw std::invalid_argument("Expected 4 arguments for classification: `k`, `N`, `add_I` and `closed`");
    }
    std::istringstream iss_k(argv[1]);
    std::istringstream iss_N(argv[2]);
    std::istringstream iss_add_I(argv[3]);
    std::istringstream iss_closed(argv[4]);

    int k;
    iss_k >> k;
    int N;
    iss_N >> N;
    bool add_I;
    iss_add_I >> add_I;
    bool closed;
    iss_closed >> closed;

    get_commutators_A_k(k, N, add_I, closed);

    return 0;
}