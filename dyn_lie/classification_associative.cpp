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
        throw std::invalid_argument("Expected 3 arguments for classification: `N`, `add_I` and `closed`");
    }

    std::istringstream iss_N(argv[1]);
    std::istringstream iss_add_I(argv[2]);
    std::istringstream iss_closed(argv[3]);

    int N;
    iss_N >> N;
    bool add_I;
    iss_add_I >> add_I;
    bool closed;
    iss_closed >> closed;

    get_associative_algebra(N, add_I, closed);
    

    return 0;
}