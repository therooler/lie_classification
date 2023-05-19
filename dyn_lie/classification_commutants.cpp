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
    std::cout<<"Getting su("<<pow(2,N)<<") commutants \n";
    bool add_I;
    iss_add_I >> add_I;
    PSVec commutant;
    std::string filename;
    std::ofstream myfile;
    for (unsigned k= 0; k < 23; k++)
    {
        commutant = get_commutant(N, k, add_I);
        filename = get_pauliset_filename(N, add_I) + "commutant_" + std::to_string(k) + ".txt"; 
        myfile.open(filename);
        int l = 0;
        std::sort(commutant.begin(), commutant.end(), [](const PauliString &lhs, const PauliString &rhs)
              { return lhs<rhs;});
        for (PSVec::iterator it = commutant.begin(); it != commutant.end(); ++it)
        {
            if (l < (commutant.size() - 1))
            {
                myfile << (*it).to_str() << ",";
            }
            else
            {
                myfile << (*it).to_str();
            }
            l += 1;
        }
        myfile.close();
        myfile.clear();
    }
    return 0;
}