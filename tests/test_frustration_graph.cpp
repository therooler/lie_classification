#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "../dyn_lie/pauli.hh"
#include "../dyn_lie/pauli_set.hh"
#include "../dyn_lie/frustration_graph.hh"


bool test_frustration_graph_1()
{
    PSVec paulistrings;
    paulistrings.push_back(PauliString(2, "XY"));
    paulistrings.push_back(PauliString(2, "ZZ"));
    paulistrings.push_back(PauliString(2, "XI"));
    paulistrings.push_back(PauliString(2, "YZ"));

   std::vector<PSVec> all_ps = get_all_subsets(paulistrings);
   for (std::vector<PSVec>::iterator it=all_ps.begin(); it!=all_ps.end(); ++it){
        FrustrationGraph fg(*it);
        // fg.print_vertices();
        // fg.print_edges();
   }
   return true;
}

int main()
{
    if (test_frustration_graph_1())
    {
        std::cout << "\033[32m"
                  << "Passed: test_frustration_graph_1"
                  << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m"
                  << "Failed: test_frustration_graph_1"
                  << "\033[0m" << std::endl;
    }
}