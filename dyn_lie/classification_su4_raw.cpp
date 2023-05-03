#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include "pauli.hh"
#include "pauli_set.hh"
#include "frustration_graph.hh"

void write_pauli_unordered_set_verbose_su4(PSVec pv, std::ofstream &myfile)
{

    int l = 0;
    int s_pauli = 0;
    int d_pauli = 0;
    int d_dis_pauli = 0;
    // Distinguish single paulis by (#pairs, #left paulis, #right paulis)
    for (PSVec::iterator jt = pv.begin(); jt != pv.end(); ++jt)
    {
        if (((*jt)[0] == 0 & (*jt)[2] == 0) | ((*jt)[1] == 0 & (*jt)[3] == 0))
        {
            s_pauli += 1;
        }
        else if (((*jt)[0] == (*jt)[1]) & ((*jt)[2] == (*jt)[3]))
        {
            d_pauli += 1;
        }
        else if (((*jt)[0] != (*jt)[2]) | ((*jt)[1] != (*jt)[3]))
        {
            d_dis_pauli += 1;
        }

        if (l < (pv.size() - 1))
        {
            myfile << (*jt).to_str() << ", ";
        }
        else
        {
            myfile << (*jt).to_str();
        }
        l += 1;
    }
    myfile << " - size = " << pv.size() << " - (" << s_pauli << "," << d_pauli << "," << d_dis_pauli << ")\n";
}

void print_pauli_unordered_set_verbose_su4(PSVec pv)
{
    std::cout << "{";
    int l = 0;
    int s_pauli = 0;
    int d_pauli = 0;
    int d_dis_pauli = 0;
    // Distinguish single paulis by (#pairs, #left paulis, #right paulis)
    for (PSVec::iterator jt = pv.begin(); jt != pv.end(); ++jt)
    {
        // Count the number of single paulis (IC or CI where C in (X,Y,Z))
        if (((*jt)[0] == 0 & (*jt)[2] == 0) | ((*jt)[1] == 0 & (*jt)[3] == 0))
        {
            s_pauli += 1;
        }
        // Count the number of equal double paulis (XX, YY, ZZ)
        else if (((*jt)[0] == (*jt)[1]) & ((*jt)[2] == (*jt)[3]))
        {
            d_pauli += 1;
        }
        // Count the number of distinct double paulis (XY, YZ, etc.)
        else if (((*jt)[0] != (*jt)[2]) | ((*jt)[1] != (*jt)[3]))
        {
            d_dis_pauli += 1;
        }

        if (l < (pv.size() - 1))
        {
            std::cout << (*jt).to_str() << ", ";
        }
        else
        {
            std::cout << (*jt).to_str();
        }
        l += 1;
    }
    std::cout << "} - size = " << pv.size() << " - (" << s_pauli << "," << d_pauli << "," << d_dis_pauli << ")\n";
}

int main(int argc, char **argv)
{
    if (!(argc == 2))
    {
        throw std::invalid_argument("Expected 1 arguments for classification: `add_I`");
    }
    std::istringstream iss_add_I(argv[1]);

    bool add_I;
    iss_add_I >> add_I;
    if (add_I)
    {
        std::cout << "Adding I to Pauli strings" << std::endl;
    }
    std::vector<std::vector<std::string>> result;
    PSVec paulistrings;
    std::vector<std::string> v1;

    if (add_I)
    {
        v1.push_back("I");
    }

    v1.push_back("X");
    v1.push_back("Y");
    v1.push_back("Z");
    result = cartesian_product<std::string>(v1, v1);
    for (std::vector<std::vector<std::string>>::iterator itr = result.begin(); itr != result.end(); ++itr)
    {
        std::string s;
        for (std::vector<std::string>::iterator jtr = (*itr).begin(); jtr != (*itr).end(); ++jtr)
        {
            s = s + (*jtr);
        }
        if (!(s == "II"))
        {
            paulistrings.push_back(PauliString(2, s));
        }
    }
    std::unordered_set<PSSet, PSHashFunction> all_subalgebras;
    std::vector<PSVec> all_ps = get_all_subsets(paulistrings);

    PSSet pset;
    int counts[16];
    for (unsigned i = 0; i < 16; i++)
    {
        counts[i] = 0;
    }
    for (std::vector<PSVec>::iterator itr = all_ps.begin(); itr != all_ps.end(); ++itr)
    {
        pset.clear();
        // print_pauli_vector(*itr);
        for (PSVec::iterator it = (*itr).begin(); it != (*itr).end(); ++it)
        {
            pset.insert(*it);
            // If we add and I, make sure that we add the respective pair as well.
            if (add_I)
            {
                if ((*it)[0] == 0 & (*it)[2] == 0)
                {
                    // std::cout << (*it).to_str() << std::endl;
                    std::string pair_pauli_s = (*it).to_str()[1] + std::string(1, 'I');
                    // std::cout << pair_pauli_s << std::endl;
                    pset.insert(PauliString(2, pair_pauli_s));

                }
                if ((*it)[1] == 0 & (*it)[3] == 0)
                {
                    // std::cout << (*it).to_str() << std::endl;
                    std::string pair_pauli_s = std::string(1, 'I') + (*it).to_str()[0];
                    // std::cout << pair_pauli_s << std::endl;
                    pset.insert(PauliString(2, pair_pauli_s));
                }
            }
        }
        PSSet temp_pset(pset);

        for (PSSet::iterator it = pset.begin(); it != pset.end(); ++it)
        {
            nested_commutator(*it, temp_pset);
        }

        if (counts[temp_pset.size()] == 1)
        {
            std::cout << "Dim = " << temp_pset.size() << std::endl;
            std::cout << "Final contains the elements (";
            for (PSSet::iterator itr = temp_pset.begin(); itr != temp_pset.end(); ++itr)
            {
                std::cout << (*itr).to_str() << ",";
            }
            std::cout << ")" << std::endl;
        }

        counts[temp_pset.size()] += 1;
        all_subalgebras.insert(temp_pset);
    }
    std::cout << "Dimensions of the subsets: " << std::endl;
    std::cout << "(";
    int total = 0;
    for (unsigned i = 0; i < 16; i++)
    {
        std::cout << counts[i] << ", ";
        total += counts[i];
    }
    std::cout << ")\n";
    std::cout << "Total =" << total << std::endl;
    std::cout << "Size of unique basis elements " << all_subalgebras.size() << std::endl;

    std::vector<PSVec> ps_vector;
    std::vector<PSVec> ps_vector_ordered;
    for (std::unordered_set<PSSet, PSHashFunction>::iterator kt = all_subalgebras.begin(); kt != all_subalgebras.end(); ++kt)
    {
        ps_vector.push_back(PSVec((*kt).begin(), (*kt).end()));
    }
    std::sort(ps_vector.begin(), ps_vector.end(), [](const PSVec &lhs, const PSVec &rhs)
              { return lhs.size() < rhs.size(); });

    std::ofstream myfile;
    if (add_I)
    {
        myfile.open("./data/su4_I_raw/all_unique_su4.txt");
    }
    else
    {
        myfile.open("./data/su4_raw/all_unique_su4.txt");
    }
    myfile << "{<the set>} - <size of the set> - (<#single paulis>,<#double equal paulis>,<#double different paulis>)" << std::endl;

    for (std::vector<PSVec>::iterator kt = ps_vector.begin(); kt != ps_vector.end(); ++kt)
    {
        FrustrationGraph fg(*kt);
        write_pauli_unordered_set_verbose_su4(*kt, myfile);
    }
}