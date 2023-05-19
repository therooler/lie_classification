#ifndef PAULISET_HH
#define PAULISET_HH

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <set>
#include "pauli.hh"

typedef std::unordered_set<PauliString, PauliString::HashFunction> PSSet;
typedef std::vector<PauliString> PSVec;

// Template function for the cartesian product of two vectors
template <typename T>
std::vector<std::vector<T>> cartesian_product(const std::vector<T> &v1, const std::vector<T> &v2)
{
    typedef typename std::vector<T>::const_iterator iterator; // Need to define custom iterator
    std::vector<std::vector<T>> result;
    for (iterator itr = v1.begin(); itr != v1.end(); ++itr)
    {
        for (iterator jtr = v2.begin(); jtr != v2.end(); ++jtr)
        {
            std::vector<T> temp;
            temp.push_back(*itr);
            temp.push_back(*jtr);
            result.push_back(temp);
        }
    }

    return result;
}

// Get the power set of a set of Paulis
std::vector<PSVec> get_all_subsets(PSVec set)
{
    std::vector<PSVec> subset;
    PSVec empty;
    subset.push_back(empty);

    for (int i = 0; i < set.size(); i++)
    {
        std::vector<PSVec> subsetTemp = subset; // making a copy of given 2-d vector.

        for (int j = 0; j < subsetTemp.size(); j++)
            subsetTemp[j].push_back(set[i]); // adding set[i] element to each subset of subsetTemp. like adding {2}(in 2nd iteration  to {{},{1}} which gives {{2},{1,2}}.

        for (int j = 0; j < subsetTemp.size(); j++)
            subset.push_back(subsetTemp[j]); // now adding modified subsetTemp to original subset (before{{},{1}} , after{{},{1},{2},{1,2}})
    }
    subset.erase(subset.begin());
    return subset;
}

// Print a vector of Pauli strings
void print_pauli_vector(PSVec pv)
{
    std::cout << "{";
    int l = 0;
    for (PSVec::iterator jt = pv.begin(); jt != pv.end(); ++jt)
    {
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
    std::cout << "}\n";
}

// Print a set of pauli strings
void print_pauli_unordered_set(const PSSet &pv)
{
    std::cout << "{";
    int l = 0;
    for (PSSet::iterator jt = pv.begin(); jt != pv.end(); ++jt)
    {
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
    std::cout << "} - dim = " << pv.size() << "\n";
}

// Print all the subsets in a vector of Pauli strings
void print_subsets(std::vector<PSVec> sets)
{
    std::cout << "Subsets:" << std::endl;
    for (std::vector<PSVec>::iterator it = sets.begin(); it != sets.end(); ++it)
    {
        print_pauli_vector((*it));
    }
    std::cout << std::endl;
}

// Recursively calculate the nested commutator of a pauli string with a set of pauli strings.
void nested_commutator(PauliString ps, PSSet &psset)
{
    // psset.insert(ps);
    for (PSSet::iterator itr = psset.begin(); itr != psset.end(); ++itr)
    {
        if (!(comm(ps, (*itr)))) // If it does not commute, add the result to the set
        {
            PauliString result = ps * (*itr);
            // if the results is not in the set, keep going.
            if (!(psset.count(result)))
            {
                psset.insert(result);
                nested_commutator(result, psset);
            }
        }
    }
}
void nested_product(PauliString ps, PSSet &psset)
{
    // psset.insert(ps);
    for (PSSet::iterator itr = psset.begin(); itr != psset.end(); ++itr)
    {
        PauliString result = ps * (*itr);
        if (std::find(psset.begin(), psset.end(), result)==psset.end()) // If it does not commute, add the result to the set
        {
            psset.insert(result);
            nested_product(result, psset);
        }
    }
}

struct PSHashFunction
// Hash function for unordered pauli sets.
{
    size_t operator()(const PSSet &p) const
    {
        std::vector<std::string> string_reps;
        // Create a vector of string representations of the paulis in the set.
        for (PSSet::iterator it = p.begin(); it != p.end(); ++it)
        {
            string_reps.push_back((*it).to_str());
        }
        // Sort the list of strings lexicographically.
        std::sort(string_reps.begin(), string_reps.end());
        // Concatenate the sorted strings into one big string: the hash.
        std::string s = "";
        for (std::vector<std::string>::iterator jt = string_reps.begin(); jt != string_reps.end(); ++jt)
        {
            s = s + *jt;
        }
        size_t pHash = std::hash<std::string>()(s);
        return pHash;
    }
};

PSVec get_sun_basis(int N, bool add_I = true)
{
    PSVec paulistrings;
    std::vector<std::string> paulis;
    if (add_I)
    {
        paulis.push_back("I");
    }
    paulis.push_back("X");
    paulis.push_back("Y");
    paulis.push_back("Z");
    std::vector<std::string> vec(paulis);

    std::vector<std::vector<std::string>> result;
    for (unsigned i = 1; i < N; i++)
    {
        result = cartesian_product<std::string>(paulis, vec);
        vec.clear();
        for (std::vector<std::vector<std::string>>::iterator itr = result.begin(); itr != result.end(); ++itr)
        {
            std::string s;
            for (std::vector<std::string>::iterator jtr = (*itr).begin(); jtr != (*itr).end(); ++jtr)
            {
                s = s + (*jtr);
            }
            vec.push_back(s);
        }
    }
    std::string id(N, 'I');
    for (std::vector<std::string>::iterator v = vec.begin(); v != vec.end(); ++v)
    {
        if ((*v) != id)
        {
            paulistrings.push_back(PauliString(N, *v));
        }
    }
    std::cout << paulistrings.size() << std::endl;
    return paulistrings;
}

std::string get_pauliset_filename(int N, bool add_I)
{
    int dim = pow(2, N);
    std::string filename;
    if (add_I)
    {
        filename = "./data/su" + std::to_string(dim) + "_I/";
    }
    else
    {
        filename = "./data/su" + std::to_string(dim) + "/";
    }
    return filename;
};

PSVec load_pauliset(int N, int k, bool add_I)
{
    std::string filename = get_pauliset_filename(N, add_I) + "pauliset_" + std::to_string(k) + ".txt";
    ;
    std::ifstream file(filename); // replace "example.txt" with your file name
    if (file.good() && file.peek() != std::ifstream::traits_type::eof())
    {
        std::cout << "Loading " << filename << std::endl;
    }
    else
    {
        throw std::runtime_error("File " + filename + " does not exist.");
    }
    std::string line;
    PSVec entries;

    // read the second line of the file
    if (std::getline(file, line) && std::getline(file, line))
    {
        // split the line by commas and add each entry to the vector
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(",")) != std::string::npos)
        {
            token = line.substr(0, pos);
            entries.push_back(PauliString(N, token));
            line.erase(0, pos + 1);
        }
        entries.push_back(PauliString(N, line)); // add the last entry
    }
    // print_pauli_vector(entries);
    return entries;
};

PSVec get_commutant(int N, int k, bool add_I)
{
    PSVec A_k = load_pauliset(N, k, add_I);
    PSVec sun = get_sun_basis(N);
    PSVec commutant;
    bool commutes;
    for (PSVec::iterator it = sun.begin(); it != sun.end(); it++)
    {
        commutes = true;
        for (PSVec::iterator jt = A_k.begin(); jt != A_k.end(); jt++)
        {
            if (!(comm(*it, *jt)))
            {
                commutes = false;
                break;
            }
        }
        if (commutes)
        {
            commutant.push_back(*it);
        }
    }
    // std::cout << "A_" << k << " for N = " << N << ":\n";
    // print_pauli_vector(A_k);
    // std::cout << "the commutant is \n";
    // print_pauli_vector(commutant);
    return commutant;
}

#endif