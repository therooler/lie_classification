#ifndef PAULISET_HH
#define PAULISET_HH

#include <iostream>
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
    psset.insert(ps);
    for (PSSet::iterator itr = psset.begin(); itr != psset.end(); ++itr)
    {
        if (!(comm(ps, (*itr)))) // If it does not commute, add the result to the set
        {
            PauliString result = ps * (*itr);
            if (!(psset.count(result)))
            {
                psset.insert(result);
                nested_commutator(result, psset);
            }
        }
    }
}

struct PSHashFunction
{
    size_t operator()(const PSSet &p) const
    {
        std::vector<std::string> string_reps;
        for (PSSet::iterator it = p.begin(); it != p.end(); ++it)
        {
            string_reps.push_back((*it).to_str());
        }
        std::sort(string_reps.begin(), string_reps.end());
        std::string s = "";
        for (std::vector<std::string>::iterator jt = string_reps.begin(); jt != string_reps.end(); ++jt)
        {
            s = s + *jt;
        }
        size_t pHash = std::hash<std::string>()(s);
        return pHash;
    }
};

#endif