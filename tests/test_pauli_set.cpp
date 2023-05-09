#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "../dyn_lie/pauli.hh"
#include "../dyn_lie/pauli_set.hh"
#include <math.h>

bool test_cartesian_product_same_v()
{
    std::vector<std::string> v;
    v.push_back("I");
    v.push_back("X");
    v.push_back("Y");
    v.push_back("Z");
    std::vector<std::vector<std::string>> result = cartesian_product<std::string>(v, v);
    return (result.size() == 16);
}

bool test_cartesian_product_different_v()
{
    std::vector<std::string> v1;
    std::vector<std::string> v2;

    v1.push_back("I");
    v1.push_back("X");
    v1.push_back("Y");
    v1.push_back("Z");

    v2.push_back("X");
    v2.push_back("Y");
    v2.push_back("Z");
    std::vector<std::vector<std::string>> result = cartesian_product<std::string>(v1, v2);
    return (result.size() == 12);
}

bool test_cartesian_product_nested(int nest)
{
    std::vector<std::string> v1;

    v1.push_back("I");
    v1.push_back("X");
    v1.push_back("Y");
    v1.push_back("Z");
    std::vector<std::string> concat_v;
    std::vector<std::vector<std::string>> result;
    for (unsigned int n = 0; n < nest; n++)
    {
        if (!n)
        {
            result = cartesian_product<std::string>(v1, v1);
        }
        else
        {
            result = cartesian_product<std::string>(v1, concat_v);
        }
        concat_v.clear();

        for (std::vector<std::vector<std::string>>::iterator itr = result.begin(); itr != result.end(); ++itr)
        {
            std::string s;
            for (std::vector<std::string>::iterator jtr = (*itr).begin(); jtr != (*itr).end(); ++jtr)
            {
                s = s + (*jtr);
            }
            // std::cout << s << std::endl;
            concat_v.push_back(s);
        }
    }
    return (result.size() == pow(4, (nest + 1)));
}

bool test_all_paulistring_subsets()
{
    std::vector<std::vector<std::string>> result;
    PSVec paulistrings;
    std::vector<std::string> v1;
    v1.push_back("I");
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
    std::vector<PSVec> all_ps = get_all_subsets(paulistrings);
    return (all_ps.size() == pow(2, 15) - 1);
}

bool test_nested_commutator_example_1()
{
    PSSet example;
    example.insert(PauliString(2, "XX"));
    example.insert(PauliString(2, "XY"));
    example.insert(PauliString(2, "YX"));
    example.insert(PauliString(2, "ZX"));

    PSSet temp_pset(example);

    for (PSSet::iterator it = example.begin(); it != example.end(); ++it)
    {
        nested_commutator(*it, temp_pset);
    }
    PSSet result(example);
    result.insert(PauliString(2, "XI"));
    result.insert(PauliString(2, "YY"));
    result.insert(PauliString(2, "IZ"));
    result.insert(PauliString(2, "YI"));
    result.insert(PauliString(2, "ZY"));
    result.insert(PauliString(2, "ZI"));
    return (result == temp_pset);
}

bool test_nested_commutator_example_2()
{
    PSSet example;
    example.insert(PauliString(1, "X"));
    example.insert(PauliString(1, "Y"));

    PSSet temp_pset(example);

    for (PSSet::iterator it = example.begin(); it != example.end(); ++it)
    {
        nested_commutator(*it, temp_pset);
    }
    PSSet result(example);
    result.insert(PauliString(1, "Z"));
    return (result == temp_pset);
}

bool test_nested_commutator_example_3()
{
    PSSet example;
    example.insert(PauliString(1, "X"));

    PSSet temp_pset(example);

    for (PSSet::iterator it = example.begin(); it != example.end(); ++it)
    {
        nested_commutator(*it, temp_pset);
    }
    PSSet result(example);

    return (result == temp_pset);
}


int main()
{
if (test_cartesian_product_same_v())
{
    std::cout << "\033[32m"
              << "Passed: test_cartesian_product_same_v"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_cartesian_product_same_v"
              << "\033[0m" << std::endl;
}
if (test_cartesian_product_different_v())
{
    std::cout << "\033[32m"
              << "Passed: test_cartesian_product_different_v"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_cartesian_product_different_v"
              << "\033[0m" << std::endl;
}
if (test_cartesian_product_nested(2))
{
    std::cout << "\033[32m"
              << "Passed: test_cartesian_product_nested_2"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_cartesian_product_nested_2"
              << "\033[0m" << std::endl;
}
if (test_cartesian_product_nested(3))
{
    std::cout << "\033[32m"
              << "Passed: test_cartesian_product_nested_3"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_cartesian_product_nested_3"
              << "\033[0m" << std::endl;
}
if (test_all_paulistring_subsets())
{
    std::cout << "\033[32m"
              << "Passed: test_all_paulistring_subsets"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_all_paulistring_subsets"
              << "\033[0m" << std::endl;
}
if (test_nested_commutator_example_1())
{
    std::cout << "\033[32m"
              << "Passed: test_nested_commutator_example_1"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_nested_commutator_example_1"
              << "\033[0m" << std::endl;
}
if (test_nested_commutator_example_2())
{
    std::cout << "\033[32m"
              << "Passed: test_nested_commutator_example_2"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_nested_commutator_example_2"
              << "\033[0m" << std::endl;
}
if (test_nested_commutator_example_3())
{
    std::cout << "\033[32m"
              << "Passed: test_nested_commutator_example_3"
              << "\033[0m" << std::endl;
}
else
{
    std::cout << "\033[31m"
              << "Failed: test_nested_commutator_example_3"
              << "\033[0m" << std::endl;
}

}