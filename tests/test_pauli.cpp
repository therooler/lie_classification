#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include "../dyn_lie/pauli.hh"

bool test_single_qubit()
{
    PauliString pauli_x(1, "X");
    PauliString pauli_y(1, "Y");
    PauliString pauli_z(1, "Z");
    PauliString res_z = pauli_x * pauli_y;
    PauliString res_x = pauli_y * pauli_z;
    PauliString res_y = pauli_z * pauli_x;

    if (!(res_z.to_str() == "Z"))
    {
        return false;
    }
    if (!(res_x.to_str() == "X"))
    {
        return false;
    }
    if (!(res_y.to_str() == "Y"))
    {
        return false;
    }
    return true;
}
bool test_double_qubit()
{
    PauliString pauli_xi(2, "XI");
    PauliString pauli_yi(2, "YI");
    PauliString pauli_zi(2, "ZI");
    PauliString pauli_iz(2, "IZ");

    PauliString res_zi = pauli_xi * pauli_yi;
    PauliString res_xi = pauli_yi * pauli_zi;
    PauliString res_yi = pauli_zi * pauli_xi;

    if (!(res_zi.to_str() == "ZI"))
    {
        return false;
    }
    if (!(res_xi.to_str() == "XI"))
    {
        return false;
    }
    if (!(res_yi.to_str() == "YI"))
    {
        return false;
    }

    PauliString res_chain = pauli_zi * pauli_iz;
    res_chain = res_chain * pauli_xi;
    res_chain *= pauli_yi;

    if (!(res_chain.to_str() == "IZ"))
    {
        return false;
    }
    return true;
}

int main()
{
    if (test_single_qubit())
    {
        std::cout << "\033[32m"
                  << "Passed: test_single_qubit"
                  << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m"
                  << "Failed: test_single_qubit"
                  << "\033[0m" << std::endl;
    }
    if (test_double_qubit())
    {
        std::cout << "\033[32m"
                  << "Passed: test_double_qubit"
                  << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[31m"
                  << "Failed: test_double_qubit"
                  << "\033[0m" << std::endl;
    }
}