#ifndef PAULI_HH
#define PAULI_HH

#include <iostream>
#include <string>

/* Object for handling Pauli strings that relies on the binary symplectic form
See section 2 of https://quantum-journal.org/papers/q-2020-06-04-278/
The binary symplectic form works as follows. For N = 1 we have
I = (0 | 0)
X = (1 | 0)
Y = (1 | 1)
Z = (0 | 1)
This extends obviously for N>1, for example XYZ = (1,1,0|0,1,1)
By performing modular arithmetic on this array we can implement the Pauli algebra.
*/
class PauliString
{
public:
    // Initialize the 2*n binary array
    PauliString(int n) : n(n)
    {
        arrlen = 2 * n;
        numarr = new int[arrlen];
        for (unsigned i = 0; i < arrlen; i++)
        {
            numarr[i] = 0;
        }
    }
    // Overload initializer with string
    PauliString(int n, std::string str) : n(n)
    {
        arrlen = 2 * n;
        numarr = new int[arrlen];
        for (unsigned i = 0; i < arrlen; i++)
        {
            numarr[i] = 0;
        }
        set(str);
    }
    // Standard copy constructor
    PauliString(const PauliString &obj)
    {
        n = obj.n;
        arrlen = obj.arrlen;
        numarr = new int[arrlen];
        *numarr = *obj.numarr;
        for (unsigned i = 0; i < arrlen; i++)
        {
            numarr[i] = obj.numarr[i];
        }
    }
    ~PauliString()
    {
        delete[] numarr;
    }
    // Return the length of the Pauli string
    int get_n() const
    {
        return n;
    }

    // Set the binary array based on a Pauli string
    void set(std::string str)
    {
        if (str.length() != n)
        {
            throw std::invalid_argument("Input string longer than array");
        }
        // Set the binary elements according to the rules described above.
        for (unsigned i = 0; i < n; i++)
        {
            if (str[i] == 'X')
            {
                numarr[i] = 1;
            }
            else if (str[i] == 'Y')
            {
                numarr[i] = 1;
                numarr[i + n] = 1;
            }
            else if (str[i] == 'Z')
            {
                numarr[i + n] = 1;
            }
            else if (str[i] == 'I')
            {
                numarr[i] = 0;
                numarr[i + n] = 0;
            }
            else
            {
                throw std::invalid_argument("Input string must be X,Y or Z");
            }
        }
    }
    // Convert the binary array to a string representation
    const std::string to_str() const
    {
        std::string str = "";
        for (unsigned i = 0; i < n; i++)
        {
            if ((numarr[i] == 1) & (numarr[i + n] == 0))
            {
                str = str + 'X';
            }
            else if ((numarr[i] == 0) & (numarr[i + n] == 1))
            {
                str = str + 'Z';
            }
            else if ((numarr[i] == 1) & (numarr[i + n] == 1))
            {
                str = str + 'Y';
            }
            else
            {
                str = str + 'I';
            }
        }
        return str;
    }
    // Accessor for element i in the binary array
    const int &operator[](int index) const
    {
        return numarr[index];
    }
    // Mulitply two Pauli strings
    PauliString &operator*=(PauliString const rhs)
    {

        for (unsigned i = 0; i < arrlen; i++)
        {
            (*this).numarr[i] = ((*this).numarr[i] + rhs.numarr[i]) % 2;
        }

        return *this;
    }
    // Assign a new pauli string
    PauliString &operator=(PauliString const rhs)
    {
        if ((*this).arrlen != (rhs).arrlen)
        {
            throw std::invalid_argument("Number of possible digits not equal");
        }
        (*this).arrlen = rhs.arrlen;
        (*this).n = rhs.n;
        for (unsigned i = 0; i < arrlen; i++)
        {
            (*this).numarr[i] = rhs.numarr[i];
        }
        return *this;
    }

    // Compare two Pauli strings
    bool operator==(const PauliString &other_ps) const
    {
        for (unsigned i = 0; i < arrlen; i++)
        {
            if ((*this)[i] != other_ps[i])
            {
                return false;
            }
        }
        return true;
    }
    // Multiplication inplace
    PauliString operator*(const PauliString rhs)
    {
        PauliString temp(*this);
        temp *= rhs;
        return temp;
    }
    // Print the Pauli string (binary)
    void print() const
    {
        for (unsigned i = 0; i < arrlen; i++)
        {
            std::cout << numarr[i];
        }
        std::cout << '\n';
    }
    // Print the Pauli string (string)
    void print_str(bool nl = false) const
    {
        std::string s = (*this).to_str();
        if (nl)
        {
            std::cout << s << std::endl;
        }
        else
        {
            std::cout << s << ", ";
        }
    }
    // Hash function based on string representation
    struct HashFunction
    {
        size_t operator()(const PauliString &p) const
        {
            const std::string s = p.to_str();
            size_t pHash = std::hash<std::string>()(s);
            return pHash;
        }
    };

private:
    int *numarr; // The binary array
    int n;       // Number of paulis
    int arrlen;  // Length of the binary array
};

// Commutator check.
bool comm(const PauliString &a, const PauliString &b)
{
    int n = a.get_n();
    if (n != b.get_n())
    {
        throw std::invalid_argument("a and b must have the same length");
    }
    int a_dot_b = 0;
    int b_dot_a = 0;

    for (unsigned i = 0; i < n; i++)
    {
        a_dot_b += (a[i] & b[i + n]);
        b_dot_a += (a[i + n] & b[i]);
    }

    a_dot_b %= 2;
    b_dot_a %= 2;

    return a_dot_b == b_dot_a;
}

#endif