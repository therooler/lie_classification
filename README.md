# The Classification of Dynamical Lie algebras for translationally invariant 2-local Hamiltonians in one dimension

This repository contains C++ code to generate all unique dynamical Lie algebras for Hamiltonians of the form

$$
    H = \sum_{i=1}^L A_i B_{i+1}
$$
where $A_i, B_i \in \{I,\sigma^X, \sigma^Y, \sigma^Z\}$.

To reproduce the figures install `Python==3.9+` and the `requirements.txt` file. Assuming you have a g++ compiler that can compile C++11 code, run
```bash
sh main.sh
```
to automatically create the folders, data and figures.


    