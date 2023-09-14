# Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension: C++ code

This code was used as an inspiration for the proofs in https://arxiv.org/abs/2309.05690 .

This repository contains C++ code to generate all unique dynamical Lie algebras for Hamiltonians of the form

$$H = \sum_{i=1}^n A_i B_{i+1}$$

where $A_i, B_i \in \{I,\sigma^X, \sigma^Y, \sigma^Z\}$ up to $n=7$.

To verify that the code is working correctly, run 
```bash
sh ./scripts/test.sh
```

To reproduce the figures install `Python==3.9+` and the `requirements.txt` file. Assuming you have a g++ compiler that can compile C++11 code, run
```bash
sh main.sh
```
to automatically create the folders, data and figures for up to $n=7$.

The data is structured as follows:

```text
data -> closed -> su4
               -> su4_I
               -> su8
                  ...
     
data -> open   -> su4
               -> su4_I
               -> su8
                  ...
        
```
The `data` folder contains the folders `open` and `closed`, corresponding to open
end closed boundary conditions, respectively. Then, in each respective folder there is a subfolder
named `su_<2^n>` and `su_<2^n>_I`. The former corresponds to the $\mathfrak{a}$-type algebras, whereas the latter
contains the $\mathfrak{b}$ type algebras. In each folder, we save the following text files.

- Each folder has a file called `meta.text`, which contains
information about how many unique DLAs of a specific dimensionality are found. 
For example, `data/closed/su8` contains
```text
dim,count
3,1
6,1
10,1
12,1
15,3
21,1
28,2
30,6
63,7
```
Hence we find 1 DLA of dimension 3, 1 of dimension 6, etc.. 
- For each of the $k$ DLAs
we save a file `pauliset_<k>.txt`, which contains the Paulistrings that form a basis for the DLA. 
- For up to $n=5$ we also calculate the associative algebra and store it in the text file `associative_<k>.txt`
We also include the file `meta_associative.txt` which again counts the number algebras with a specific dimension.
- Finally, for up to $n=6$, we calculate the commutants of the DLAs and save them as `commutant_<k>.txt`.