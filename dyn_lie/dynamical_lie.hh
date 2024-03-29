#ifndef DYNLIE_HH
#define DYNLIE_HH

#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_set>
#include "pauli.hh"
#include "pauli_set.hh"
#include "frustration_graph.hh"

std::vector<PSSet> get_all_su4_dynamical_lie_algebras()
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

    PSSet pset;
    std::vector<PSSet> all_sets;

    int counts[16];
    for (unsigned i = 0; i < 16; i++)
    {
        counts[i] = 0;
    }
    for (std::vector<PSVec>::iterator itr = all_ps.begin(); itr != all_ps.end(); ++itr)
    {
        pset.clear();

        for (PSVec::iterator it = (*itr).begin(); it != (*itr).end(); ++it)
        {
            pset.insert(*it);
        }
        PSSet temp_pset(pset);

        for (PSSet::iterator it = pset.begin(); it != pset.end(); ++it)
        {
            nested_commutator(*it, temp_pset);
        }
        all_sets.push_back(temp_pset);
        counts[temp_pset.size()] += 1;
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
    std::cout << "2^15 -1 =" << pow(2, 15) - 1 << std::endl;
    return all_sets;
}

std::vector<PSVec> get_unique_algebras_su8()
{
    std::vector<PSVec> ps_vec;
    // Set 0
    ps_vec.push_back(PSVec());
    ps_vec[0].push_back(PauliString(3, "XXI"));
    ps_vec[0].push_back(PauliString(3, "IXX"));
    // Set 1
    ps_vec.push_back(PSVec());
    ps_vec[1].push_back(PauliString(3, "XYI"));
    ps_vec[1].push_back(PauliString(3, "IXY"));
    // Set 2
    ps_vec.push_back(PSVec());
    ps_vec[2].push_back(PauliString(3, "XYI"));
    ps_vec[2].push_back(PauliString(3, "IXY"));
    ps_vec[2].push_back(PauliString(3, "YXI"));
    ps_vec[2].push_back(PauliString(3, "IYX"));
    // Set 3
    ps_vec.push_back(PSVec());
    ps_vec[3].push_back(PauliString(3, "YZI"));
    ps_vec[3].push_back(PauliString(3, "IYZ"));
    ps_vec[3].push_back(PauliString(3, "XXI"));
    ps_vec[3].push_back(PauliString(3, "IXX"));
    // Set 4
    ps_vec.push_back(PSVec());
    ps_vec[4].push_back(PauliString(3, "YYI"));
    ps_vec[4].push_back(PauliString(3, "IYY"));
    ps_vec[4].push_back(PauliString(3, "XXI"));
    ps_vec[4].push_back(PauliString(3, "IXX"));
    // Set 5
    ps_vec.push_back(PSVec());
    ps_vec[5].push_back(PauliString(3, "YZI"));
    ps_vec[5].push_back(PauliString(3, "IYZ"));
    ps_vec[5].push_back(PauliString(3, "XYI"));
    ps_vec[5].push_back(PauliString(3, "IXY"));
    // Set 6
    ps_vec.push_back(PSVec());
    ps_vec[6].push_back(PauliString(3, "ZZI"));
    ps_vec[6].push_back(PauliString(3, "IZZ"));
    ps_vec[6].push_back(PauliString(3, "YXI"));
    ps_vec[6].push_back(PauliString(3, "IYX"));
    ps_vec[6].push_back(PauliString(3, "XYI"));
    ps_vec[6].push_back(PauliString(3, "IXY"));
    // Set 7
    ps_vec.push_back(PSVec());
    ps_vec[7].push_back(PauliString(3, "ZZI"));
    ps_vec[7].push_back(PauliString(3, "IZZ"));
    ps_vec[7].push_back(PauliString(3, "YYI"));
    ps_vec[7].push_back(PauliString(3, "IYY"));
    ps_vec[7].push_back(PauliString(3, "XXI"));
    ps_vec[7].push_back(PauliString(3, "IXX"));
    // Set 8
    ps_vec.push_back(PSVec());
    ps_vec[8].push_back(PauliString(3, "XZI"));
    ps_vec[8].push_back(PauliString(3, "IXZ"));
    ps_vec[8].push_back(PauliString(3, "IYI"));
    ps_vec[8].push_back(PauliString(3, "IIY"));
    ps_vec[8].push_back(PauliString(3, "XXI"));
    ps_vec[8].push_back(PauliString(3, "IXX"));
    // Set 9
    ps_vec.push_back(PSVec());
    ps_vec[9].push_back(PauliString(3, "XZI"));
    ps_vec[9].push_back(PauliString(3, "IXZ"));
    ps_vec[9].push_back(PauliString(3, "XYI"));
    ps_vec[9].push_back(PauliString(3, "IXY"));
    ps_vec[9].push_back(PauliString(3, "IXI"));
    ps_vec[9].push_back(PauliString(3, "IIX"));
    // Set 10
    ps_vec.push_back(PSVec());
    ps_vec[10].push_back(PauliString(3, "ZXI"));
    ps_vec[10].push_back(PauliString(3, "IZX"));
    ps_vec[10].push_back(PauliString(3, "YZI"));
    ps_vec[10].push_back(PauliString(3, "IYZ"));
    ps_vec[10].push_back(PauliString(3, "XYI"));
    ps_vec[10].push_back(PauliString(3, "IXY"));
    // Set 11
    ps_vec.push_back(PSVec());
    ps_vec[11].push_back(PauliString(3, "YXI"));
    ps_vec[11].push_back(PauliString(3, "IYX"));
    ps_vec[11].push_back(PauliString(3, "XZI"));
    ps_vec[11].push_back(PauliString(3, "IXZ"));
    ps_vec[11].push_back(PauliString(3, "XYI"));
    ps_vec[11].push_back(PauliString(3, "IXY"));
    ps_vec[11].push_back(PauliString(3, "IXI"));
    ps_vec[11].push_back(PauliString(3, "IIX"));
    // Set 12
    ps_vec.push_back(PSVec());
    ps_vec[12].push_back(PauliString(3, "XYI"));
    ps_vec[12].push_back(PauliString(3, "IXY"));
    ps_vec[12].push_back(PauliString(3, "YZI"));
    ps_vec[12].push_back(PauliString(3, "IYZ"));
    ps_vec[12].push_back(PauliString(3, "XXI"));
    ps_vec[12].push_back(PauliString(3, "IXX"));
    ps_vec[12].push_back(PauliString(3, "IZI"));
    ps_vec[12].push_back(PauliString(3, "IIZ"));
    // Set 13
    ps_vec.push_back(PSVec());
    ps_vec[13].push_back(PauliString(3, "YYI"));
    ps_vec[13].push_back(PauliString(3, "IYY"));
    ps_vec[13].push_back(PauliString(3, "YZI"));
    ps_vec[13].push_back(PauliString(3, "IYZ"));
    ps_vec[13].push_back(PauliString(3, "XXI"));
    ps_vec[13].push_back(PauliString(3, "IXX"));
    ps_vec[13].push_back(PauliString(3, "IXI"));
    ps_vec[13].push_back(PauliString(3, "IIX"));
    // Set 14
    ps_vec.push_back(PSVec());
    ps_vec[14].push_back(PauliString(3, "YXI"));
    ps_vec[14].push_back(PauliString(3, "IYX"));
    ps_vec[14].push_back(PauliString(3, "ZII"));
    ps_vec[14].push_back(PauliString(3, "IZI"));
    ps_vec[14].push_back(PauliString(3, "YYI"));
    ps_vec[14].push_back(PauliString(3, "IYY"));
    ps_vec[14].push_back(PauliString(3, "XYI"));
    ps_vec[14].push_back(PauliString(3, "IXY"));
    ps_vec[14].push_back(PauliString(3, "XXI"));
    ps_vec[14].push_back(PauliString(3, "IXX"));
    ps_vec[14].push_back(PauliString(3, "IZI"));
    ps_vec[14].push_back(PauliString(3, "IIZ"));
    // Set 15
    ps_vec.push_back(PSVec());
    ps_vec[15].push_back(PauliString(3, "IYI"));
    ps_vec[15].push_back(PauliString(3, "IIY"));
    ps_vec[15].push_back(PauliString(3, "IXI"));
    ps_vec[15].push_back(PauliString(3, "IIX"));
    ps_vec[15].push_back(PauliString(3, "XXI"));
    ps_vec[15].push_back(PauliString(3, "IXX"));
    ps_vec[15].push_back(PauliString(3, "XYI"));
    ps_vec[15].push_back(PauliString(3, "IXY"));
    ps_vec[15].push_back(PauliString(3, "XZI"));
    ps_vec[15].push_back(PauliString(3, "IXZ"));
    ps_vec[15].push_back(PauliString(3, "IZI"));
    ps_vec[15].push_back(PauliString(3, "IIZ"));
    // Set 16
    ps_vec.push_back(PSVec());
    ps_vec[16].push_back(PauliString(3, "ZXI"));
    ps_vec[16].push_back(PauliString(3, "IZX"));
    ps_vec[16].push_back(PauliString(3, "YXI"));
    ps_vec[16].push_back(PauliString(3, "IYX"));
    ps_vec[16].push_back(PauliString(3, "XZI"));
    ps_vec[16].push_back(PauliString(3, "IXZ"));
    ps_vec[16].push_back(PauliString(3, "XYI"));
    ps_vec[16].push_back(PauliString(3, "IXY"));
    ps_vec[16].push_back(PauliString(3, "XII"));
    ps_vec[16].push_back(PauliString(3, "IXI"));
    ps_vec[16].push_back(PauliString(3, "IXI"));
    ps_vec[16].push_back(PauliString(3, "IIX"));
    // Set 17
    ps_vec.push_back(PSVec());
    ps_vec[17].push_back(PauliString(3, "ZXI"));
    ps_vec[17].push_back(PauliString(3, "IZX"));
    ps_vec[17].push_back(PauliString(3, "XYI"));
    ps_vec[17].push_back(PauliString(3, "IXY"));
    ps_vec[17].push_back(PauliString(3, "XXI"));
    ps_vec[17].push_back(PauliString(3, "IXX"));
    ps_vec[17].push_back(PauliString(3, "YII"));
    ps_vec[17].push_back(PauliString(3, "IYI"));
    ps_vec[17].push_back(PauliString(3, "ZYI"));
    ps_vec[17].push_back(PauliString(3, "IZY"));
    ps_vec[17].push_back(PauliString(3, "IZI"));
    ps_vec[17].push_back(PauliString(3, "IIZ"));
    // Set 18
    ps_vec.push_back(PSVec());
    ps_vec[18].push_back(PauliString(3, "ZYI"));
    ps_vec[18].push_back(PauliString(3, "IZY"));
    ps_vec[18].push_back(PauliString(3, "YYI"));
    ps_vec[18].push_back(PauliString(3, "IYY"));
    ps_vec[18].push_back(PauliString(3, "XZI"));
    ps_vec[18].push_back(PauliString(3, "IXZ"));
    ps_vec[18].push_back(PauliString(3, "XXI"));
    ps_vec[18].push_back(PauliString(3, "IXX"));
    ps_vec[18].push_back(PauliString(3, "XII"));
    ps_vec[18].push_back(PauliString(3, "IXI"));
    ps_vec[18].push_back(PauliString(3, "IYI"));
    ps_vec[18].push_back(PauliString(3, "IIY"));
    // Set 19
    ps_vec.push_back(PSVec());
    ps_vec[19].push_back(PauliString(3, "ZXI"));
    ps_vec[19].push_back(PauliString(3, "IZX"));
    ps_vec[19].push_back(PauliString(3, "YZI"));
    ps_vec[19].push_back(PauliString(3, "IYZ"));
    ps_vec[19].push_back(PauliString(3, "XYI"));
    ps_vec[19].push_back(PauliString(3, "IXY"));
    ps_vec[19].push_back(PauliString(3, "XXI"));
    ps_vec[19].push_back(PauliString(3, "IXX"));
    ps_vec[19].push_back(PauliString(3, "YII"));
    ps_vec[19].push_back(PauliString(3, "IYI"));
    ps_vec[19].push_back(PauliString(3, "ZYI"));
    ps_vec[19].push_back(PauliString(3, "IZY"));
    ps_vec[19].push_back(PauliString(3, "IZI"));
    ps_vec[19].push_back(PauliString(3, "IIZ"));
    // Set 20
    ps_vec.push_back(PSVec());
    ps_vec[20].push_back(PauliString(3, "ZZI"));
    ps_vec[20].push_back(PauliString(3, "IZZ"));
    ps_vec[20].push_back(PauliString(3, "YXI"));
    ps_vec[20].push_back(PauliString(3, "IYX"));
    ps_vec[20].push_back(PauliString(3, "XYI"));
    ps_vec[20].push_back(PauliString(3, "IXY"));
    ps_vec[20].push_back(PauliString(3, "XXI"));
    ps_vec[20].push_back(PauliString(3, "IXX"));
    ps_vec[20].push_back(PauliString(3, "YYI"));
    ps_vec[20].push_back(PauliString(3, "IYY"));
    ps_vec[20].push_back(PauliString(3, "ZII"));
    ps_vec[20].push_back(PauliString(3, "IZI"));
    ps_vec[20].push_back(PauliString(3, "IZI"));
    ps_vec[20].push_back(PauliString(3, "IIZ"));
    // Set 21
    ps_vec.push_back(PSVec());
    ps_vec[21].push_back(PauliString(3, "YXI"));
    ps_vec[21].push_back(PauliString(3, "IYX"));
    ps_vec[21].push_back(PauliString(3, "XYI"));
    ps_vec[21].push_back(PauliString(3, "IXY"));
    ps_vec[21].push_back(PauliString(3, "YII"));
    ps_vec[21].push_back(PauliString(3, "IYI"));
    ps_vec[21].push_back(PauliString(3, "YYI"));
    ps_vec[21].push_back(PauliString(3, "IYY"));
    ps_vec[21].push_back(PauliString(3, "ZYI"));
    ps_vec[21].push_back(PauliString(3, "IZY"));
    ps_vec[21].push_back(PauliString(3, "XXI"));
    ps_vec[21].push_back(PauliString(3, "IXX"));
    ps_vec[21].push_back(PauliString(3, "IIZ"));
    ps_vec[21].push_back(PauliString(3, "IZI"));
    ps_vec[21].push_back(PauliString(3, "IXI"));
    ps_vec[21].push_back(PauliString(3, "XII"));
    ps_vec[21].push_back(PauliString(3, "IZI"));
    ps_vec[21].push_back(PauliString(3, "ZII"));
    ps_vec[21].push_back(PauliString(3, "IZX"));
    ps_vec[21].push_back(PauliString(3, "ZXI"));
    // Set 22
    ps_vec.push_back(PSVec());
    ps_vec[22].push_back(PauliString(3, "YXI"));
    ps_vec[22].push_back(PauliString(3, "IYX"));
    ps_vec[22].push_back(PauliString(3, "IYI"));
    ps_vec[22].push_back(PauliString(3, "IIY"));
    ps_vec[22].push_back(PauliString(3, "XYI"));
    ps_vec[22].push_back(PauliString(3, "IXY"));
    ps_vec[22].push_back(PauliString(3, "YII"));
    ps_vec[22].push_back(PauliString(3, "IYI"));
    ps_vec[22].push_back(PauliString(3, "YYI"));
    ps_vec[22].push_back(PauliString(3, "IYY"));
    ps_vec[22].push_back(PauliString(3, "ZYI"));
    ps_vec[22].push_back(PauliString(3, "IZY"));
    ps_vec[22].push_back(PauliString(3, "XXI"));
    ps_vec[22].push_back(PauliString(3, "IXX"));
    ps_vec[22].push_back(PauliString(3, "IIZ"));
    ps_vec[22].push_back(PauliString(3, "IZI"));
    ps_vec[22].push_back(PauliString(3, "IIX"));
    ps_vec[22].push_back(PauliString(3, "IXI"));
    ps_vec[22].push_back(PauliString(3, "IXI"));
    ps_vec[22].push_back(PauliString(3, "XII"));
    ps_vec[22].push_back(PauliString(3, "IZI"));
    ps_vec[22].push_back(PauliString(3, "ZII"));
    ps_vec[22].push_back(PauliString(3, "IXZ"));
    ps_vec[22].push_back(PauliString(3, "XZI"));
    ps_vec[22].push_back(PauliString(3, "IZX"));
    ps_vec[22].push_back(PauliString(3, "ZXI"));
    ps_vec[22].push_back(PauliString(3, "YZI"));
    ps_vec[22].push_back(PauliString(3, "IYZ"));
    ps_vec[22].push_back(PauliString(3, "ZZI"));
    ps_vec[22].push_back(PauliString(3, "IZZ"));
    return ps_vec;
}

std::vector<PSVec> get_unique_algebras_su4(bool add_I = false)
{
    std::vector<PSVec> ps_vec;
    // Set 0
    ps_vec.push_back(PSVec());
    ps_vec[0].push_back(PauliString(2, "XX"));
    // Set 1
    ps_vec.push_back(PSVec());
    ps_vec[1].push_back(PauliString(2, "XY"));
    // Set 2
    ps_vec.push_back(PSVec());
    ps_vec[2].push_back(PauliString(2, "XY"));
    ps_vec[2].push_back(PauliString(2, "YX"));
    // Set 3
    ps_vec.push_back(PSVec());
    ps_vec[3].push_back(PauliString(2, "YZ"));
    ps_vec[3].push_back(PauliString(2, "XX"));
    // Set 4
    ps_vec.push_back(PSVec());
    ps_vec[4].push_back(PauliString(2, "YY"));
    ps_vec[4].push_back(PauliString(2, "XX"));
    // Set 5
    ps_vec.push_back(PSVec());
    ps_vec[5].push_back(PauliString(2, "YZ"));
    ps_vec[5].push_back(PauliString(2, "XY"));
    // Set 6
    ps_vec.push_back(PSVec());
    ps_vec[6].push_back(PauliString(2, "ZZ"));
    ps_vec[6].push_back(PauliString(2, "YX"));
    ps_vec[6].push_back(PauliString(2, "XY"));
    // Set 7
    ps_vec.push_back(PSVec());
    ps_vec[7].push_back(PauliString(2, "ZZ"));
    ps_vec[7].push_back(PauliString(2, "YY"));
    ps_vec[7].push_back(PauliString(2, "XX"));
    // Set 8
    ps_vec.push_back(PSVec());
    ps_vec[8].push_back(PauliString(2, "XZ"));
    ps_vec[8].push_back(PauliString(2, "IY"));
    ps_vec[8].push_back(PauliString(2, "XX"));
    if (add_I)
    {
        ps_vec[8].push_back(PauliString(2, "YI"));
    }
    // Set 9
    ps_vec.push_back(PSVec());
    ps_vec[9].push_back(PauliString(2, "XZ"));
    ps_vec[9].push_back(PauliString(2, "XY"));
    ps_vec[9].push_back(PauliString(2, "IX"));
    if (add_I)
    {
        ps_vec[9].push_back(PauliString(2, "XI"));
    }
    // Set 10
    ps_vec.push_back(PSVec());
    ps_vec[10].push_back(PauliString(2, "ZX"));
    ps_vec[10].push_back(PauliString(2, "YZ"));
    ps_vec[10].push_back(PauliString(2, "XY"));
    // Set 11
    ps_vec.push_back(PSVec());
    ps_vec[11].push_back(PauliString(2, "YX"));
    ps_vec[11].push_back(PauliString(2, "XZ"));
    ps_vec[11].push_back(PauliString(2, "XY"));
    ps_vec[11].push_back(PauliString(2, "IX"));
    // Set 12
    ps_vec.push_back(PSVec());
    ps_vec[12].push_back(PauliString(2, "XY"));
    ps_vec[12].push_back(PauliString(2, "YZ"));
    ps_vec[12].push_back(PauliString(2, "XX"));
    ps_vec[12].push_back(PauliString(2, "IZ"));
    if (add_I)
    {
        ps_vec[12].push_back(PauliString(2, "ZI"));
    }
    // Set 13
    ps_vec.push_back(PSVec());
    ps_vec[13].push_back(PauliString(2, "YY"));
    ps_vec[13].push_back(PauliString(2, "YZ"));
    ps_vec[13].push_back(PauliString(2, "XX"));
    ps_vec[13].push_back(PauliString(2, "IX"));
    if (add_I)
    {
        ps_vec[13].push_back(PauliString(2, "XI"));
    }
    // Set 14
    ps_vec.push_back(PSVec());
    ps_vec[14].push_back(PauliString(2, "YX"));
    ps_vec[14].push_back(PauliString(2, "ZI"));
    ps_vec[14].push_back(PauliString(2, "YY"));
    ps_vec[14].push_back(PauliString(2, "XY"));
    ps_vec[14].push_back(PauliString(2, "XX"));
    ps_vec[14].push_back(PauliString(2, "IZ"));

    // Set 15
    ps_vec.push_back(PSVec());
    ps_vec[15].push_back(PauliString(2, "IY"));
    ps_vec[15].push_back(PauliString(2, "IX"));
    ps_vec[15].push_back(PauliString(2, "XX"));
    ps_vec[15].push_back(PauliString(2, "XY"));
    ps_vec[15].push_back(PauliString(2, "XZ"));
    ps_vec[15].push_back(PauliString(2, "IZ"));
    if (add_I)
    {
        ps_vec[15].push_back(PauliString(2, "YI"));
        ps_vec[15].push_back(PauliString(2, "XI"));
        ps_vec[15].push_back(PauliString(2, "ZI"));
    }
    // Set 16
    ps_vec.push_back(PSVec());
    ps_vec[16].push_back(PauliString(2, "ZX"));
    ps_vec[16].push_back(PauliString(2, "YX"));
    ps_vec[16].push_back(PauliString(2, "XZ"));
    ps_vec[16].push_back(PauliString(2, "XY"));
    ps_vec[16].push_back(PauliString(2, "XI"));
    ps_vec[16].push_back(PauliString(2, "IX"));

    // Set 17
    ps_vec.push_back(PSVec());
    ps_vec[17].push_back(PauliString(2, "ZX"));
    ps_vec[17].push_back(PauliString(2, "XY"));
    ps_vec[17].push_back(PauliString(2, "XX"));
    ps_vec[17].push_back(PauliString(2, "YI"));
    ps_vec[17].push_back(PauliString(2, "ZY"));
    ps_vec[17].push_back(PauliString(2, "IZ"));
    if (add_I)
    {
        ps_vec[17].push_back(PauliString(2, "IY"));
        ps_vec[17].push_back(PauliString(2, "ZI"));
    }
    // Set 18
    ps_vec.push_back(PSVec());
    ps_vec[18].push_back(PauliString(2, "ZY"));
    ps_vec[18].push_back(PauliString(2, "YY"));
    ps_vec[18].push_back(PauliString(2, "XZ"));
    ps_vec[18].push_back(PauliString(2, "XX"));
    ps_vec[18].push_back(PauliString(2, "XI"));
    ps_vec[18].push_back(PauliString(2, "IY"));
    if (add_I)
    {
        ps_vec[18].push_back(PauliString(2, "IX"));
        ps_vec[18].push_back(PauliString(2, "YI"));
    }
    // Set 19
    ps_vec.push_back(PSVec());
    ps_vec[19].push_back(PauliString(2, "ZX"));
    ps_vec[19].push_back(PauliString(2, "YZ"));
    ps_vec[19].push_back(PauliString(2, "XY"));
    ps_vec[19].push_back(PauliString(2, "XX"));
    ps_vec[19].push_back(PauliString(2, "YI"));
    ps_vec[19].push_back(PauliString(2, "ZY"));
    ps_vec[19].push_back(PauliString(2, "IZ"));
    if (add_I)
    {
        ps_vec[18].push_back(PauliString(2, "IY"));
        ps_vec[18].push_back(PauliString(2, "ZI"));
    }
    // Set 20
    ps_vec.push_back(PSVec());
    ps_vec[20].push_back(PauliString(2, "ZZ"));
    ps_vec[20].push_back(PauliString(2, "YX"));
    ps_vec[20].push_back(PauliString(2, "XY"));
    ps_vec[20].push_back(PauliString(2, "XX"));
    ps_vec[20].push_back(PauliString(2, "YY"));
    ps_vec[20].push_back(PauliString(2, "ZI"));
    ps_vec[20].push_back(PauliString(2, "IZ"));

    // Set 21
    ps_vec.push_back(PSVec());
    ps_vec[21].push_back(PauliString(2, "YX"));
    ps_vec[21].push_back(PauliString(2, "XY"));
    ps_vec[21].push_back(PauliString(2, "YI"));
    ps_vec[21].push_back(PauliString(2, "YY"));
    ps_vec[21].push_back(PauliString(2, "ZY"));
    ps_vec[21].push_back(PauliString(2, "XX"));
    ps_vec[21].push_back(PauliString(2, "IZ"));
    ps_vec[21].push_back(PauliString(2, "XI"));
    ps_vec[21].push_back(PauliString(2, "ZI"));
    ps_vec[21].push_back(PauliString(2, "ZX"));
    if (add_I)
    {
        ps_vec[21].push_back(PauliString(2, "IY"));
        ps_vec[21].push_back(PauliString(2, "IX"));
    }
    // Set 22
    ps_vec.push_back(PSVec());
    ps_vec[22].push_back(PauliString(2, "YX"));
    ps_vec[22].push_back(PauliString(2, "IY"));
    ps_vec[22].push_back(PauliString(2, "XY"));
    ps_vec[22].push_back(PauliString(2, "YI"));
    ps_vec[22].push_back(PauliString(2, "YY"));
    ps_vec[22].push_back(PauliString(2, "ZY"));
    ps_vec[22].push_back(PauliString(2, "XX"));
    ps_vec[22].push_back(PauliString(2, "IZ"));
    ps_vec[22].push_back(PauliString(2, "IX"));
    ps_vec[22].push_back(PauliString(2, "XI"));
    ps_vec[22].push_back(PauliString(2, "ZI"));
    ps_vec[22].push_back(PauliString(2, "XZ"));
    ps_vec[22].push_back(PauliString(2, "ZX"));
    ps_vec[22].push_back(PauliString(2, "YZ"));
    ps_vec[22].push_back(PauliString(2, "ZZ"));

    // for (std::vector<PSVec>::iterator it=ps_vec.begin();it!=ps_vec.end(); ++it){
    //     std::cout<<"size: "<<(*it).size()<< " ";
    //     print_pauli_vector(*it);
    // }
    return ps_vec;
}

void get_dynamical_lie_algebra(int N, bool add_I, bool closed)
{
    int sun_dim = pow(2, N);
    std::vector<PSVec> subalgebras_su4 = get_unique_algebras_su4(add_I);

    if (!(N > 1))
    {
        throw std::invalid_argument("N must be larger than 2");
    }
    std::cout << "Getting dynamical Lie algebras of SU(" << sun_dim << ")" << std::endl;
    std::cout << "Added I = " << add_I << std::endl;

    std::vector<PSVec> new_subalgebras;
    // SU(N) dimension
    int dim = pow(4, N);
    // Count how many algebras of a certain dimension we encounter
    int counts[dim];
    for (unsigned i = 0; i < dim; i++)
    {
        counts[i] = 0;
    }
    // For SU(4) use the hardcoded algebras
    if (N == 2)
    {
        new_subalgebras = subalgebras_su4;
        for (std::vector<PSVec>::iterator itr = new_subalgebras.begin(); itr != new_subalgebras.end(); ++itr)
        {
            // Count the dimension of the final algebra
            counts[(*itr).size()] += 1;
        }
    }
    // For SU(N>4) perform nested commutators
    else
    {
        std::vector<PSVec> subalgebras_shifted_N;
        for (std::vector<PSVec>::iterator itr = subalgebras_su4.begin(); itr != subalgebras_su4.end(); ++itr)
        {
            PSVec shifted_vec;
            // Add N-2 identities left and right.
            for (PSVec::iterator vec = (*itr).begin(); vec != (*itr).end(); ++vec)
            {
                for (unsigned i = 0; i < (N - 1); i++)
                {
                    shifted_vec.push_back(PauliString(N, std::string(i, 'I') + (*vec).to_str() + std::string(N - 2 - i, 'I')));
                }
                if (closed)
                {
                    shifted_vec.push_back(PauliString(N, (*vec).to_str()[1] + std::string(N - 2, 'I') + (*vec).to_str()[0]));
                }
            }
            // print_pauli_vector(shifted_vec);
            subalgebras_shifted_N.push_back(shifted_vec);
        }
        // Loop over the shifted algebras
        for (std::vector<PSVec>::iterator itr = subalgebras_shifted_N.begin(); itr != subalgebras_shifted_N.end(); ++itr)
        {
            // Create an unordered set for the nested commutator recursion
            PSSet temp_pset = PSSet((*itr).begin(), (*itr).end());
            // std::cout<<"dim before = "<<temp_pset.size()<<std::endl;
            // For each A in the Lie algebra g, calculate [A,g] and add the results to the set
            for (PSVec::iterator j = (*itr).begin(); j != (*itr).end(); ++j)
            {
                nested_commutator(*j, temp_pset);
            }
            // std::cout<<"dim after = "<<temp_pset.size()<<std::endl;
            // Count the dimension of the final algebra
            counts[temp_pset.size()] += 1;
            // Add the new algebra to a vector
            new_subalgebras.push_back(PSVec(temp_pset.begin(), temp_pset.end()));
        }
        // Print the dimensions of the subsets
        std::cout << "Dimensions of the subsets: " << std::endl;
        std::cout << "";
        int total = 0;
        for (unsigned i = 0; i < dim; i++)
        {
            if (counts[i] > 0)
            {
                std::cout << "dim(" << i << ") = " << counts[i] << "\n";
            }
            total += counts[i];
        }
    }
    // Write meta data to file that counts all the seen dimensions
    std::ofstream myfile;
    std::string filename = get_pauliset_filename(N, add_I, closed);
    myfile.open(filename + "meta.txt");

    myfile << "dim"
           << ","
           << "count" << '\n';
    for (unsigned i = 0; i < dim; i++)
    {
        if (counts[i] > 0)
        {
            myfile << i << "," << counts[i] << "\n";
        }
    }
    myfile.close();

    // obtain frustration graphs
    int k = 0;

    for (std::vector<PSVec>::iterator kt = new_subalgebras.begin(); kt != new_subalgebras.end(); ++kt)
    {
        // print_pauli_vector(*kt);
        FrustrationGraph fg(*kt);
        filename = get_pauliset_filename(N, add_I, closed) + "pauliset_" + std::to_string(k) + ".txt";
        std::cout << filename << std::endl;
        // Only print frustration graph to file for small number of qubits
        if ((N == 3) | (N == 4))
        {
            fg.write_to_file(filename, true);
        }
        else
        {
            fg.write_to_file(filename, false);
        }
        k += 1;
    }
}

void get_dynamical_lie_algebra_A_k(int k, int max_N, bool add_I, bool closed)
{
    if ((k < 0) | (k > 22))
    {
        throw std::invalid_argument("k must be between 0 and 22");
    }
    std::vector<PSVec> subalgebras_su4 = get_unique_algebras_su4(add_I);
    PSVec A_k(subalgebras_su4[k]);
    PSVec shifted_subalgebra;
    int sun_dim;
    int dim;

    std::cout << "Classification for A_" << k << std::endl;
    std::cout << "Added I = " << add_I << std::endl;
    for (int N = 3; N <= max_N; N++)
    {
        sun_dim = pow(2, N);
        dim = pow(4, N);
        std::cout << "Getting dynamical Lie algebras of SU(" << sun_dim << ")" << std::endl;

        // SU(N) dimension
        // Count how many algebras of a certain dimension we encounter

        // Add N-2 identities left and right.
        for (PSVec::iterator vec = (A_k).begin(); vec != (A_k).end(); ++vec)
        {
            // std::cout<<(*vec).to_str()<<std::endl;
            for (unsigned i = 0; i < 2; i++)
            {
                // std::cout<<std::string(i, 'I') + (*vec).to_str() + std::string(N - 2 - i, 'I')<<std::endl;
                shifted_subalgebra.push_back(PauliString(N, std::string(i, 'I') + (*vec).to_str() + std::string(1 - i, 'I')));
            }
            if (closed)
            {
                shifted_subalgebra.push_back(PauliString(N, (*vec).to_str()[1] + std::string(N - 2, 'I') + (*vec).to_str()[0]));
            }
        }

        // Create an unordered set for the nested commutator recursion
        PSSet temp_pset = PSSet((shifted_subalgebra).begin(), (shifted_subalgebra).end());
        // std::cout<<"dim before = "<<temp_pset.size()<<std::endl;
        // For each A in the Lie algebra g, calculate [A,g] and add the results to the set
        for (PSVec::iterator j = (shifted_subalgebra).begin(); j != (shifted_subalgebra).end(); ++j)
        {
            nested_commutator(*j, temp_pset);
        }
        std::cout << "N = " << N << " - dim = " << temp_pset.size() << std::endl;

        A_k.clear();
        shifted_subalgebra.clear();
        // Add the new algebra to a vector
        A_k = PSVec(temp_pset.begin(), temp_pset.end());
        // print_pauli_vector(A_k);
    }
}
template <typename T>
int get_index(std::vector<T> v, T K)
{
    auto it = find(v.begin(), v.end(), K);

    // If element was found
    if (it != v.end())
    {

        // calculating the index
        // of K
        return it - v.begin();
    }
    else
    {
        // If the element is not
        // present in the vector
        return -1;
    }
}

void get_commutators_A_k(int k, int N, bool add_I, bool closed)
{
    if ((k < 0) | (k > 22))
    {
        throw std::invalid_argument("k must be between 0 and 22");
    }
    PSVec su4 = get_sun_basis(N - 1, true);
    PSVec A_k = load_pauliset(N, k, add_I, closed);
    print_pauli_vector(A_k);
    PSVec shifted_su4;

    int sun_dim;
    int dim;

    std::cout << "Classification for A_" << k << std::endl;
    std::cout << "Added I = " << add_I << std::endl;
    std::cout << "Closed = " << closed << std::endl;

    sun_dim = pow(2, N);
    dim = pow(4, N);
    std::cout << "Getting dynamical Lie algebras of SU(" << sun_dim << ")" << std::endl;

    // SU(N) dimension
    // Count how many algebras of a certain dimension we encounter
    for (PSVec::iterator vec = (su4).begin(); vec != (su4).end(); ++vec)
    {
        shifted_su4.push_back(PauliString(N, std::string(1, 'I') + (*vec).to_str()));
    }
    PSVec commutators(shifted_su4.begin(), shifted_su4.end());
    PSVec commutators_temp(commutators);
    PauliString result(N);
    std::vector<int> counts(commutators.size(), 1);
    int index;
    for (unsigned depth = 0; depth < 10; depth++)
    {
        for (PSVec::iterator Pk = (A_k).begin(); Pk != (A_k).end(); ++Pk)
        {
            commutators = commutators_temp;
            for (PSVec::iterator Q = (commutators).begin(); Q != (commutators).end(); ++Q)

            {
                if (!(comm(*Pk, *Q))) // If it does not commute, add the result to the set
                {
                    result = (*Pk) * (*Q);
                    index = get_index(commutators_temp, result);
                    if (index == -1)
                    {
                        commutators_temp.push_back(result);
                        counts.push_back(1);
                    }
                    else
                    {
                        counts[index] += 1;
                    }
                }
            }
        }
    }
    std::cout << "number: " << commutators_temp.size() << std::endl;
    std::sort(commutators_temp.begin(), commutators_temp.end(), [](const PauliString &lhs, const PauliString &rhs)
              { return lhs < rhs; });
    std::ofstream myfile;
    std::string filename = get_pauliset_filename(N, add_I, closed);
    std::cout << filename;
    myfile.open(filename + "A_" + std::to_string(k) + "(" + std::to_string(N) + ")_commutators" + ".txt");
    myfile << "number:" << '\n';
    for (unsigned i = 0; i < commutators_temp.size(); i++)
    {
        std::cout << commutators_temp[i].to_str() << std::endl;
        myfile << commutators_temp[i].to_str() << std::endl;
    }
    print_pauli_vector(commutators_temp);
}

void get_associative_algebra(int N, bool add_I, bool closed)
{
    int sun_dim = pow(2, N);
    std::vector<PSVec> subalgebras_su4 = get_unique_algebras_su4(add_I);

    if (!(N > 1))
    {
        throw std::invalid_argument("N must be larger than 2");
    }
    std::cout << "Getting dynamical Lie algebras of SU(" << sun_dim << ")" << std::endl;
    std::cout << "Added I = " << add_I << std::endl;
    std::cout << "Closed = " << closed << std::endl;

    std::vector<PSVec> new_subalgebras;
    // SU(N) dimension
    int dim = pow(4, N);
    // Count how many algebras of a certain dimension we encounter
    int counts[dim + 1];
    for (unsigned i = 0; i < dim + 1; i++)
    {
        counts[i] = 0;
    }
    // For SU(4) use the hardcoded algebras

    std::vector<PSVec> subalgebras_shifted_N;
    for (std::vector<PSVec>::iterator itr = subalgebras_su4.begin(); itr != subalgebras_su4.end(); ++itr)
    {
        PSVec shifted_vec;
        // Add N-2 identities left and right.
        for (PSVec::iterator vec = (*itr).begin(); vec != (*itr).end(); ++vec)
        {
            for (unsigned i = 0; i < (N - 1); i++)
            {
                shifted_vec.push_back(PauliString(N, std::string(i, 'I') + (*vec).to_str() + std::string(N - 2 - i, 'I')));
            }
            if (closed)
            {
                shifted_vec.push_back(PauliString(N, (*vec).to_str()[1] + std::string(N - 2, 'I') + (*vec).to_str()[0]));
            }
        }
        // print_pauli_vector(shifted_vec);
        subalgebras_shifted_N.push_back(shifted_vec);
    }
    // Loop over the shifted algebras
    for (std::vector<PSVec>::iterator itr = subalgebras_shifted_N.begin(); itr != subalgebras_shifted_N.end(); ++itr)
    {
        // Create an unordered set for the nested commutator recursion
        PSSet temp_pset = PSSet((*itr).begin(), (*itr).end());
        // std::cout<<"dim before = "<<temp_pset.size()<<std::endl;
        // For each A in the Lie algebra g, calculate [A,g] and add the results to the set
        for (PSVec::iterator j = (*itr).begin(); j != (*itr).end(); ++j)
        {
            nested_product(*j, temp_pset);
        }
        // std::cout<<"dim after = "<<temp_pset.size()<<std::endl;
        // Count the dimension of the final algebra
        counts[temp_pset.size()] += 1;
        // print_pauli_unordered_set(temp_pset);
        // Add the new algebra to a vector
        new_subalgebras.push_back(PSVec(temp_pset.begin(), temp_pset.end()));
    }
    // Print the dimensions of the subsets
    std::cout << "Dimensions of the subsets: " << std::endl;
    std::cout << "";
    int total = 0;
    for (unsigned i = 0; i < dim + 1; i++)
    {
        if (counts[i] > 0)
        {
            std::cout << "dim(" << i << ") = " << counts[i] << "\n";
        }
        total += counts[i];
    }

    // Write meta data to file that counts all the seen dimensions
    std::ofstream myfile;
    std::string filename = get_pauliset_filename(N, add_I, closed);
    myfile.open(filename + "meta_associative.txt");
    myfile << "dim"
           << ","
           << "count" << '\n';
    for (unsigned i = 0; i < dim + 1; i++)
    {
        if (counts[i] > 0)
        {
            myfile << i << "," << counts[i] << "\n";
        }
    }

    for (unsigned k = 0; k < 23; k++)
    {
        myfile.open(filename + "associative_" + std::to_string(k) + ".txt");
        myfile << "dim = " << new_subalgebras[k].size() << "\n";
        int l = 0;
        std::sort(new_subalgebras[k].begin(), new_subalgebras[k].end(), [](const PauliString &lhs, const PauliString &rhs)
                  { return lhs < rhs; });
        for (PSVec::iterator it = new_subalgebras[k].begin(); it != new_subalgebras[k].end(); ++it)
        {
            if (l < (new_subalgebras[k].size() - 1))
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
}

#endif