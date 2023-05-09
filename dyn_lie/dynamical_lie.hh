#ifndef DYNLIE_HH
#define DYNLIE_HH

#include <iostream>
#include <cmath>
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

void get_dynamical_lie_algebra(int N, bool add_I)
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
    if (add_I)
    {
        myfile.open("./data/su" + std::to_string(sun_dim) + "_I/meta.txt");
    }
    else
    {
        myfile.open("./data/su" + std::to_string(sun_dim) + "/meta.txt");
    }
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
    // // obtain lexicographic ordering
    // std::sort(new_subalgebras.begin(), new_subalgebras.end(), [](const PSVec &lhs, const PSVec &rhs)
    //           { return lhs.size() < rhs.size(); });
    // obtain frustration graphs
    int i = 0;
    for (std::vector<PSVec>::iterator kt = new_subalgebras.begin(); kt != new_subalgebras.end(); ++kt)
    {
        // print_pauli_vector(*kt);
        FrustrationGraph fg(*kt);
        // Only print frustration graph to file for small number of qubits
        if ((N == 3) | (N == 4))
        {
            if (add_I)
            {
                fg.write_to_file("./data/su" + std::to_string(sun_dim) + "_I/pauliset_" + std::to_string(i) + ".txt", true);
            }
            else
            {
                fg.write_to_file("./data/su" + std::to_string(sun_dim) + "/pauliset_" + std::to_string(i) + ".txt", true);
            }
        }
        else
        {
            if (add_I)
            {
                fg.write_to_file("./data/su" + std::to_string(sun_dim) + "_I/pauliset_" + std::to_string(i) + ".txt", false);
            }
            else
            {
                fg.write_to_file("./data/su" + std::to_string(sun_dim) + "/pauliset_" + std::to_string(i) + ".txt", false);
            }
        }
        i += 1;
    }
}

void get_dynamical_lie_algebra_A_k(int k, int max_N, bool add_I)
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

#endif