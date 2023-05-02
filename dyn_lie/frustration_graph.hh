#ifndef FGRAPH_HH
#define FGRAPH_HH

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <utility>
#include <iostream>
#include <fstream>
#include "pauli.hh"
#include "pauli_set.hh"

typedef std::pair<PauliString, int> Vertex;

class FrustrationGraph
{
public:
    FrustrationGraph(PSSet pauli_set)
    {
        pv = PSVec(pauli_set.begin(), pauli_set.end());
        init();
    };
    // Convert Pauli vector to Pauli set
    FrustrationGraph(PSVec pauli_vector)
    {
        pv = pauli_vector;
        init();
    };

    ~FrustrationGraph()
    {
        for (unsigned i = 0; i < size; i++)
        {
            delete adjacency_graph[i];
        }
        delete adjacency_graph;
    };

    void print_vertices()
    {
        std::cout << "vertices: ";
        for (std::vector<Vertex>::iterator it = vertices.begin(); it != vertices.end(); ++it)
        {
            it->first.print_str(false);
        }
        std::cout << "\n";
    }
    void print_edges()
    {
        std::cout << "edges: ";
        for (std::set<std::pair<int, int>>::iterator it = edges.begin(); it != edges.end(); ++it)
        {
            std::cout << "(" << it->first << "," << it->second << "), ";
        }
        std::cout << "\n";
    }

    void write_to_file(std::string filename, bool print_frustration_graph = false)
    {
        std::ofstream myfile;
        myfile.open(filename);
        myfile << "dim = " << pv.size() << "\n";
        int l = 0;
        for (std::vector<Vertex>::iterator it = vertices.begin(); it != vertices.end(); ++it)
        {
            if (l < (vertices.size() - 1))
            {
                myfile << it->first.to_str() << ",";
            }
            else
            {
                myfile << it->first.to_str();
            }
            l += 1;
        }
        myfile << "\n";
        if (print_frustration_graph)
        {
            l = 0;
            for (std::set<std::pair<int, int>>::iterator it = edges.begin(); it != edges.end(); ++it)
            {
                if (l < (edges.size() - 1))
                {
                    myfile << "(" << it->first << "," << it->second << "),";
                }
                else
                {
                    myfile << "(" << it->first << "," << it->second << ")";
                }
                l += 1;
            }
            myfile << "\n";
        }

        myfile.close();
    }
    std::vector<Vertex> vertices;
    std::set<std::pair<int, int>> edges;
    PSVec pv;

private:
    void init()
    {
        size = pv.size();
        adjacency_graph = new int *[size];
        for (unsigned i = 0; i < size; i++)
        {
            adjacency_graph[i] = new int[size];
            for (unsigned j = 0; j < size; j++)
            {
                adjacency_graph[i][j] = 0;
            }
        }
        build_graph();
    }

    void build_graph()
    {
        // Build the vertices
        int i = 0;
        for (PSVec::iterator it = pv.begin(); it != pv.end(); ++it)
        {
            vertices.push_back(Vertex(*it, i));
            i += 1;
        }

        for (unsigned i = 0; i < size; i++)
        {
            // (&vertices[i])->first.print_str(false);
            for (unsigned j = 0; j <= i; j++)
            {
                // std::cout<<i<<","<<j<<std::endl;
                if (!(comm((&vertices[i])->first, (&vertices[j])->first)))
                {
                    std::pair<int, int> p((&vertices[i])->second, (&vertices[j])->second);
                    edges.insert(p);
                    adjacency_graph[i][j] = 1;
                    adjacency_graph[j][i] = 1;
                }
                // else{
                //     std::cout<< (&vertices[i])->first.to_str()<<" commutes with "<<(&vertices[j])->first.to_str()<<std::endl;
                // }
            }
        }
    }

    int size;
    int **adjacency_graph;
};

#endif