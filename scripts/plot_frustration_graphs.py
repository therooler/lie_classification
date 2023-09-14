import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

dir = './data/open/su16/'
su4_dir = './data/open/su4/'
fig_dir = './figures/open/su16'


def filter_names_and_edges(strings, edges, substring):
    filtered_strings = []
    filtered_edges = []
    for i, s in enumerate(strings):
        if substring in s:
            left_over_string = s.replace(substring, '')
            if all(left_over_s == 'I' for left_over_s in left_over_string):
                print(f"{s} is correct")
                filtered_strings.append(s)
                filtered_edges.append(edges[i])
            else:
                print(f"{s} is incorrect")
        else:
            print(f"{s} is incorrect")
    return filtered_strings, filtered_edges


SHOW = True
FILTER = True

for file in os.listdir(dir):
    filename = file.split('.')
    extension = filename[-1]
    pauli_name = filename[0]
    pauli_name_split = pauli_name.split('_')
    if pauli_name_split[0] == "pauliset":
        k = int(pauli_name.split('_')[-1])
        print(k)
        if k == 12:
            if extension == 'txt':
                fig = plt.figure()
                df = pd.read_csv(dir + '/' + file, delimiter='.')
                df_su4 = pd.read_csv(su4_dir + '/' + file, delimiter='.')
                dimension = df.columns[0]
                print(dimension)
                names = df.iloc[0].values[0].split(',')
                names_su4 = df_su4.iloc[0].values[0].split(',')
                edges = [eval(df.iloc[1].values[0])][0]
                graph = nx.Graph()
                lookup = dict(zip(list(range(len(names))), names))
                lookup_rev = dict(zip(names, list(range(len(names)))))
                if len(df) == 2:

                    graph.add_edges_from(edges)

                else:
                    graph.add_nodes_from(list(range(len(names))))

                # if FILTER:
                #     filtered_names = []
                #     filtered_edges = []
                #     for name in names_su4:
                #         f_n, f_e = filter_names_and_edges(names, edges, name)
                #         filtered_names.extend(f_n)
                #         filtered_edges.extend(f_e)
                #     print(filtered_names)
                #     print(filtered_edges)
                #     print(graph.nodes)
                #     graph = nx.subgraph(G=graph, nbunch=[lookup_rev[x] for x in filtered_names])
                #     lookup = dict(zip(graph.nodes,filtered_names))
                #     print(lookup)
                #     print(graph)
                #     print(graph.nodes)
                #     print(graph.edges)

                nx.draw_circular(graph, labels=lookup)
                fig.suptitle(dimension)
                fig.savefig(fig_dir + '_frustration_graph_' + pauli_name + '.svg')
                if SHOW:
                    plt.show()
                plt.close()
