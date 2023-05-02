import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

dir = './su8'
SHOW = False
for file in os.listdir(dir):
    filename = file.split('.')
    extension = filename[-1]
    pauli_name = filename[0]
    if extension == 'txt':
        fig = plt.figure()
        df = pd.read_csv(dir+'/'+file, delimiter='.')
        dimension = df.columns[0]
        print(dimension)
        names = df.iloc[0].values[0].split(',')
        graph = nx.Graph()
        if len(df) == 2:
            edges = [eval(df.iloc[1].values[0])]
            graph.add_edges_from(edges[0])

        else:
            graph.add_nodes_from(list(range(len(names))))

        nx.draw_circular(graph, labels=dict(zip(list(range(len(names))),names)))
        fig.suptitle(dimension)
        fig.savefig(dir + '/' +pauli_name)
        if SHOW:
            plt.show()
        plt.close()