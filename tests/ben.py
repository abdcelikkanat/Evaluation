import networkx as nx
import numpy as np

g = nx.read_gml("../datasets/citeseer_undirected.gml")

n = g.number_of_nodes()

node2comm = nx.get_node_attributes(g, 'community')

score = 0.0
for node in node2comm:
    nb_com_list = [0 for _ in range(100)]
    for nb in nx.neighbors(g, node):
        nb_com_list[node2comm[nb]] += 1
        for nb_nb in nx.neighbors(g, nb):
            if nb_nb != nb and nb_nb != node:
                nb_com_list[node2comm[nb_nb]] +=  #ILGINC

    if node2comm[node] == np.argmax(nb_com_list):
        score += 1.0


print(score/float(n)*100)