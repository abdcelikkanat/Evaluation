import numpy as npy
import networkx as nx

g = nx.read_gml("./dblp_undirected.gml")
print(g.number_of_nodes())
print(g.number_of_edges())
