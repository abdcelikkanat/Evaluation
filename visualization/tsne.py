import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.pylab as plt


def get_embeddings(embedding_file):
    id2node = []
    x = []
    with open(embedding_file, 'r') as f:
        f.readline()
        for line in f.readlines():
            tokens = line.strip().split()
            id2node.append(tokens[0])
            x.append([float(value) for value in tokens[1:]])

    return id2node, x


g = nx.read_gml("../examples/outputs/synthetic_n1000_c3.gml")
embedding_file = "../examples/outputs/synthetic_n1000_c3_n80_l10_w10_k50_deepwalk_final_max.embedding"
id2node, x = get_embeddings(embedding_file)
communities_true = nx.get_node_attributes(g, name='community')


z = TSNE(n_components=2, verbose=0).fit_transform(x)

plt.figure()
for node in g.nodes():
    if communities_true[node] == 0:
        plt.plot(z[int(node), 0], z[int(node), 1], 'r.')
    if communities_true[node] == 1:
        plt.plot(z[int(node), 0], z[int(node), 1], 'b.')
    if communities_true[node] == 2:
        plt.plot(z[int(node), 0], z[int(node), 1], 'g.')
plt.show()