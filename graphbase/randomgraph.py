import networkx as nx

import numpy as np


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K) # probs * K where probs may be updated during iterations
    J = np.zeros(K, dtype=np.int) # stores the class indicies having prob. of larger than 1/K

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        print(K)
        print(prob)
        q[kk] = K * prob
        if q[kk] < 1.0: # if prob < 1.0 / float(K):
            smaller.append(kk)
        else:
            larger.append(kk)

    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small]) # remaining amount, or remaining prob prob_large - (1.0 - prob_small)

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand() * K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


K = 5
N = 1000

# Get a random probability vector.
probs = np.random.dirichlet(np.ones(K), 1).ravel()

# Construct the table.
J, q = alias_setup(probs)

# Generate variates.
X = np.zeros(N)
for nn in xrange(N):
    X[nn] = alias_draw(J, q)


class RanGraphGen:
    def __init__(self, model=None):
        self._model = model

    def set_model(self, model):
        self._model = model

    def stochastic_block_model(self):

        N = self._model['N'] # the number of nodes
        K = self._model['K'] # the number of clusters
        P = self._model['P'] # edge probability matrix between nodes belonging different communities
        block_sizes = self._model['block_sizes']

        graph = nx.Graph()

        #J, q = alias_setup(P)
        clusters = [[] for _ in range(K)]

        for node in range(N):

            """
            k = alias_draw(J, q)
            graph.add_node(str(node), clusters=[k])
            """
            k = int(np.rint(float(node) / np.sum(block_sizes)))
            clusters[k].append(str(node))

        for k1 in range(K):
            for k2 in range(k1, K):
                # Generate edges for each cluster
                if k1 == k2:
                    subg = nx.fast_gnp_random_graph(n=len(clusters[k1]), p=P[k1][k2])
                # Generate edges for nodes belonging to distinct clusters
                else:
                    subg = nx.algorithms.bipartite.random_graph(n=len(clusters[k1]), m=len(clusters[k2]), p=P[k1][k2])

                for edge in subg.edges():
                    n1 = clusters[k1][subg[int(edge[0])]]
                    n2 = clusters[k2][subg[int(edge[1])-len(clusters[k1])]]
                    graph.add_edge(n1, n2)

        return graph

model = {}
model['N'] = 10
model['K'] = 2
model['P'] = [[0.3, 0.4], [0.2, 0.7]]
model['block_sizes'] = [5, 5]
rg = RanGraphGen(model)
graph = rg.stochastic_block_model()
print(graph.number_of_nodes())

"""
def SBM_simulate_fast(model):
    G = nx.Graph()
    b = model['a']
    J,q = alias_setup(b)
    n = model['N']
    k = model['K']
    B = model['B0']*model['alpha']
    totaledges =0
# add nodes with communities attributes
    grps = {}
    for t in range(k):
        grps[t] = []
    for key in range(n):
        comm = alias_draw(J,q)
        G.add_node(key, community = comm)
        grps[comm].append(key)
    for i in range(k):
        grp1 = grps[i]
        L1 = len(grp1)
        for j in range(i,k):
            grp2 = grps[j]
            L2 = len(grp2)
            if i==j:
                Gsub = nx.fast_gnp_random_graph(L1,B[i,i])
            else:
                Gsub = nx.algorithms.bipartite.random_graph(L1,L2,B[i,j])
            for z in Gsub.edges():
                nd1 = grp1[z[0]]
                nd2 = grp2[z[1]-L1]
                G.add_edge(nd1, nd2, weight = 1.0)
                totaledges +=1
#                if totaledges%1000 == 1:
#                    print 'constructed ', totaledges, ' number of edges'
    print 'the size of graph is ', totaledges, 'number of edges'
return G
"""