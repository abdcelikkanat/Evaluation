import networkx as nx
import igraph as ig
import numpy as np
from networkx.algorithms.community import LFR_benchmark_graph


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K) # probs * K where probs may be updated during iterations
    J = np.zeros(K, dtype=np.int) # stores the class indicies having prob. of larger than 1/K

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
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

        n = self._model['sbm_N']  # the number of nodes
        p = self._model['sbm_P']  # edge probability matrix between nodes belonging different communities
        block_sizes = self._model['sbm_block_sizes']
        K = len(p)  # number of communities

        assert n == np.sum(block_sizes), "The sum of the block sizes must be equal to the number of nodes, N"

        """
        graph = nx.Graph()
        igraph = ig.Graph.SBM(n=n, pref_matrix=p, block_sizes=block_sizes, directed=False, loops=False)
        graph.add_edges_from([edge.tuple for edge in igraph.es])
        """

        graph = nx.Graph()
        node2community = {}

        communities = [[] for _ in range(K)]

        for node in range(n):
            k = int(np.rint(float(node) / np.sum(block_sizes)))
            communities[k].append(str(node))
            graph.add_node(str(node))
            node2community.update({str(node): [k]})

        for k1 in range(K):
            for k2 in range(k1, K):
                # Generate edges for each community
                if k1 == k2:
                    subg = nx.fast_gnp_random_graph(n=len(communities[k1]), p=p[k1][k2])

                else:
                    subg = nx.algorithms.bipartite.random_graph(n=len(communities[k1]), m=len(communities[k2]), p=p[k1][k2])

                for edge in subg.edges():
                    n1 = communities[k1][int(edge[0])]
                    n2 = communities[k2][int(edge[1])-len(communities[k1])]
                    graph.add_edge(n1, n2)

        nx.set_node_attributes(graph, values=node2community, name='community')

        return graph

    def lfr_model(self):

        n = self._model['lfr_N']
        tau1 = self._model['lfr_tau1']  # power law exponent for node degree distribution
        tau2 = self._model['lfr_tau2']  # power law exponent for community size distribution
        mu = self._model['lfr_mu']  # fraction of edges between communities
        max_deg = self._model['lfr_max_deg'] if 'lfr_max_deg' in self._model else n
        min_comm = self._model['lfr_min_community']
        max_comm = self._model['lfr_max_community'] if 'lfr_max_community' in self._model else n

        if 'lfr_average_degree' in self._model:
            avg_deg = self._model['lfr_average_degree']
            graph = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu,
                                        average_degree=avg_deg, max_degree=max_deg,
                                        min_community=min_comm, max_community=max_comm)
            return graph
        elif 'lfr_min_degree' in self._model:
            min_deg = self._model['lfr_min_degree']
            graph = LFR_benchmark_graph(n=n, tau1=tau1, tau2=tau2, mu=mu,
                                        min_degree=min_deg, max_degree=max_deg,
                                        min_community=min_comm, max_community=max_comm)

            return graph

"""

model = {}
# Parameters used for Stochastic Block Model
model['sbm_N'] = 10
model['sbm_P'] = [[0.3, 0.4], [0.4, 0.7]]
model['sbm_block_sizes'] = [5, 5]
# Parameters used for Lancichinetti-Fortunato-Radicchi (LFR)
model['lfr_N'] = 0
model['lfr_tau1'] = 0
model['lfr_tau2'] = 0
model['lfr_mu'] = 0
model['lfr_min_degree'] = 0
model['lfr_max_degree'] = 0
model['lfr_average_degree'] = 0
model['lfr_min_comm'] = 0
model['lfr_max_comm'] = 0


rg = RanGraphGen(model)
graph = rg.stochastic_block_model()
print(graph.number_of_nodes())


"""

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