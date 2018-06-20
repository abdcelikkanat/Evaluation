# generate a lfr graph
# find embedding with word2vec
# apply k-means to find non-overlapping communities
# get nmi scores for predicted and ground-truth labels

import sys
sys.path.append("../../deepwalk/deepwalk")
import graph as dw
import networkx as nx
from gensim.models.word2vec import Word2Vec
from graphbase import randomgraph
from sklearn.cluster import KMeans
from community_detection.community_detection import *

N = 1000
kmeans_num_of_communities = 2

lfr_params = {}
lfr_params['lfr_N'] = N
lfr_params['lfr_tau1'] = 2.1  # power law exponent for node degree distribution
lfr_params['lfr_tau2'] = 2.4  # power law exponent for community size distribution
lfr_params['lfr_mu'] = 0.2  # fraction of edges between communities
lfr_params['lfr_min_degree'] = 3
lfr_params['lfr_min_community'] = 300  # minimum size of communities in the graph

dw_params = {}
dw_params['n'] = 80
dw_params['l'] = 10
dw_params['w'] = 10
dw_params['d'] = 128
dw_params['workers'] = 3

rg = randomgraph.RanGraphGen()
rg.set_model(model=lfr_params)
g = rg.lfr_model()

nx.write_gml(g, "./outputs/lfr_synthetic_n1000.gml")

# Find the embedding of the
temp_adjlist_file = "./temp/graph.adjlist"
embedding_file = "./outputs/output.embedding"
nx.write_edgelist(g, temp_adjlist_file)

dwg = dw.load_edgelist(temp_adjlist_file, undirected=True)
walks = dw.build_deepwalk_corpus(dwg, num_paths=dw_params['n'], path_length=dw_params['l'], alpha=0)
model = Word2Vec(walks, size=dw_params['d'], window=dw_params['w'], min_count=0, sg=1, hs=1, workers=dw_params['workers'])
model.wv.save_word2vec_format(embedding_file)


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


id2node, x = get_embeddings(embedding_file)
kmeans = KMeans(n_clusters=kmeans_num_of_communities, random_state=0).fit(x)
labels = kmeans.labels_.tolist()

communities_true = nx.get_node_attributes(g, name='community')
communities_pred = {id2node[i]: [labels[i]] for i in range(len(x))}

#print(communities_true)
#print(communities_pred)

comdetect = CommunityDetection()
comdetect.set_graph(nxg=g)
number_of_communities = comdetect.detect_number_of_communities()
comdetect.set_number_of_communities(number_of_communities)
score = comdetect.nmi_score(communities_pred)
print("Score: {}".format(score))
