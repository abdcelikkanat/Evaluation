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
from community_detection.detection import CommunityDetection

N = 1000
kmeans_num_of_communities = 2

lfr_params = {}
lfr_params['lfr_N'] = N
lfr_params['lfr_tau1'] = 2.1  # power law exponent for node degree distribution
lfr_params['lfr_tau2'] = 2.1  # power law exponent for community size distribution
lfr_params['lfr_mu'] = 0.2  # fraction of edges between communities
lfr_params['lfr_min_degree'] = 7
lfr_params['lfr_min_community'] = 500  # minimum size of communities in the graph

dw_params = {}
dw_params['n'] = 80
dw_params['l'] = 10
dw_params['w'] = 10
dw_params['d'] = 128
dw_params['workers'] = 3

rg = randomgraph.RanGraphGen()
rg.set_model(model=lfr_params)
g = rg.lfr_model()

graph_path = "./outputs/lfr_synthetic_n1000.gml"
nx.write_gml(g, graph_path)

# Find the embedding of the
temp_adjlist_file = "./temp/graph.adjlist"
embedding_file = "./outputs/output.embedding"
nx.write_edgelist(g, temp_adjlist_file)

dwg = dw.load_edgelist(temp_adjlist_file, undirected=True)
walks = dw.build_deepwalk_corpus(dwg, num_paths=dw_params['n'], path_length=dw_params['l'], alpha=0)
model = Word2Vec(walks, size=dw_params['d'], window=dw_params['w'], min_count=0, sg=1, hs=1, workers=dw_params['workers'])
model.wv.save_word2vec_format(embedding_file)



comdetect = CommunityDetection(embedding_file, graph_path, params={'directed': False})
score = comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Score: {}".format(score))
