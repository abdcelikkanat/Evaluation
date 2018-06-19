import sys
sys.path.append("../../deepwalk/deepwalk")
import graph as dw
import networkx as nx
from gensim.models.word2vec import Word2Vec
from graphbase import randomgraph
from sklearn.cluster import KMeans
from community_detection.community_detection import *

N = 1000
kmeans_num_of_communities = 3

if kmeans_num_of_communities == 2:
    sbm_params = {}
    sbm_params['sbm_N'] = N  # the number of nodes
    sbm_params['sbm_P'] = [[0.7, 0.5], [0.4, 0.6]]  # edge probability matrix between nodes belonging different communities
    sbm_params['sbm_block_sizes'] = [300, 700]
elif kmeans_num_of_communities == 3:
    sbm_params = {}
    sbm_params['sbm_N'] = N  # the number of nodes
    sbm_params['sbm_P'] = [[0.7, 0.3, 0.4],
                           [0.3, 0.6, 0.2],
                           [0.4, 0.2, 0.9]]  # edge probability matrix between nodes belonging different communities
    sbm_params['sbm_block_sizes'] = [300, 500, 200]

dw_params = {}
dw_params['n'] = 80
dw_params['l'] = 10
dw_params['w'] = 10
dw_params['d'] = 128
dw_params['workers'] = 3

rg = randomgraph.RanGraphGen()
rg.set_model(model=sbm_params)
g = rg.stochastic_block_model()

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
kmeans = KMeans(n_clusters=kmeans_num_of_communities, random_state=0)
kmeans.fit(x)
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

"""
cluster_labels = {'0': [4], '1': [3], '2': [2], '3': [1], '4': [0], '5': [1]}
nx.set_node_attributes(testg, values=cluster_labels, name='clusters')

comdetect = CommunityDetection()
comdetect.set_graph(nxg=testg)
number_of_clusters = comdetect.detect_number_of_clusters()
comdetect.set_number_of_clusters(number_of_clusters)
clusters_pred = {'0': [0], '1': [1], '2': [2], '3': [3], '4': [4], '5': [2]}
comdetect.avg_f1_score(clusters_pred=clusters_pred)
comdetect.nmi_score(clusters_pred)
"""