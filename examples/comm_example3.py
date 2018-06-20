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


g = nx.read_gml("./outputs/synthetic_n1000_c3.gml")
embedding_file = "./outputs/synthetic_n1000_c3_n80_l10_w10_k50_deepwalk_final_max.embedding"
#embedding_file = "./outputs/synthetic_n1000_c3_n80_l10_w10_k50_deepwalk_node.embedding"
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


