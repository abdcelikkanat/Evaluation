import constants as const
import sys, os
sys.path.append(os.path.join(const._BASE_FOLDER, "edge_prediction", "TNE"))
from tne.tne import TNE
from community_detection.by_random_walks import *


import time

dataset_folder = "../datasets/"
dataset_file = "karate.gml"

outputs_folder = "../outputs/"
temp_folder = "../temp/"

# Set all parameters #
params = {}
params['method'] = "deepwalk"
# Common parameters
params['number_of_walks'] = 80
params['walk_length'] = 3
params['window_size'] = 10
params['embedding_size'] = 128
params['number_of_topics'] = 2
# Parameters for LDA
params['lda_number_of_iters'] = 1000
params['lda_alpha'] = 50.0 / float(params['number_of_topics'])
params['lda_beta'] = 0.1
# Parameters for Deepwalk
params['dw_alpha'] = 0
# Parameters for Node2vec
params['n2v_p'] = 1.0
params['n2v_q'] = 1.0

params['hs'] = 0
params['negative'] = 5

# Define the file paths
nx_graph_path = dataset_folder + dataset_file



corpus_path_for_node = os.path.join(const._BASE_FOLDER, "tests", "temp", "test.walks")



####
tne = TNE(nx_graph_path, params)
tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
#tne.save_corpus(corpus_path_for_lda, with_title=True)




brw = ByRandomWalks()
brw.set_graph(nxg=nx.read_gml(nx_graph_path))
brw.read_walks_file(corpus_path_for_node)
node2comm = brw.get_community_assignments_by(method="HMM", params=params)

print(node2comm)

#brw.plot_graph(node2comm=node2comm)