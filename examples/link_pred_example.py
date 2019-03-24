import os
import sys
p = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'edge_prediction', 'TNE'))
sys.path.insert(0, p)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
from os.path import splitext, basename
from edge_prediction.edge_prediction import EdgePrediction


train_test_ratio = 0.5
repeat_count = 1

graph_names = ["facebook_combined_gcc"] #ca-AstroPh_gcc  facebook_combined_gcc # ["facebook_combined", "CA-AstroPh", "p2p-Gnutella09"]
# Set all parameters #
params = {}
params['comm_detection_method'] = "lda"
params['random_walk'] = "node2vec"
# Common parameters
params['number_of_walks'] = 8 #80
params['walk_length'] = 10
params['window_size'] = 10
params['embedding_size'] = 128
params['number_of_communities'] = 8 #80
# Parameters for LDA
params['lda_number_of_iters'] = 10 #1000
params['lda_alpha'] = 0.2
params['lda_beta'] = 0.1
# Parameters for Deepwalk
params['dw_alpha'] = 0
# Parameters for Node2vec
params['n2v_p'] = 4.0
params['n2v_q'] = 1.0
# Parameters for SkipGram
params['hs'] = 0
params['negative'] = 5
# Parameters for hmm
params['hmm_p0'] = 0.3
params['hmm_t0'] = 0.2
params['hmm_e0'] = 0.1
params['hmm_number_of_iters'] = 2000
params['hmm_subset_size'] = 100


#for graphname in ["facebook_combined", "CA-AstroPh", "p2p-Gnutella09"]:
for graphname in graph_names:

    ### Initial path definitions ###
    base_folder = os.path.dirname(os.path.abspath(__file__))

    nx_graph_path = os.path.realpath(os.path.join(base_folder, "..", "datasets", graphname + ".gml"))

    file_desc = "{}_{}_n{}_l{}_w{}_k{}_{}".format(splitext(basename(nx_graph_path))[0],
                                                  params['comm_detection_method'],
                                                  params['number_of_walks'],
                                                  params['walk_length'],
                                                  params['window_size'],
                                                  params['number_of_communities'],
                                                  params['random_walk'])

    if params['random_walk'] == 'node2vec':
        file_desc += "_p={}_q={}".format(params['n2v_p'], params['n2v_q'])

    # temp folder
    temp_folder = os.path.join(base_folder, "temp", file_desc)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # output folder
    embedding_folder = os.path.join(base_folder, "embeddings", file_desc)
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    # scores folder
    scores_folder = os.path.join(base_folder, "scores", graphname)
    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)
    ### ------------------- ###

    ep = EdgePrediction()
    ep.read_graph(graph_path=nx_graph_path)

    total_scores_node = []
    total_scores_tne = []
    for iter in range(repeat_count):
        scores_node, scores_final = ep.get_scores(train_test_ratio=train_test_ratio, params=params,
                                                  file_desc=file_desc,
                                                  temp_folder=temp_folder,
                                                  embedding_folder=embedding_folder)

        total_scores_node.append(scores_node)
        total_scores_tne.append(scores_final)

    with open(os.path.join(scores_folder, "{}_{}_node.score".format(graphname, params['random_walk'])), "wb") as fp:
        pickle.dump(total_scores_node, fp)
    with open(os.path.join(scores_folder, "{}_{}_final.score".format(graphname, params['random_walk'])), "wb") as fp:
        pickle.dump(total_scores_tne, fp)
