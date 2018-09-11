import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing

from graphbase.graphbase import *
from os.path import basename, splitext, join
sys.path.append("./TNE")
from tne.tne import TNE
from utils.utils import find_topics_for_nodes, concatenate_embeddings


_operators = ["hadamard", "average", "l1", "l2"]


class EdgePrediction(GraphBase):

    def __init__(self):
        GraphBase.__init__(self)

    def split_into_train_test_sets(self, ratio, max_trial_limit=10000):
        # Split the graph into two disjoint sets while keeping the training graph connected

        test_set_size = int(ratio * self.g.number_of_edges())

        # Generate the positive test edges
        test_pos_samples = []
        residual_g = self.g.copy()
        num_of_ccs = nx.number_connected_components(residual_g)
        if num_of_ccs != 1:
            raise ValueError("The graph contains more than one connected component!")

        num_of_pos_samples = 0

        edges = list(residual_g.edges())
        perm = np.arange(len(edges))
        np.random.shuffle(perm)
        edges = [edges[inx] for inx in perm]

        for i in range(len(edges)):
            # Remove the chosen edge
            chosen_edge = edges[i]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            if chosen_edge[1] in nx.connected._plain_bfs(residual_g, chosen_edge[0]):
                num_of_pos_samples += 1
                test_pos_samples.append(chosen_edge)
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_samples == test_set_size:
                break

        if num_of_pos_samples != test_set_size:
            raise ValueError("Enough positive edge samples could not be found!")

        # Generate the negative samples
        non_edges = list(nx.non_edges(self.g))
        perm = np.arange(len(non_edges))
        np.random.shuffle(perm)
        non_edges = [non_edges[inx] for inx in perm]
        chosen_non_edge_inx = np.random.choice(perm, size=test_set_size*2, replace=False)

        train_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[:test_set_size]]
        test_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[test_set_size:]]

        train_pos_samples = list(residual_g.edges())
        return residual_g, train_pos_samples, train_neg_samples, test_pos_samples, test_neg_samples

    def read_embedding_file(self, file_path):

        with open(file_path, 'r') as f:
            f.readline() # Skip the first line
            embeddings = {}
            for line in f:
                tokens = line.strip().split()
                embeddings.update({str(tokens[0]): [float(val) for val in tokens[1:]]})

        return embeddings

    def get_features_from_embeddings(self, edges, embeddings, binary_operator):

        features = []
        for i in range(len(edges)):
            edge = edges[i]
            vec1 = np.asarray(embeddings[edge[0]])
            vec2 = np.asarray(embeddings[edge[1]])

            value = 0
            if binary_operator == "hadamard":
                value = [vec1[i]*vec2[i] for i in range(len(vec1))]
            if binary_operator == "average":
                value = 0.5 * (vec1 + vec2)
            if binary_operator == "l1":
                value = abs(vec1 - vec2)
            if binary_operator == "l2":
                value = abs(vec1 - vec2)**2

            features.append(value)

        return np.asarray(features)

    def learn_tne_embeddings(self, nxg, base_folder, params):

        tne = TNE(params=params)
        tne.set_graph(nxg)

        temp_folder = os.path.join("./temp", base_folder)
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        outputs_folder = os.path.join("./outputs", base_folder)
        if not os.path.exists(outputs_folder):
            os.makedirs(outputs_folder)

        base_desc = "{}_n{}_l{}_w{}_k{}_{}".format(self.get_graph_name(),
                                                   params['number_of_walks'],
                                                   params['walk_length'],
                                                   params['window_size'],
                                                   params['number_of_topics'],
                                                   params['method'])

        # node corpus
        corpus_path_for_node = join(temp_folder, "{}_node_corpus.corpus".format(base_desc))
        # lda corpus
        corpus_path_for_lda = join(temp_folder, "{}_lda_corpus.corpus".format(base_desc))
        # embedding files
        node_embedding_file = join(outputs_folder, "{}_node.embedding".format(base_desc))
        topic_embedding_file = join(outputs_folder, "{}_topic.embedding".format(base_desc))
        concatenated_embedding_file_max = join(outputs_folder, "{}_final_max.embedding".format(base_desc))

        tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
        tne.save_corpus(corpus_path_for_lda, with_title=True)
        id2node = tne.run_lda(lda_corpus_path=corpus_path_for_lda)
        tne.learn_node_embedding(node_corpus_path=corpus_path_for_node, node_embedding_file=node_embedding_file)
        tne.learn_topic_embedding(node_corpus_path=corpus_path_for_node, topic_embedding_file=topic_embedding_file)

        # Concatenate the embeddings
        phi_file = tne.get_file_path(filename='phi')
        node2topic_max = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="max")
        concatenate_embeddings(node_embedding_file=node_embedding_file,
                               topic_embedding_file=topic_embedding_file,
                               node2topic=node2topic_max,
                               output_filename=concatenated_embedding_file_max)

        return self.read_embedding_file(node_embedding_file), self.read_embedding_file(concatenated_embedding_file_max)

    def learn_embedings(self, algorithm, nxg, base_folder, params):

        if algorithm == 'tne':
            return self.learn_tne_embeddings(nxg, base_folder, params)
        if algorithm == 'node2vec':
            pass

    def run(self, algorithm, ratio, params):

        train_g, train_pos, train_neg, test_pos, test_neg = self.split_into_train_test_sets(ratio=ratio)
        #test_g, _, _, test_pos, test_neg = self.split_into_train_test_sets(ratio=ratio)

        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]

        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        if algorithm == 'tne':
            embedding_folder = os.path.join(self.get_graph_name())
            node_embedding, final_embedding = self.learn_tne_embeddings(nxg=train_g,
                                                                        base_folder=embedding_folder,
                                                                        params=params)

            test_g = nx.Graph()
            test_g.add_edges_from(test_pos)

            test_node_embedding, test_final_embedding = self.learn_tne_embeddings(nxg=train_g,
                                                                                  base_folder=embedding_folder,
                                                                                  params=params)


            scores_node = {op: {'train': [], 'test': []} for op in _operators}
            scores_final = {op: {'train': [], 'test': []} for op in _operators}
            for op in _operators:
                print("tekrar")
                train_node_features = self.get_features_from_embeddings(edges=train_samples,
                                                                        embeddings=node_embedding,
                                                                        binary_operator=op)
                train_final_features = self.get_features_from_embeddings(edges=train_samples,
                                                                         embeddings=final_embedding,
                                                                         binary_operator=op)
                test_node_features = self.get_features_from_embeddings(edges=test_samples,
                                                                       embeddings=test_node_embedding,
                                                                       binary_operator=op)
                test_final_features = self.get_features_from_embeddings(edges=test_samples,
                                                                        embeddings=test_final_embedding,
                                                                        binary_operator=op)
                # For node
                for _ in range(1):
                    clf_node = LogisticRegression()
                    clf_node.fit(train_node_features, train_labels)

                    train_node_preds = clf_node.predict_proba(train_node_features)[:, 1]
                    test_node_preds = clf_node.predict_proba(test_node_features)[:, 1]

                    train_node_roc = roc_auc_score(y_true=train_labels, y_score=train_node_preds)
                    test_node_roc = roc_auc_score(y_true=test_labels, y_score=test_node_preds)

                    scores_node[op]['train'].append(train_node_roc)
                    scores_node[op]['test'].append(test_node_roc)

                    # For final
                    clf_final = LogisticRegression()
                    clf_final.fit(train_final_features, train_labels)

                    train_final_preds = clf_final.predict_proba(train_final_features)[:, 1]
                    test_final_preds = clf_final.predict_proba(test_final_features)[:, 1]

                    train_final_roc = roc_auc_score(y_true=train_labels, y_score=train_final_preds)
                    test_final_roc = roc_auc_score(y_true=test_labels, y_score=test_final_preds)

                    scores_final[op]['train'].append(train_final_roc)
                    scores_final[op]['test'].append(test_final_roc)

            return scores_node, scores_final


params = {}
params['method'] = 'deepwalk'
ratio = 0.5
params['number_of_walks'] = 80
params['number_of_topics'] = 80
params['lda_number_of_iters'] = 1000
params['walk_length'] = 5
params['window_size'] = 10
params['embedding_size'] = 128
params['dw_alpha'] = 0.0
params['n2v_p'] = 1
params['n2v_q'] = 1
params['lda_alpha'] = 50.0 / float(params['number_of_topics'])
params['lda_beta'] = 0.1



#for graphname in ["facebook_combined", "CA-AstroPh", "p2p-Gnutella09"]:
for graphname in ["facebook_combined"]:
    graph_path = "../examples/inputs/" + graphname + ".txt"

    ep = EdgePrediction()
    ep.read_graph(file_path=graph_path, file_type="edgelist")
    ep.set_graph_name(graphname)

    iter_count = 2
    total_scores_node = []
    total_scores_tne = []
    for iter in range(iter_count):
        scores_node, scores_final = ep.run(algorithm='tne', ratio=ratio, params=params)

        total_scores_node.append(scores_node)
        total_scores_tne.append(scores_final)


    import pickle

    with open("{}_{}_node.node".format(graphname, params['method']), "wb") as fp:
        pickle.dump(total_scores_node, fp)
    with open("{}_{}_tne.node".format(graphname, params['method']), "wb") as fp:
        pickle.dump(total_scores_tne, fp)
