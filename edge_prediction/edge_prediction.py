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
import pickle

from graphbase.graphbase import *
from edge_prediction.learn_embeddings import learn_embeddings


_operators = ["hadamard", "average", "l1", "l2"]


class EdgePrediction(GraphBase):

    def __init__(self):
        GraphBase.__init__(self)

    def split_into_train_test_sets(self, ratio):
        # Split the graph into two disjoint sets while keeping the training graph connected

        test_set_size = int(ratio * self.g.number_of_edges())

        # Generate the positive test edges
        test_pos_samples = []
        residual_g = self.g.copy()
        num_of_ccs = nx.number_connected_components(residual_g)
        if num_of_ccs != 1:
            raise ValueError("The graph contains more than one connected component!")

        num_of_pos_test_samples = 0

        edges = list(residual_g.edges())
        perm = np.arange(len(edges))
        np.random.shuffle(perm)
        edges = [edges[inx] for inx in perm]

        for i in range(len(edges)):
            # Remove the chosen edge
            chosen_edge = edges[i]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            if chosen_edge[1] in nx.connected._plain_bfs(residual_g, chosen_edge[0]):
                num_of_pos_test_samples += 1
                test_pos_samples.append(chosen_edge)
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_test_samples == test_set_size:
                break

        if num_of_pos_test_samples != test_set_size:
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

    def get_feature_vectors_from_embeddings(self, edges, embeddings, binary_operator):

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

    def get_scores(self, train_test_ratio, params, file_desc, temp_folder, embedding_folder):

        # Divide the data into training and test sets
        residual_g, train_pos, train_neg, test_pos, test_neg = self.split_into_train_test_sets(ratio=train_test_ratio)
        # Save the residual graph
        residual_g_save_path = os.path.join(temp_folder, self.get_graph_name() + "_residual.gml")
        nx.write_gml(residual_g, residual_g_save_path)
        #test_g, _, _, test_pos, test_neg = self.split_into_train_test_sets(ratio=ratio)
        # Prepare the positive and negative samples for training set
        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]
        # Prepare the positive and negative samples for test set
        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        # Run TNE
        train_node_embedding, train_final_embedding = learn_embeddings(nx_graph_path=residual_g_save_path,
                                                                       params=params,
                                                                       file_desc=file_desc,
                                                                       temp_folder=temp_folder,
                                                                       embedding_folder=embedding_folder)
        """
        test_g = nx.Graph()
        test_g.add_nodes_from(train_g.nodes())
        test_g.add_edges_from(test_pos)
        """
        #test_embedding_folder = os.path.join(self.get_graph_name(), "test")
        #test_node_embedding, test_final_embedding = learn_tne_embeddings(nxg=residual_g,
        #                                                                 base_folder=test_embedding_folder,
        #                                                                 params=params)

        scores_node = {op: {'train': [], 'test': []} for op in _operators}
        scores_final = {op: {'train': [], 'test': []} for op in _operators}

        for op in _operators:

            train_node_features = self.get_feature_vectors_from_embeddings(edges=train_samples,
                                                                           embeddings=train_node_embedding,
                                                                           binary_operator=op)
            train_final_features = self.get_feature_vectors_from_embeddings(edges=train_samples,
                                                                            embeddings=train_final_embedding,
                                                                            binary_operator=op)
            test_node_features = self.get_feature_vectors_from_embeddings(edges=test_samples,
                                                                          embeddings=train_node_embedding,
                                                                          binary_operator=op)
            test_final_features = self.get_feature_vectors_from_embeddings(edges=test_samples,
                                                                           embeddings=train_final_embedding,
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

    def split_network(self, train_test_ratio, target_folder):

        # Divide the data into training and test sets
        residual_g, train_pos, train_neg, test_pos, test_neg = self.split_into_train_test_sets(ratio=train_test_ratio)

        # Prepare the positive and negative samples for training set
        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]
        # Prepare the positive and negative samples for test set
        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        # Check if the target folder exists or not
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)

        # Save the residual network
        residual_g_save_path = os.path.join(target_folder, self.get_graph_name() + "_residual.gml")
        nx.write_gml(residual_g, residual_g_save_path)

        # Save positive and negative samples for training and test sets
        save_file_path = os.path.join(target_folder, self.get_graph_name() + "_samples.pkl")
        with open(save_file_path, 'wb') as f:
            pickle.dump({'train': {'edges':train_samples, 'labels': train_labels },
                         'test': {'edges':test_samples, 'labels': test_labels}}, f, pickle.HIGHEST_PROTOCOL)

    def read_samples(self, file_path):

        with open(file_path, 'rb') as f:
            temp = pickle.load(f)
            #residual_g = temp['residual_g']
            train_samples, train_labels = temp['train']['edges'], temp['train']['labels']
            test_samples, test_labels = temp['test']['edges'], temp['test']['labels']

            return train_samples, train_labels, test_samples, test_labels

    def predict(self, embedding_file_path, train_samples, train_labels, test_samples, test_labels):

        embeddings = {}
        with open(embedding_file_path, 'r') as fin:
            # skip the first line
            num_of_nodes, dim = fin.readline().strip().split()
            # read the embeddings
            for line in fin.readlines():
                tokens = line.strip().split()
                embeddings[tokens[0]] = [float(v) for v in tokens[1:]]

        scores = {op: {'train': [], 'test': []} for op in _operators}

        for op in _operators:

            train_features = self.get_feature_vectors_from_embeddings(edges=train_samples,
                                                                      embeddings=embeddings,
                                                                      binary_operator=op)

            test_features = self.get_feature_vectors_from_embeddings(edges=test_samples,
                                                                     embeddings=embeddings,
                                                                     binary_operator=op)
            # For node
            for _ in range(1):
                clf = LogisticRegression()
                clf.fit(train_features, train_labels)

                train_preds = clf.predict_proba(train_features)[:, 1]
                test_preds = clf.predict_proba(test_features)[:, 1]

                train_roc = roc_auc_score(y_true=train_labels, y_score=train_preds)
                test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)

                scores[op]['train'].append(train_roc)
                scores[op]['test'].append(test_roc)

        return scores

    def precision_recall_at_k(self, embedding_file_path, train_samples, train_labels, test_samples, test_labels):

        embeddings = {}
        with open(embedding_file_path, 'r') as fin:
            # skip the first line
            num_of_nodes, dim = fin.readline().strip().split()
            # read the embeddings
            for line in fin.readlines():
                tokens = line.strip().split()
                embeddings[tokens[0]] = [float(v) for v in tokens[1:]]

        num_of_samples = len(test_samples)
        #scores = {op: {'train': [], 'test': []} for op in _operators}
        precisions = {op: [0.0 for _ in range(num_of_samples)] for op in _operators}
        recalls = {op: [0.0 for _ in range(num_of_samples)] for op in _operators}

        for op in _operators:

            train_features = self.get_feature_vectors_from_embeddings(edges=train_samples,
                                                                      embeddings=embeddings,
                                                                      binary_operator=op)

            test_features = self.get_feature_vectors_from_embeddings(edges=test_samples,
                                                                     embeddings=embeddings,
                                                                     binary_operator=op)

            clf = LogisticRegression()
            clf.fit(train_features, train_labels)

            test_pred_probs = clf.predict_proba(test_features)[:, 1]
            test_true_labels = np.asarray(test_labels, dtype=np.int)

            data = np.vstack((test_pred_probs, test_true_labels)).T
            sorted_data = np.asarray(sorted(data, key=lambda x: x[0], reverse=True))

            test_pred_probs = sorted_data[:, 0]
            test_true_labels = sorted_data[:, 1]

            n_rec_k = 0.0
            n_rel = sum(test_true_labels == 1)
            n_rel_and_rec_k = 0.0
            for k in range(num_of_samples):
                # Number of recommended items in top k
                n_rec_k += (test_pred_probs[k] >= 0.5)
                # Number of relevant and recommended items in top k
                n_rel_and_rec_k += (test_pred_probs[k] >= 0.5 and test_true_labels[k] == 1)
                # Precision@K: Proportion of recommended items that are relevant
                precisions[op][k] = n_rel_and_rec_k / n_rec_k #

                recalls[op][k] = n_rel_and_rec_k / float(n_rel)

            '''
            #print(len(test_labels))
            k = 80000
            # Number of recommended items in top k
            n_rec_k = float(sum(test_pred_probs[:k] >= 0.5))

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum((test_pred_probs[kk] >= 0.5 and test_true_labels[kk] == 1) for kk in range(k))

            # Precision@K: Proportion of recommended items that are relevant
            precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            '''

        return precisions, recalls

    def precision_recall_at_k_yeni(self, embedding_file_path, train_samples, train_labels, test_samples, test_labels):

        embeddings = {}
        with open(embedding_file_path, 'r') as fin:
            # skip the first line
            num_of_nodes, dim = fin.readline().strip().split()
            # read the embeddings
            for line in fin.readlines():
                tokens = line.strip().split()
                embeddings[tokens[0]] = [float(v) for v in tokens[1:]]

        num_of_samples = len(test_samples)
        #scores = {op: {'train': [], 'test': []} for op in _operators}
        precisions = {op: [0.0 for _ in range(num_of_samples)] for op in _operators}
        recalls = {op: [0.0 for _ in range(num_of_samples)] for op in _operators}

        for op in _operators:

            train_features = self.get_feature_vectors_from_embeddings(edges=train_samples,
                                                                      embeddings=embeddings,
                                                                      binary_operator=op)

            test_features = self.get_feature_vectors_from_embeddings(edges=test_samples,
                                                                     embeddings=embeddings,
                                                                     binary_operator=op)

            clf = LogisticRegression()
            clf.fit(train_features, train_labels)

            test_pred_probs = clf.predict_proba(test_features)[:, 1]
            test_true_labels = np.asarray(test_labels, dtype=np.int)

            data = np.vstack((test_pred_probs, test_true_labels)).T
            sorted_data = np.asarray(sorted(data, key=lambda x: x[0], reverse=True))

            #test_pred_probs = sorted_data[:, 0]
            test_true_labels = sorted_data[:, 1]

            n_rec_k = 0.0
            n_rel = sum(test_true_labels == 1)
            n_rel_and_rec_k = 0.0

            for k in range(num_of_samples):
                # Number of recommended items in top k
                n_rec_k += 1
                # Number of relevant and recommended items in top k
                n_rel_and_rec_k += ( test_true_labels[k] == 1 )

                # Precision@K: Proportion of recommended items that are relevant
                precisions[op][k] = n_rel_and_rec_k / n_rec_k

                recalls[op][k] = n_rel_and_rec_k / float(n_rel)

            '''
            #print(len(test_labels))
            k = 80000
            # Number of recommended items in top k
            n_rec_k = float(sum(test_pred_probs[:k] >= 0.5))

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum((test_pred_probs[kk] >= 0.5 and test_true_labels[kk] == 1) for kk in range(k))

            # Precision@K: Proportion of recommended items that are relevant
            precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
            '''

        return precisions, recalls


"""

params = {}
params['method'] = 'deepwalk'
ratio = 0.5
params['number_of_walks'] = 8
params['number_of_topics'] = 8
params['lda_number_of_iters'] = 1000
params['walk_length'] = 10
params['window_size'] = 10
params['embedding_size'] = 128
params['dw_alpha'] = 0.0
params['n2v_p'] = 1.0
params['n2v_q'] = 1.0
params['lda_alpha'] = 50.0 / float(params['number_of_topics'])
params['lda_beta'] = 0.1
# Parameters for ComWalk
params['cw_p'] = 4.0
params['cw_r'] = 2.0
params['cw_q'] = 0.5


#for graphname in ["facebook_combined", "CA-AstroPh", "p2p-Gnutella09"]:
for graphname in ["facebook_combined"]:
    graph_path = "../examples/inputs/" + graphname + ".edgelist"

    ep = EdgePrediction()
    ep.read_graph(graph_path=graph_path)
    ep.set_graph_name(graphname)
    params['graph_name'] = graphname

    iter_count = 1
    total_scores_node = []
    total_scores_tne = []
    for iter in range(iter_count):
        scores_node, scores_final = ep.run(algorithm='tne', ratio=ratio, params=params)

        total_scores_node.append(scores_node)
        total_scores_tne.append(scores_final)

    scores_folder = os.path.join("./scores", graphname)
    if not os.path.exists(scores_folder):
        os.makedirs(scores_folder)

    with open(os.path.join(scores_folder,"{}_{}_node.score".format(graphname, params['method'])), "wb") as fp:
        pickle.dump(total_scores_node, fp)
    with open(os.path.join(scores_folder,"{}_{}_final.score".format(graphname, params['method'])), "wb") as fp:
        pickle.dump(total_scores_tne, fp)

"""