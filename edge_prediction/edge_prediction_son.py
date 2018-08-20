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

# Library
from os.path import basename, splitext, join
sys.path.append("./TNE")
#from tne.tne import TNE
#from utils.utils import find_max_topic_for_nodes, concatenate_embeddings_max

"""
def run_tne(graph, graph_name, method, params):

    tne = TNE()
    tne.set_graph(graph, graph_name)

    params = {}
    params['number_of_walks'] = 8
    params['walk_length'] = 10
    params['window_size'] = 10
    params['embedding_size'] = 12
    params['p'] = 1
    params['q'] = 1
    number_of_topics = 65
    alpha = 50.0/float(number_of_topics)
    beta = 0.1
    number_of_iters = 100


    ## File paths ##
    base_desc = "{}_n{}_l{}_w{}_k{}_{}".format(graph_name,
                                               params['number_of_walks'],
                                               params['walk_length'],
                                               params['window_size'],
                                               number_of_topics,
                                               method)
    temp_folder = "../temp/{}".format(graph_name)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    outputs_folder = "../outputs"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    if not os.path.exists(outputs_folder):
        os.mkdir(outputs_folder)

    corpus_path_for_lda = join(temp_folder, "{}_lda_corpus.corpus".format(base_desc))

    node_embedding_file = join(outputs_folder, "{}_node.embedding".format(base_desc))
    topic_embedding_file = join(outputs_folder, "{}_topic.embedding".format(base_desc))
    concatenated_embedding_file_max = join(outputs_folder, "{}_final_max.embedding".format(base_desc))
    ## == ##


    tne.perform_random_walks(method=method, params=params)
    tne.save_corpus(corpus_path_for_lda, with_title=True)
    id2node = tne.run_lda(alpha=alpha, beta=beta, number_of_iters=number_of_iters,
                          number_of_topics=number_of_topics, lda_corpus_path=corpus_path_for_lda)
    tne.extract_node_embedding(node_embedding_file)
    tne.extract_topic_embedding(number_of_topics=number_of_topics,
                                topic_embedding_file=topic_embedding_file)

    phi_file = tne.get_file_path(filename='phi')
    node2topic_max = find_max_topic_for_nodes(phi_file, id2node, number_of_topics)
    # Concatenate the embeddings

    concatenate_embeddings_max(node_embedding_file=node_embedding_file,
                               topic_embedding_file=topic_embedding_file,
                               node2topic=node2topic_max,
                               output_filename=concatenated_embedding_file_max)

    return node_embedding_file, concatenated_embedding_file_max
"""

class EdgePrediction:
    def __init__(self):
        self.g = None

    def read_graph(self, file_path, file_type="gml"):
        if file_type == "gml":
            self.g = nx.read_gml(file_path)

        elif file_type == "edgelist":
            self.g = nx.read_edgelist(file_path)

        else:
            raise ValueError("Unknown graph type!")

        self.number_of_nodes = self.g.number_of_nodes()
        self.number_of_edges = self.g.number_of_edges()

        print("Number of nodes: {}".format(self.number_of_nodes))
        print("Number of edges: {}".format(self.number_of_edges))


    def get_graph(self):
        return self.g

    def split_into_train_test_sets(self, ratio, max_trial_limit=10000):

        test_set_size = int(ratio * self.number_of_edges)

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
                print("\r{0} tp edges found out of {1}".format(num_of_pos_samples, test_set_size))
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_samples == test_set_size:
                break

        if num_of_pos_samples != test_set_size:
            raise ValueError("Not pos edges found!")


        # Generate the negative samples
        train_neg_samples = []
        test_neg_samples = []

        non_edges = list(nx.non_edges(self.g))
        perm = np.arange(len(non_edges))
        np.random.shuffle(perm)
        non_edges = [non_edges[inx] for inx in perm]
        print("There are {} number of non-edges".format(len(non_edges)))
        chosen_non_edge_inx = np.random.choice(perm, size=test_set_size*2, replace=False)

        train_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[:test_set_size]]
        test_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx[test_set_size:]]

        train_pos_samples = list(residual_g.edges())
        return residual_g, train_pos_samples, train_neg_samples, test_pos_samples, test_neg_samples


    def sil_run_embeddings_method(self, method_name, nx_graph, output_embedding_file):
        if method_name == "deepwalk":
            pass
        if method_name == "node2vec":

            node2vec_path = "./node2vec/src/main.py"
            edge_list_path = "./temp_graph.edgelist"
            nx.write_edgelist(nx_graph, edge_list_path)


            cmd = sys.executable + " " + node2vec_path + " "
            cmd += "--input " + edge_list_path + " "
            cmd += "--output " + output_embedding_file + " "
            #cmd += "--num-walks 40 --walk-length 10 "
            cmd += "--p 1.0 --q 1.0"
            os.system(cmd)

        if method_name == "tne":
            pass
            """
            sys.path.append("./TNE")
            import tne
            tne = TNE("")
            tne.graph = nx_graph
            tne.number_of_nodes = nx_graph.number_of_nodes()
            tne.graph_name = "tne_dataset_name"
            params = {}
            params['number_of_walks'] = 80
            params['walk_length'] = 10
            params['window_size'] = 10
            params['embedding_size'] = 128
            params['p'] = 1
            params['q'] = 1
            corpus_path_for_lda = "./lda_node.corpus"

            tne.perform_random_walks(method_name="node2vec", params=params)
            tne.save_corpus(corpus_path_for_lda=corpus_path_for_lda, with_title=True)
            tne.run_lda(alpha=0.1, beta=50/0/65.0 , number_of_iters=1000, number_of_topics=65, corpus_path_for_lda=corpus_path_for_lda)
            """
        if method_name == "tne":
            sys.path.append("../TNE")
            #import tne
            #TNE()
            pass

    def read_embedding_file(self, file_path):

        with open(file_path, 'r') as f:
            f.readline() # Skip the first line
            embeddings = {}
            for line in f:
                tokens = line.strip().split()
                embeddings.update({str(tokens[0]): [float(val) for val in tokens[1:]]})

        return embeddings


    def get_features(self, edges, embeddings, binary_operator):

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

    def get_embeddings(self, method_name, nx_graph, output_embedding_file):
        if method_name == 'tne':
            node_emb, tne_emb = run_tne(graph=nx_graph, graph_name="facebook+{}".format(output_embedding_file),
                                        method="node2vec", params={})
            return self.read_embedding_file(node_emb), self.read_embedding_file(tne_emb)
        else:
            self.run_embeddings_method(method_name, nx_graph, output_embedding_file)
            return self.read_embedding_file(output_embedding_file)

    def sil_run(self, method_name, params):

        train_output_embedding_file = "./{}_train.embedding".format(method_name)
        test_output_embedding_file = "./{}_test.embedding".format(method_name)

        train_g, train_pos, train_neg, _, _ = self.split_into_train_test_sets(ratio=0.5)
        test_g, _, _, test_pos, test_neg = self.split_into_train_test_sets(ratio=0.5)


        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]

        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        train_embeddings = self.get_embeddings(method_name, train_g, train_output_embedding_file)
        test_embeddings = self.get_embeddings(method_name, test_g, test_output_embedding_file)

        scores = {op: {'train': [], 'test': []} for op in ["hadamard", "average", "l1", "l2"]}
        for op in ["hadamard", "average", "l1", "l2"]:
            train_features = self.get_features(edges=train_samples, embeddings=train_embeddings, binary_operator=op)
            test_features = self.get_features(edges=test_samples, embeddings=test_embeddings, binary_operator=op)

            clf = LogisticRegression()
            clf.fit(train_features, train_labels)

            train_pred_label = clf.predict_proba(train_features)[:, 1]
            test_pred_label = clf.predict_proba(test_features)[:, 1]

            train_roc = roc_auc_score(y_true=train_labels, y_score=train_pred_label)
            test_roc = roc_auc_score(y_true=test_labels, y_score=test_pred_label)
            #scores.update({op: {'train': train_roc, 'test': test_roc}})
            scores[op]['train'].append(train_roc)
            scores[op]['test'].append(test_roc)

        return scores


    def run_run(self, method_name, params):

        train_g, train_pos, train_neg, test_pos, test_neg = self.split_into_train_test_sets(ratio=0.5)

        test_g = nx.Graph()
        test_g.add_nodes_from(train_g.nodes())
        test_g.add_edges_from(test_pos)
        test_g.add_edges_from(test_neg)

        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]

        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        train_node_output_embedding_file = "facebook_node_train.embedding"
        test_node_output_embedding_file = "acebook_node_test.embedding"
        train_tne_output_embedding_file = "facebook_node_train.embedding"
        test_tne_output_embedding_file = "facebook_node_test.embedding"
        #train_node_emb, train_tne_emb = self.get_embeddings(method_name, train_g, train_output_embedding_file)
        #test_node_emb, test_tne_emb = self.get_embeddings(method_name, test_g, test_output_embedding_file)

        train_node_emb = self.read_embedding_file(train_node_output_embedding_file)
        test_node_emb = self.read_embedding_file(test_node_output_embedding_file)
        train_tne_emb = self.read_embedding_file(train_tne_output_embedding_file)
        test_tne_emb = self.read_embedding_file(test_tne_output_embedding_file)

        scores_node = {op: {'train': [], 'test': []} for op in ["hadamard", "average", "l1", "l2"]}
        scores_tne = {op: {'train': [], 'test': []} for op in ["hadamard", "average", "l1", "l2"]}
        for op in ["hadamard", "average", "l1", "l2"]:
            train_node_features = self.get_features(edges=train_samples, embeddings=train_node_emb, binary_operator=op)
            test_node_features = self.get_features(edges=test_samples, embeddings=test_node_emb, binary_operator=op)

            train_tne_features = self.get_features(edges=train_samples, embeddings=train_tne_emb, binary_operator=op)
            test_tne_features = self.get_features(edges=test_samples, embeddings=test_tne_emb, binary_operator=op)

            clf_node = LogisticRegression()
            clf_node.fit(train_node_features, train_labels)

            train_node_pred_label = clf_node.predict_proba(train_node_features)[:, 1]
            test_node_pred_label = clf_node.predict_proba(test_node_features)[:, 1]

            train_node_roc = roc_auc_score(y_true=train_labels, y_score=train_node_pred_label)
            test_node_roc = roc_auc_score(y_true=test_labels, y_score=test_node_pred_label)
            # scores.update({op: {'train': train_roc, 'test': test_roc}})
            scores_node[op]['train'].append(train_node_roc)
            scores_node[op]['test'].append(test_node_roc)

            clf_tne = LogisticRegression()
            clf_tne.fit(train_tne_features, train_labels)

            train_tne_pred_label = clf_tne.predict_proba(train_tne_features)[:, 1]
            test_tne_pred_label = clf_tne.predict_proba(test_tne_features)[:, 1]

            train_tne_roc = roc_auc_score(y_true=train_labels, y_score=train_tne_pred_label)
            test_tne_roc = roc_auc_score(y_true=test_labels, y_score=test_tne_pred_label)
            #scores.update({op: {'train': train_roc, 'test': test_roc}})
            scores_tne[op]['train'].append(train_tne_roc)
            scores_tne[op]['test'].append(test_tne_roc)

        return scores_node, scores_tne

"""
#graph_path = "../examples/inputs/facebook.gml"
graph_path = "../examples/inputs/facebook_combined.txt"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="edgelist")
g = ep.get_graph()
run_tne(graph=g, graph_name="facebook")


for _ in range(1):
    scores = ep.run(method_name="node2vec", params={})
    for op in scores:
        print("{}: train:{}, test:{}".format(op, np.mean(scores[op]['train']), np.mean(scores[op]['test'])))

run_tne(graph=)
"""
#graph_path = "../examples/inputs/facebook.gml"
graph_path = "../examples/inputs/facebook_combined.txt"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="edgelist")


for _ in range(1):
    scores_node, scores_tne = ep.run_run(method_name="tne", params={})
    for op_node, op_tne in zip(scores_node, scores_tne):
        print("{}: train node:{}, test node:{}".format(op_node, np.mean(scores_node[op_node]['train']), np.mean(scores_node[op_node]['test'])))
        print("{}: train tne:{}, test tne:{}".format(op_tne, np.mean(scores_tne[op_tne]['train']),
                                             np.mean(scores_tne[op_tne]['test'])))

