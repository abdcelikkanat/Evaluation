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


    def run_embeddings_method(self, method_name, nx_graph, output_embedding_file):
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
            sys.path.append("./TNE")
            import tne
            tne = TNE("")
            tne.graph = nx_graph
            tne.number_of_nodes = nx_graph.number_of_nodes()
            tne.graph_name = "tne_dataset_name"
            tne.perform_random_walks(method_name="node2vec", )

    def read_embedding_file(self, file_path):

        with open(file_path, 'r') as f:
            f.readline() # Skip the first line
            embeddings = {}
            for line in f:
                tokens = line.strip().split()
                embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})

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
                value = abs(vec1 + vec2)**2

            features.append(value)

        return np.asarray(features)

    def get_embeddings(self, method_name, nx_graph, output_embedding_file):
        self.run_embeddings_method(method_name, nx_graph, output_embedding_file)
        return self.read_embedding_file(output_embedding_file)

    def run(self, method_name, params):

        output_embedding_file = "./{}.embedding".format(method_name)

        train_g, train_pos, train_neg, test_pos, test_neg = self.split_into_train_test_sets(ratio=0.5)

        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]

        test_samples = test_pos + test_neg
        test_labels = [1 for _ in test_pos] + [0 for _ in test_neg]

        embeddings = self.get_embeddings(method_name, train_g, output_embedding_file)

        scores = {}
        for op in ["hadamard", "average", "l1", "l2"]:
            train_features = self.get_features(edges=train_samples, embeddings=embeddings, binary_operator=op)
            test_features = self.get_features(edges=test_samples, embeddings=embeddings, binary_operator=op)

            clf = LogisticRegression()
            clf.fit(train_features, train_labels)
            train_pred_label = clf.predict_proba(train_features)[:, 1]
            test_pred_label = clf.predict_proba(test_features)[:, 1]

            train_roc = roc_auc_score(y_true=train_labels, y_score=train_pred_label)
            test_roc = roc_auc_score(y_true=test_labels, y_score=test_pred_label)
            scores.update({op: {'train': train_roc, 'test': test_roc}})

        return scores

#graph_path = "../examples/inputs/facebook.gml"
graph_path = "../examples/inputs/facebook_combined.txt"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="edgelist")

scores = ep.run(method_name="tne", params={})

for op in scores:
    print("{}: {}".format(op, scores[op]))
