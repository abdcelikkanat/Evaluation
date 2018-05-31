import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


def compute_vectorial(vec1, vec2, method):
    if method == "hadamard":
        return np.multiply(vec1, vec2)
    if method == "average":
        return 0.5 * (vec1 + vec2)
    if method == "l1":
        return np.abs(vec1-vec2)
    if method == "l2":
        return np.abs(vec1 - vec2) ** 2


class EdgePrediction:
    def __init__(self):
        self.g = None

    def read_graph(self, file_path, file_type="gml"):
        if file_type == "gml":
            self.g = nx.read_gml(file_path)
            self.number_of_nodes = self.g.number_of_nodes()
            self.number_of_edges = self.g.number_of_edges()
        else:
            raise ValueError("Unknown graph type!")

        print("Number of nodes: {}".format(self.number_of_nodes))
        print("Number of edges: {}".format(self.number_of_edges))

    def split_into_train_test_sets(self, ratio, max_trial_limit=10000):

        test_set_size = int(ratio * self.number_of_edges)
        train_set_size = self.number_of_edges - test_set_size

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
                print("\r{0} tp edges found out of {1}".format(num_of_pos_samples, test_set_size)),
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])

            if num_of_pos_samples == test_set_size:
                break

        if num_of_pos_samples != test_set_size:
            raise ValueError("Not pos edges found!")


        # Generate the negative samples
        test_neg_samples = []

        non_edges = list(nx.non_edges(self.g))
        perm = np.arange(len(non_edges))
        np.random.shuffle(perm)
        non_edges = [non_edges[inx] for inx in perm]

        chosen_non_edge_inx = np.random.choice(perm, size=test_set_size, replace=False)

        test_neg_samples = [non_edges[perm[p]] for p in chosen_non_edge_inx]

        return residual_g, test_pos_samples, test_neg_samples

    def train(self, train_graph, test_edges):

        pos_samples, neg_samples = test_edges
        n = train_graph.number_of_nodes()

        coeff_matrix = np.zeros(shape=(n, n), dtype=np.float)

        samples = pos_samples + neg_samples

        preds = nx.jaccard_coefficient(train_graph, samples)
        for i, j, p in preds:
            coeff_matrix[int(i), int(j)] = p
            coeff_matrix[int(j), int(i)] = p

        coeff_matrix = coeff_matrix / coeff_matrix.max()

        ytrue = [1 for _ in range(len(pos_samples))] + [0 for _ in range(len(neg_samples))]
        y_score = [coeff_matrix[int(edge[0]), int(edge[1])] for edge in pos_samples] + [coeff_matrix[int(edge[0]), int(edge[1])] for edge in neg_samples]

        auc = roc_auc_score(y_true=ytrue, y_score=y_score)

        print(auc)

        return auc

    def learn_embeddings(self, method, nx_graph, output_embedding_file):
        if method == "deepwalk":
            pass
        if method == "node2vec":

            node2vec_path = "./node2vec/src/main.py"
            sys.path.append(node2vec_path)

            edge_list_path = "./temp_graph.edgelist"
            nx.write_edgelist(nx_graph, edge_list_path)


            cmd = "python " + node2vec_path + " "
            cmd += "--input " + edge_list_path + " "
            cmd += "--walk-length 10 --num-walks 40 "
            cmd += "--output " + output_embedding_file
            os.system(cmd)

    def read_embedding_file(self, file_path):

        with open(file_path, 'r') as f:
            f.readline() # Skip the first line
            embeddings = {}
            for line in f:
                tokens = line.strip().split()
                embeddings.update({tokens[0]: [float(val) for val in tokens[1:]]})

        return embeddings

    def compute_features(self, nxg, edges, metric, params):

        features = []
        if metric == "jaccard":
            for i in range(len(edges)):
                for _, _, val in nx.jaccard_coefficient(nxg, [edges[i]]):
                   features.append(val)

            features = np.asarray(features)

        if metric == "embedding":
            method_name = params['method_name']
            distance = params['distance']

            output_embedding_file = "graph.embedding"
            self.learn_embeddings(method=method_name, nx_graph=nxg, output_embedding_file=output_embedding_file)
            embeddings = self.read_embedding_file(file_path=output_embedding_file)


            for edge in edges:
                val = compute_vectorial(embeddings[edge[0]], embeddings[edge[1]], method=distance)
                features.append(val)

        return features

    def run(self, metric, params):
        train_residual_g, train_pos, train_neg = self.split_into_train_test_sets(ratio=0.5)

        train_samples = train_pos + train_neg
        train_labels = [1 for _ in train_pos] + [0 for _ in train_neg]
        train_features = self.compute_features(nxg=train_residual_g, edges=train_samples, metric="embedding", params=params)

        scaler = StandardScaler()
        lin_clf = LogisticRegression()
        clf = pipeline.make_pipeline(scaler, lin_clf)
        # Train classifier
        clf.fit(train_features, train_labels)
        auc_train = metrics.scorer.roc_auc_scorer(clf, train_features, train_labels)
        avg = metrics.scorer.average_precision_scorer(clf, train_features, train_labels)
        # Test classifier
        #auc_test = metrics.scorer.roc_auc_scorer(clf, test_features, test_labels)
        print("Train-Test: {}, {}".format(auc_train, avg))

graph_path = "../examples/inputs/facebook.gml"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="gml")
params = {'method_name': 'node2vec', 'distance': 'hadamard'}
ep.run(metric="jaccard", params=params)


