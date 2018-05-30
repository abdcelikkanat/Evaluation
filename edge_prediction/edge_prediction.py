import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

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
        num_of_removed_edges = 0
        trial_counter = 0

        while num_of_removed_edges < test_set_size:
            # Randomly choose an edge index
            pos_inx = np.arange(residual_g.number_of_edges())
            np.random.shuffle(pos_inx)
            edge_inx = np.random.choice(a=pos_inx)
            # Remove the chosen edge
            chosen_edge = list(residual_g.edges())[edge_inx]
            residual_g.remove_edge(chosen_edge[0], chosen_edge[1])

            if num_of_ccs == nx.number_connected_components(residual_g):
                num_of_removed_edges += 1
                test_pos_samples.append(chosen_edge)
                trial_counter = 0
            else:
                residual_g.add_edge(chosen_edge[0], chosen_edge[1])
                trial_counter += 1

            if trial_counter == max_trial_limit:
                raise ValueError("In {} trial, any possible edge for removing could not be found!")

        # Generate the negative samples
        test_neg_samples = []

        num_of_neg_samples = 0
        while num_of_neg_samples < test_set_size:

            pos_inx = np.arange(self.g.number_of_nodes())
            np.random.shuffle(pos_inx)
            # Self-loops are allowed
            u, v = np.random.choice(a=pos_inx, size=2)

            candiate_edge = (unicode(u), unicode(v))
            if not self.g.has_edge(candiate_edge[0], candiate_edge[1]) and candiate_edge not in self.g.edges():
                test_neg_samples.append(candiate_edge)
                num_of_neg_samples += 1

        return residual_g, test_pos_samples, test_neg_samples

    def train(self, train_graph, test_edges):

        pos_samples, neg_samples = test_edges
        n = train_graph.number_of_nodes()

        coeff_matrix = np.zeros(shape=(n, n), dtype=np.float)

        samples = pos_samples + neg_samples
        print(samples)
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


graph_path = "../examples/inputs/facebook.gml"
ep = EdgePrediction()
ep.read_graph(file_path=graph_path, file_type="gml")
train, pos, neg = ep.split_into_train_test_sets(ratio=0.5)
m = ep.train(ep.g, (pos, neg))
print(m)

#https://github.com/adocherty/node2vec_linkprediction/blob/master/link_prediction.py
#node2vec link prediction github
