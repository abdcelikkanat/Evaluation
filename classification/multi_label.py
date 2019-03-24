import numpy as np
from collections import OrderedDict
import scipy.io as sio
import gensim
from evaluation.evaluation import Evaluation
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix


class NodeClassification(Evaluation):
    """
    Multi-label node classification
    """

    def __init__(self, embedding_file, graph_path, params={}):
        Evaluation.__init__(self)

        self._embedding_file = embedding_file
        self._graph_path = graph_path
        self._directed = params['directed'] if 'directed' in params else False

        self.results = None

    def evaluate(self, number_of_shuffles, training_ratios):
        g = self._get_networkx_graph(self._graph_path, directed=self._directed, params={})
        node2embedding = self.read_embedding_file(embedding_file_path=self._embedding_file)
        node2community = self.get_node2community(g)

        N = g.number_of_nodes()
        K = self.detect_number_of_communities(g)

        nodelist = [node for node in g.nodes()]
        x = np.asarray([node2embedding[node] for node in nodelist])
        label_matrix = [[1 if k in node2community[node] else 0 for k in range(K)] for node in nodelist]
        label_matrix = csr_matrix(label_matrix)

        results = {}
        averages = ['micro', 'macro']
        for average in averages:
            results[average] = OrderedDict()
            for ratio in training_ratios:
                results[average].update({ratio: []})

        for train_ratio in training_ratios:

            for _ in range(number_of_shuffles):
                # Shuffle the data
                shuffled_features, shuffled_labels = shuffle(x, label_matrix)

                # Get the training size
                train_size = int(train_ratio * N)
                # Divide the data into the training and test sets
                train_features = shuffled_features[0:train_size, :]
                train_labels = shuffled_labels[0:train_size]

                test_features = shuffled_features[train_size:, :]
                test_labels = shuffled_labels[train_size:]

                # Train the classifier
                ovr = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
                ovr.fit(train_features, train_labels)

                # Find the predictions, each node can have multiple labels
                test_prob = np.asarray(ovr.predict_proba(test_features))
                y_pred = []
                for i in range(test_labels.shape[0]):
                    k = test_labels[i].getnnz()  # The number of labels to be predicted
                    pred = test_prob[i, :].argsort()[-k:]
                    y_pred.append(pred)

                # Find the true labels
                y_true = [[] for _ in range(test_labels.shape[0])]
                co = test_labels.tocoo()
                for i, j in zip(co.row, co.col):
                    y_true[i].append(j)

                mlb = MultiLabelBinarizer(range(K))
                for average in averages:
                    score = f1_score(y_true=mlb.fit_transform(y_true),
                                     y_pred=mlb.fit_transform(y_pred),
                                     average=average)

                    results[average][train_ratio].append(score)

        self.results = results

    def get_results(self, shuffle_var, detailed=False, format='txt'):

        if format == 'txt':
            output = ""
            if detailed is True:
                for average in ["micro", "macro"]:
                    output += average + "\n"
                    for ratio in self.results[average]:
                        output += " percent {}%\n".format(ratio)
                        scores = self.results[average][ratio]
                        for num in range(len(scores)):
                            output += "  Shuffle #{}: {}\n".format(num, self.results[average][ratio][num])
                        output += "  Average: {} Std-dev: {}\n".format(np.mean(scores), np.std(scores))
                    output += "\n"
            else:
                output += "Training percents: " + " ".join("{0:.2f}%".format(p * 100) for p in self.results["micro"]) + "\n"
                for average in ["micro", "macro"]:
                    output += average + ": "
                    for ratio in self.results[average]:
                        scores = self.results[average][ratio]
                        output += "{0:.5g}".format(np.mean(scores))
                        if shuffle_var is True:
                            output += ":{1:.5g}".format(np.std(scores))
                        output += " "
                    output += "\n"

            return output

        if format == "npy":
            results = [[], []]
            for m, average in enumerate(["micro", "macro"]):
                for ratio in self.results[average]:
                    scores = self.results[average][ratio]
                    results[m].append(np.mean(scores))

            return results


    def print_results(self, shuffle_var, detailed=False):
        output = self.get_results(shuffle_var, detailed=detailed)
        print(output)

    def save_results(self, output_file, shuffle_var, detailed=False, save_format="txt"):
        output = self.get_results(shuffle_var, detailed=detailed, format=save_format)

        if save_format == "txt":
            with open(output_file, 'w') as f:
                f.write(output)
        if save_format == "npy":
            np.save(output_file, output)





