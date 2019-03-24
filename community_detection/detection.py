import networkx as nx
import gensim
from scipy.optimize import linear_sum_assignment
from evaluation.evaluation import Evaluation
from sklearn.cluster import KMeans
from evaluation.evaluation import Evaluation
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, confusion_matrix, adjusted_rand_score
from sklearn import metrics
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.cluster import contingency_matrix


class CommunityDetection(Evaluation):

    def __init__(self, embedding_file, graph_path, params={}):
        Evaluation.__init__(self)

        self._embedding_file = embedding_file
        self._graph_path = graph_path
        self._directed = params['directed'] if 'directed' in params else False

        self.results = None

    def _compute_nmi_score(self, id2node, node2community_true_labels, node2community_pred_labels):
        # Normalized Mutual Information (NMI) has been proposed as a performance metric for community detection
        # It is implemented only for non-overlapping communities

        n = len(id2node)
        labels_true = [node2community_true_labels[id2node[inx]][0] for inx in range(n)]
        labels_pred = [node2community_pred_labels[id2node[inx]][0] for inx in range(n)]

        score = normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)

        return score

    def _compute_ccr_score(self, id2node, node2community_true_labels, node2community_pred_labels):
        # Correct Classification Rate (CCR) has been proposed as a performance metric for community detection
        # It is implemented only for non-overlapping communities
        n = len(id2node)
        labels_true = [node2community_true_labels[id2node[inx]][0] for inx in range(n)]
        labels_pred = [node2community_pred_labels[id2node[inx]][0] for inx in range(n)]

        conf_matrix = confusion_matrix(labels_true, labels_pred)
        # Find the matching community labels
        r, c = linear_sum_assignment(-conf_matrix)  # uses the Hungarian algorithm to solve it

        score = float(conf_matrix[r, c].sum()) / float(n)

        return score

    def _compute_entropy(self, labels):

        return entropy(np.bincount(labels).astype(np.float))

    def _compute_vi_score(self, id2node, node2community_true_labels, node2community_pred_labels):
        # Computes the variational information score
        # V(X,Y) = H(X) + H(Y) - 2I(X;Y)

        n = len(id2node)
        labels_true = [node2community_true_labels[id2node[inx]][0] for inx in range(n)]
        labels_pred = [node2community_pred_labels[id2node[inx]][0] for inx in range(n)]

        h_true, h_pred = self._compute_entropy(labels_true), self._compute_entropy(labels_pred)

        con_matrix = np.array(contingency_matrix(labels_true=labels_true, labels_pred=labels_pred), dtype=np.float)
        mi = mutual_info_score(labels_true=labels_true, labels_pred=labels_pred, contingency=con_matrix)

        score = h_true + h_pred - 2.0*mi

        return score

    def _compute_ari_score(self, id2node, node2community_true_labels, node2community_pred_labels):
        # Compute the adjusted rand index score

        n = len(id2node)
        labels_true = [node2community_true_labels[id2node[inx]][0] for inx in range(n)]
        labels_pred = [node2community_pred_labels[id2node[inx]][0] for inx in range(n)]

        score = adjusted_rand_score(labels_true=labels_true, labels_pred=labels_pred)

        return score

    def _get_node2community(self, nxg):

        node2community = nx.get_node_attributes(nxg, name='community')
        for node in node2community:
            if type(node2community[node]) == int:
                node2community[node] = [node2community[node]]

        return node2community

    def evaluate(self, num_of_communities=None):
        # Read the graph
        g = self._get_networkx_graph(self._graph_path, directed=self._directed, params={})

        if num_of_communities is None:
            num_of_communities = self.detect_number_of_communities(nxg=g)

        # Read the embedding file
        node_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(self._embedding_file, binary=False)
        # Convert embeddings into feature vectors
        id2node = [node for node in g.nodes()]
        x = [node_model[id2node[inx]] for inx in range(g.number_of_nodes())]
        # Apply k-means algorithm

        kmeans = KMeans(n_clusters=num_of_communities, random_state=0).fit(x)
        labels = kmeans.labels_.tolist()

        node2community_true = self._get_node2community(nxg=g)
        node2community_pred = {id2node[inx]: [labels[inx]] for inx in range(g.number_of_nodes())}

        # print(communities_true)
        # print(communities_pred)

        nmi_score = self._compute_nmi_score(id2node, node2community_true, node2community_pred)
        ccr_score = self._compute_ccr_score(id2node, node2community_true, node2community_pred)
        vi_score = self._compute_vi_score(id2node, node2community_true, node2community_pred)
        ari_score = self._compute_ari_score(id2node, node2community_true, node2community_pred)

        return nmi_score, ccr_score, vi_score, ari_score


