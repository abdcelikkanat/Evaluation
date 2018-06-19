import os
import numpy as np
import networkx as nx
from graphbase.graphbase import *
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.preprocessing import MultiLabelBinarizer


class CommunityDetection(GraphBase):

    def __init__(self):
        GraphBase.__init__(self)
        self.number_of_clusters = 0

    def detect_corresponding_clusters(self, clusters_pred, best_matching):
        # if best_matching is 0,
        #       it finds the the best matching ground-truth community to each detected community
        # if best_matching is 1,
        #       it finds the best-matching detected community to each ground-truth community

        cluster_pairs_f1 = np.zeros(shape=(self.number_of_clusters, self.number_of_clusters), dtype=np.float)

        node2cluster_true_labels = nx.get_node_attributes(self.g, "clusters")

        id2node = [node for node in self.g.nodes()]

        # Find the corresponding cluster labels
        for cluster_id in range(self.number_of_clusters):
            for estimated_cluster_id in range(self.number_of_clusters):
                y_true = [1 if cluster_id in list(node2cluster_true_labels[id2node[i]]) else 0 for i in
                          range(self.g.number_of_nodes())]
                y_pred = [1 if estimated_cluster_id in list(clusters_pred[id2node[i]]) else 0 for i in
                          range(self.g.number_of_nodes())]

                cluster_pairs_f1[cluster_id, estimated_cluster_id] = f1_score(y_true=y_true, y_pred=y_pred)

        corr_cluster_labels = np.argmax(cluster_pairs_f1, axis=best_matching)

        # Check that all labels are distinct, otherwise it means that multiple are assigned to the same label
        if corr_cluster_labels.size != np.unique(corr_cluster_labels).size:
            raise ValueError("A label has been assigned multiple times!")

        return corr_cluster_labels.tolist()

    def replace_assignments(self):
        pass


    def avg_f1_score(self, clusters_pred):
        pass
        """
        id2node = [node for node in self.g.nodes()]
        node2cluster_true_labels = nx.get_node_attributes(self.g, "clusters")

        corr_cluster_labels = self.detect_corresponding_clusters(clusters_pred=clusters_pred, best_matching=0)

        K = self.number_of_clusters
        for k in range(K):
            y_true = [1 if k in list(node2cluster_true_labels[id2node[i]]) else 0 for i in
                      range(self.g.number_of_nodes())]
            #node2cluster_replaced_labels = [list(node2cluster_true_labels[id2node[i]]) for i in range(self.g.number_of_nodes())]
            y_pred = [1 if k in list(node2cluster_true_labels[id2node[i]]) else 0 for i in
                      range(self.g.number_of_nodes())]


        cluster_pairs_f1[cluster_id, estimated_cluster_id] = f1_score(y_true=y_true, y_pred=y_pred)

        """

    def nmi_score(self, node2community_pred_labels):
        # Normalized Mutual Information (NMI) has been proposed as a performance metric for community detection

        id2node = [node for node in self.g.nodes()]
        node2community_true_labels = nx.get_node_attributes(self.g, "community")

        labels_true = [node2community_true_labels[node][0] for node in self.g.nodes()]
        labels_pred = [node2community_pred_labels[node][0] for node in self.g.nodes()]

        score = normalized_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)

        return score

"""
comdetect = CommunityDetection()
comdetect.read_graph("../datasets/blogcatalog.gml")
comdetect.avg_f1_score()

"""

""" 
# test 1
testg = nx.Graph()
for i in range(6):
    testg.add_node(str(i))

cluster_labels = {'0': [4], '1': [3], '2': [2], '3': [1], '4': [0], '5': [1]}
nx.set_node_attributes(testg, values=cluster_labels, name='clusters')

comdetect = CommunityDetection()
comdetect.set_graph(nxg=testg)
number_of_clusters = comdetect.detect_number_of_clusters()
comdetect.set_number_of_clusters(number_of_clusters)
clusters_pred = {'0': [0], '1': [1], '2': [2], '3': [3], '4': [4], '5': [2]}
comdetect.avg_f1_score(clusters_pred=clusters_pred)
comdetect.nmi_score(clusters_pred)

"""