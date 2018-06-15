import os
import networkx as nx


class GraphBase:
    def __init__(self):
        self.g = None
        self.number_of_clusters = 0

    def read_graph(self, graph_path):

        ext = os.path.splitext(graph_path)[-1].lower()

        if ext == ".gml":
            self.g = nx.read_gml(graph_path)

        else:
            raise ValueError("Invalid graph file format!")

    def set_graph(self, nxg):
        self.g = nxg

    def set_number_of_clusters(self, value):
        self.number_of_clusters = value

    def detect_number_of_clusters(self):

        # It is assumed that the labels of clusters starts from 0 to K-1
        max_cluster_label = -1
        clusters = nx.get_node_attributes(self.g, "clusters")
        for node in self.g.nodes():
            c_max = max(list(clusters[node]))
            if c_max > max_cluster_label:
                max_cluster_label = c_max

        return max_cluster_label + 1