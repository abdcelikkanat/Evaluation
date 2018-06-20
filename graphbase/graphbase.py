import os
import networkx as nx


class GraphBase:
    def __init__(self):
        self.g = None
        self.number_of_communities = 0

    def read_graph(self, graph_path):

        ext = os.path.splitext(graph_path)[-1].lower()

        if ext == ".gml":
            self.g = nx.read_gml(graph_path)

        else:
            raise ValueError("Invalid graph file format!")

    def set_graph(self, nxg):
        self.g = nxg

    def set_number_of_communities(self, value):
        self.number_of_communities = value

    def detect_number_of_communities(self):

        # It is assumed that the labels of communities starts from 0 to K-1
        max_community_label = -1
        communities = nx.get_node_attributes(self.g, "community")
        for node in self.g.nodes():
            comm_list = communities[node]
            if type(comm_list) is int:
                comm_list = [comm_list]

            c_max = max(comm_list)
            if c_max > max_community_label:
                max_community_label = c_max

        return max_community_label + 1