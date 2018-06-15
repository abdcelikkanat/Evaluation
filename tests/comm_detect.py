import unittest
import networkx as nx
import community_detection.community_detection as comm


class TestStringMethods(unittest.TestCase):

    def test_detect_corr_comm1(self):

        testg = nx.Graph()
        testg.add_nodes_from([str(i) for i in range(5)])

        cluster_true_labels = {'0': [4], '1': [3], '2': [2], '3': [1], '4': [0]}
        clusters_pred_labels = {'0': [0], '1': [1], '2': [2], '3': [3], '4': [4]}

        nx.set_node_attributes(testg, values=cluster_true_labels, name='clusters')

        comdetect = comm.CommunityDetection()
        comdetect.set_graph(nxg=testg)
        number_of_clusters = comdetect.detect_number_of_clusters()
        comdetect.set_number_of_clusters(number_of_clusters)

        corr_cluster_labels = comdetect.detect_corresponding_clusters(clusters_pred_labels)

        self.assertListEqual([4, 3, 2, 1, 0], corr_cluster_labels)

    def test_detect_corr_comm2(self):

        testg = nx.Graph()
        testg.add_nodes_from([str(i) for i in range(5)])

        cluster_true_labels = {'0': [2, 4], '1': [3], '2': [0, 2], '3': [1], '4': [0, 1, 2]}
        clusters_pred_labels = {'0': [0, 2], '1': [1], '2': [3], '3': [0, 1, 2], '4': [2, 4]}

        nx.set_node_attributes(testg, values=cluster_true_labels, name='clusters')

        comdetect = comm.CommunityDetection()
        comdetect.set_graph(nxg=testg)
        number_of_clusters = comdetect.detect_number_of_clusters()
        comdetect.set_number_of_clusters(number_of_clusters)

        corr_cluster_labels = comdetect.detect_corresponding_clusters(clusters_pred_labels)

        self.assertListEqual([2, 3, 1, 4, 0], corr_cluster_labels)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()