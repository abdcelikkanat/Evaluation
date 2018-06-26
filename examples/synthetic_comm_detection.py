import sys
sys.path.append("../../deepwalk/deepwalk")
from community_detection.detection import CommunityDetection
import numpy as np


p = 0.7
q = 0.3
k = 80
sizes = ":".join(str(v) for v in np.ones(5, dtype=np.int)*600)


params = {'directed': False}
kmeans_num_of_communities = 5

#basename = "synthetic_n1000"
basename = "synthetic_k5"
graph_path = "./inputs/"+basename+"_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk/".format(k)
graph_path += basename+"_p{}_q{}_sizes".format(p, q)+sizes+".gml"


embedding_file = "./inputs/"+basename+"_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk/".format(k)
final_embedding_file = embedding_file + basename + "_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk_final_new.embedding".format(k)
node_embedding_file = embedding_file + basename + "_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk_node.embedding".format(k)

final_comdetect = CommunityDetection(final_embedding_file, graph_path, params=params)
score = final_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Final Score: {}".format(score))


node_comdetect = CommunityDetection(node_embedding_file, graph_path, params=params)
score = node_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Node Score: {}".format(score))