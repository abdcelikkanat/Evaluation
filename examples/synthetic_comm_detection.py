import sys
sys.path.append("../../deepwalk/deepwalk")
from community_detection.detection import CommunityDetection
import numpy as np


params = {'directed': False}
kmeans_num_of_communities = 8
p = 0.3
q = 0.2
k = 128
sizes = ":".join(str(v) for v in np.ones(kmeans_num_of_communities, dtype=np.int)*150)




#basename = "synthetic_n1000"
basename = "sbm_synthetic_k{}".format(kmeans_num_of_communities)
graph_path = "./inputs/"+basename+"_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk/".format(k)
graph_path += basename+"_p{}_q{}_sizes".format(p, q)+sizes+".gml"


embedding_file = "./inputs/"+basename+"_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk/".format(k)
final_embedding_file = embedding_file + basename + "_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk_final_new.embedding".format(k)
node_embedding_file = embedding_file + basename + "_p{}_q{}_sizes".format(p, q)+sizes+"_n80_l10_w10_k{}_deepwalk_node.embedding".format(k)

final_comdetect = CommunityDetection(final_embedding_file, graph_path, params=params)
nmi_score, ccr_score, vi_score, ari_score = final_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Final Scores NMI:{} CCR:{} VI:{} ARI:{}".format(nmi_score, ccr_score, vi_score, ari_score))


node_comdetect = CommunityDetection(node_embedding_file, graph_path, params=params)
nmi_score, ccr_score, vi_score, ari_score = node_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Node Scores NMI:{} CCR:{} VI:{} ARI:{}".format(nmi_score, ccr_score, vi_score, ari_score))