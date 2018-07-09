import sys
sys.path.append("../../deepwalk/deepwalk")
from community_detection.detection import CommunityDetection
import numpy as np


params = {'directed': False}
kmeans_num_of_communities = 3
n=1200
tau1 = 2.2
tau2 = 2.3
mindeg = 1
mincomm = 20
k = 80
sizes = ":".join(str(v) for v in np.ones(kmeans_num_of_communities, dtype=np.int)*400)




#basename = "synthetic_n1000"
basename = "lfr_synthetic_n{}_tau1{}_tau2{}_mindeg{}_mincom{}".format(n, tau1, tau2, mindeg, mincomm)
graph_path = "./inputs/"+basename+"_n80_l10_w10_k{}_deepwalk/".format(k)
graph_path += basename+".gml"


embedding_file = "./inputs/"+basename+"_n80_l10_w10_k{}_deepwalk/".format(k)
final_embedding_file = embedding_file + basename + "_n80_l10_w10_k{}_deepwalk_final_new.embedding".format(k)
node_embedding_file = embedding_file + basename + "_n80_l10_w10_k{}_deepwalk_node.embedding".format(k)

final_comdetect = CommunityDetection(final_embedding_file, graph_path, params=params)
nmi_score, ccr_score, vi_score, ari_score = final_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Final Scores NMI:{} CCR:{} VI:{} ARI:{}".format(nmi_score, ccr_score, vi_score, ari_score))


node_comdetect = CommunityDetection(node_embedding_file, graph_path, params=params)
nmi_score, ccr_score, vi_score, ari_score = node_comdetect.evaluate(num_of_communities=kmeans_num_of_communities)
print("Node Scores NMI:{} CCR:{} VI:{} ARI:{}".format(nmi_score, ccr_score, vi_score, ari_score))