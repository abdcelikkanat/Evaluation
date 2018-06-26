import numpy as np
from classification.multi_label import *




graph_path = "./inputs/synthetic_n1000_p0.8_q0.2_sizes500:500_n80_l10_w10_k80_deepwalk/"
graph_path += "synthetic_n1000_p0.8_q0.2_sizes500:500.gml"

params = {'directed': False}

number_of_shuffles = 25

embedding_file = "./inputs/synthetic_n1000_p0.8_q0.2_sizes500:500_n80_l10_w10_k80_deepwalk/"
embedding_file += "synthetic_n1000_p0.8_q0.2_sizes500:500_n80_l10_w10_k80_deepwalk_final_max.embedding"

nc_final = NodeClassification(embedding_file, graph_path, params=params)
nc_final.evaluate(number_of_shuffles=number_of_shuffles, training_ratios=np.arange(0.1, 1.0, 0.1))
nc_final.print_results(detailed=False)

print("---------------------------")

embedding_file = "./inputs/synthetic_n1000_p0.8_q0.2_sizes500:500_n80_l10_w10_k80_deepwalk/"
embedding_file += "synthetic_n1000_p0.8_q0.2_sizes500:500_n80_l10_w10_k80_deepwalk_node.embedding"

nc_node = NodeClassification(embedding_file, graph_path, params=params)
nc_node.evaluate(number_of_shuffles=number_of_shuffles, training_ratios=np.arange(0.1, 1.0, 0.1))
nc_node.print_results(detailed=False)