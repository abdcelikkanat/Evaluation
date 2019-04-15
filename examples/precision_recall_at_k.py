import os, sys
p = os.path.realpath(os.path.join(os.getcwd(), '..', 'edge_prediction', 'TNE'))
sys.path.insert(0, p)
sys.path.append(os.path.join(os.getcwd(), '..'))
import pickle
import pylab
from edge_prediction.edge_prediction import EdgePrediction
import matplotlib.pyplot as plt
import numpy as np

#######################################################################################################################

dataset_name = "wiki-Vote_gcc"
method_name = "louvain"
generic_name = "{}_n80_l10_w10_k80_deepwalk".format(method_name)


#######################################################################################################################
''' for 80% 
emb = os.path.realpath(os.path.join("../../edge_datasets_80/", "embeddings",
                                    dataset_name + "_residual_size_80_" + generic_name,
                                    dataset_name + "_residual_size_80_" + generic_name + "_final_max.embedding"))

size_suffix = "_size_80"
base_folder = os.path.join("..", "..", "edge_datasets_80")
split_file = os.path.join(base_folder, dataset_name + "_samples" + size_suffix + ".pkl")
graph_path = os.path.join(base_folder, dataset_name + "_residual" + size_suffix + ".gml")
'''
########################################################################################################################
''' for 50 % '''
emb = os.path.realpath(os.path.join("/media", "abdulkadir", "storage", "1 nisan", "4nisan_outputs", "outputs",
                                    "embeddings", dataset_name + "_residual_" + generic_name,
                                    dataset_name + "_residual_" + generic_name + "_final_max.embedding"))

size_suffix = ""
base_folder = os.path.join("..", "..", "datasets", "split")
split_file = os.path.join(base_folder, dataset_name + "_samples" + size_suffix + ".pkl")
graph_path = os.path.join(base_folder, dataset_name + "_residual" + size_suffix + ".gml")

########################################################################################################################

output_file = os.path.join(os.getcwd(), "./kpred_outputs")

print(os.path.realpath(emb))

ep = EdgePrediction()

samples_file_path = split_file                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
embedding_file = emb

ep.read_graph(graph_path=graph_path)
train_samples, train_labels, test_samples, test_labels = ep.read_samples(samples_file_path)


'''
g = ep.get_graph()
num_of_nodes = g.number_of_nodes()
num_of_test_nodes = 100
test_nodes = np.random.choice(a=list(g.nodes()), size=num_of_test_nodes)


test_samples = []
test_labels = []
for i in range(num_of_test_nodes):
    for j in range(i+1, num_of_test_nodes):
        candidate_edge = (test_nodes[i], test_nodes[j])
        test_samples.append(candidate_edge)
        if g.has_edge(candidate_edge[0], candidate_edge[1]):
            test_labels.append(1)
        else:
            test_labels.append(0)
'''


precisions, recalls = ep.precision_recall_at_k_yeni(embedding_file_path=embedding_file,
                                               train_samples=train_samples, train_labels=train_labels,
                                               test_samples=test_samples, test_labels=test_labels)

pylab.figure()
for op in ['hadamard', 'average', 'l1', 'l2']:
    plt.plot(range(10, len(precisions[op])), precisions[op][10:], label=op)
pylab.xlabel('k')
pylab.ylabel('precision@k')
pylab.legend(loc='upper right')
#pylab.savefig("./output_figures/" + dataset_name + "_residual_" + generic_name + "_final_max_50percent_size=predefined.pdf")
#pylab.savefig("./output_figures/graph=ca-HepTh_dw_embed_size=80_sample_size=100.pdf")
#pylab.show()

with open("./output_figures/" + dataset_name + "_residual_" + generic_name + "_final_max_50percent_size=predefined.pkl",'wb') as f:
    pickle.dump(precisions, f)