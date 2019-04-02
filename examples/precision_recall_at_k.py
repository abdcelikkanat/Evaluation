import os, sys
p = os.path.realpath(os.path.join(os.getcwd(), '..', 'edge_prediction', 'TNE'))
sys.path.insert(0, p)
sys.path.append(os.path.join(os.getcwd(), '..'))
import pickle
from edge_prediction.edge_prediction import EdgePrediction


emb = os.path.join(os.getcwd(), "..", "edge_prediction", "embeddings", "facebook_combined_gcc_lda_n8_l10_w10_k2_deepwalk",
                   "facebook_combined_gcc_lda_n8_l10_w10_k2_deepwalk_node.embedding")
split_file = os.path.join("..", "..", "datasets", "split", "facebook_combined_gcc_samples.pkl")
graph_path = os.path.join("..", "..", "datasets", "facebook_combined_gcc.gml")
output_file = os.path.join(os.getcwd(), "./kpred_outputs")

print(os.path.realpath(emb))

ep = EdgePrediction()

samples_file_path = split_file
embedding_file = emb

ep.read_graph(graph_path=graph_path)
train_samples, train_labels, test_samples, test_labels = ep.read_samples(samples_file_path)
precisions, recalls = ep.precision_recall_at_k(embedding_file_path=embedding_file,
                                               train_samples=train_samples, train_labels=train_labels,
                                               test_samples=test_samples, test_labels=test_labels)

with open("prec_recall.pkl", 'wb') as f:
    pickle.dump((precisions, recalls), f)