import gensim
import scipy as sp
import scipy.io as sio
import numpy as np


embedding_file = ""
file_type = ""
label_file = ""
mat_label_name = ""


number_of_instances = 0
number_of_classes = 0


# Load the embedding vectors
vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=embedding_file, binary=False)

# Load the node labels
if file_type is "mat":
    mat = sio.loadmat(label_file)
    L = mat[mat_label_name]
    number_of_classes = L.shape[1]

# 2. Load labels
mat = loadmat(matfile)
A = mat[args.adj_matrix_name]
graph = sparse2graph(A)
labels_matrix = mat[args.label_matrix_name]
labels_count = labels_matrix.shape[1]
mlb = MultiLabelBinarizer(range(labels_count))

# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
features_matrix = numpy.asarray([model[str(node)] for node in range(len(graph))])

# 2. Shuffle, to create train/test groups
shuffles = []
for x in range(args.num_shuffles):
    shuffles.append(skshuffle(features_matrix, labels_matrix))