import os
from edge_prediction.TNE.tne.tne import *
from os.path import basename, splitext, join
import time


def read_embedding_file(file_path):
    with open(file_path, 'r') as f:
        f.readline()  # Skip the first line
        embeddings = {}
        for line in f:
            tokens = line.strip().split()
            embeddings.update({str(tokens[0]): [float(val) for val in tokens[1:]]})

    return embeddings


def learn_embeddings(nx_graph_path, params, file_desc, temp_folder, embedding_folder):

    # Determine the paths of embedding files
    node_embedding_file = join(embedding_folder, "{}_node.embedding".format(file_desc))
    community_embedding_file = join(embedding_folder, "{}_community.embedding".format(file_desc))

    concatenated_embedding_file = dict()
    concatenated_embedding_file['max'] = join(embedding_folder, "{}_final_max.embedding".format(file_desc))
    concatenated_embedding_file['wmean'] = join(embedding_folder, "{}_final_wmean.embedding".format(file_desc))
    concatenated_embedding_file['min'] = join(embedding_folder, "{}_final_min.embedding".format(file_desc))

    # The path for the corpus
    corpus_path_for_node = join(temp_folder, "{}_node_corpus.corpus".format(file_desc))

    tne = TNE(nx_graph_path, temp_folder, params)
    tne.perform_random_walks(output_node_corpus_file=corpus_path_for_node)
    tne.preprocess_corpus(process="equalize")
    phi = tne.generate_community_corpus(method=params['comm_detection_method'])

    tne.learn_node_embedding(output_node_embedding_file=node_embedding_file)
    tne.learn_topic_embedding(output_topic_embedding_file=community_embedding_file)

    # Compute the corresponding topics for each node
    for embedding_strategy in ["max", "min", "wmean"]:
        initial_time = time.time()
        # Concatenate the embeddings
        concatenate_embeddings(node_embedding_file=node_embedding_file,
                               topic_embedding_file=community_embedding_file,
                               phi=phi,
                               method=embedding_strategy,
                               output_filename=concatenated_embedding_file[embedding_strategy])
        print("-> The {} embeddings were generated and saved in {:.2f} secs | {}".format(
            embedding_strategy, (time.time() - initial_time), concatenated_embedding_file[embedding_strategy]))

    return read_embedding_file(node_embedding_file), read_embedding_file(concatenated_embedding_file["max"])


def learn_embedings(algorithm, nxg, base_folder, params):

    if algorithm == 'tne':
        return learn_tne_embeddings(nxg, base_folder, params)
    if algorithm == 'node2vec':
        pass