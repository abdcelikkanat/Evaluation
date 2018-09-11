from tne.tne import TNE
from utils.utils import find_topics_for_nodes, concatenate_embeddings
from os.path import basename, splitext, join
import os


def read_embedding_file(file_path):
    with open(file_path, 'r') as f:
        f.readline()  # Skip the first line
        embeddings = {}
        for line in f:
            tokens = line.strip().split()
            embeddings.update({str(tokens[0]): [float(val) for val in tokens[1:]]})

    return embeddings

def learn_tne_embeddings(nxg, base_folder, params):

    tne = TNE(params=params)
    tne.set_graph(nxg)

    temp_folder = os.path.join("./temp", base_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    outputs_folder = os.path.join("./outputs", base_folder)
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)

    base_desc = "{}_n{}_l{}_w{}_k{}_{}".format(params['graph_name'],
                                               params['number_of_walks'],
                                               params['walk_length'],
                                               params['window_size'],
                                               params['number_of_topics'],
                                               params['method'])

    # node corpus1
    corpus_path_for_node = join(temp_folder, "{}_node_corpus.corpus".format(base_desc))
    # lda corpus
    corpus_path_for_lda = join(temp_folder, "{}_lda_corpus.corpus".format(base_desc))
    # embedding files
    node_embedding_file = join(outputs_folder, "{}_node.embedding".format(base_desc))
    topic_embedding_file = join(outputs_folder, "{}_topic.embedding".format(base_desc))
    concatenated_embedding_file_max = join(outputs_folder, "{}_final_max.embedding".format(base_desc))

    tne.perform_random_walks(node_corpus_path=corpus_path_for_node)
    tne.save_corpus(corpus_path_for_lda, with_title=True)
    id2node = tne.run_lda(lda_corpus_path=corpus_path_for_lda)
    tne.learn_node_embedding(node_corpus_path=corpus_path_for_node, node_embedding_file=node_embedding_file)
    tne.learn_topic_embedding(node_corpus_path=corpus_path_for_node, topic_embedding_file=topic_embedding_file)

    # Concatenate the embeddings
    phi_file = tne.get_file_path(filename='phi')
    node2topic_max = find_topics_for_nodes(phi_file, id2node, params['number_of_topics'], type="max")
    concatenate_embeddings(node_embedding_file=node_embedding_file,
                           topic_embedding_file=topic_embedding_file,
                           node2topic=node2topic_max,
                           output_filename=concatenated_embedding_file_max)

    return read_embedding_file(node_embedding_file), read_embedding_file(concatenated_embedding_file_max)


def learn_embedings(algorithm, nxg, base_folder, params):

    if algorithm == 'tne':
        return learn_tne_embeddings(nxg, base_folder, params)
    if algorithm == 'node2vec':
        pass