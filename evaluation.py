import os, sys
p = os.path.realpath(os.path.join(os.path.dirname(__file__), 'edge_prediction', 'TNE'))
sys.path.insert(0, p)

from classification.multi_label import NodeClassification
from community_detection.detection import CommunityDetection
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, RawTextHelpFormatter
from edge_prediction.edge_prediction import *

def parse_arguments():
    parser = ArgumentParser(description="Examples: \n Link Prediction \n  " +
                                        "python evaluation.py --method edge_prediction --option split " +
                                        "--graph networkx.gml --split_ratio 0.5 " +
                                        "--output ~/output_folder",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('--method', type=str, required=True,
                        choices=['classification', 'edge_prediction', 'community_detection'],
                        help='the name of the evaluation method')
    parser.add_argument('--emb', type=str, required=False,
                        help='the embedding file path')
    parser.add_argument('--graph', type=str, required=True,
                        help='the path of the graph, .gml or .mat files')
    parser.add_argument('--output_folder', type=str, default=None, required=False,
                        help='the path of the output folder')
    parser.add_argument('--output_file', type=str, default=None, required=False,
                        help='the path of the output file')
    parser.add_argument('--save_format', type=str, default="npy", required=False,
                        help='output file format')
    parser.add_argument('--directed', type=bool, default=False, required=False,
                        help='the number of communities')
    parser.add_argument('--train_test_ratio', type=str, default='large', required=False,
                        help='the ratios of the training set')
    parser.add_argument('--num_of_shuffles', type=int, default=25, required=False,
                        help='the number of shuffles')
    parser.add_argument('--shuffle_var', type=bool, default=False, required=False,
                        help='show the variance of the shuffles')
    parser.add_argument('--detailed', type=bool, default=False, required=False,
                        help='indicates the format of the output')
    parser.add_argument('--k', type=int, default=None, required=False,
                        help='indicates the number of communities')

    parser.add_argument('--option', type=str, required=False,
                        choices=['split', 'predict'],
                        help='indicates the optional choices')
    parser.add_argument('--split_ratio', type=float, default=0.5, required=False,
                        help='training vs test ratio')
    parser.add_argument('--split_file', type=str, default=None, required=False,
                        help='pickle file path')

    """
    parser.add_argument('--n', type=int, required=True,
                        help='the number of walks')
    parser.add_argument('--l', type=int, required=True,
                        help='the length of each walk')
    parser.add_argument('--w', type=int, default=10,
                        help='the window size')
    parser.add_argument('--d', type=int, default=128,
                        help='the size of the embedding vector')
    parser.add_argument('--k', type=int, default=80, required=True,
                        help='the number of clusters')
    parser.add_argument('--dw_alpha', type=float, default=0.0,
                        help='the parameter for Deepwalk')
    parser.add_argument('--n2v_p', type=float, default=1.0,
                        help='the parameter for node2vec')
    parser.add_argument('--n2v_q', type=float, default=1.0,
                        help='the parameter for node2vec')
    parser.add_argument('--negative', type=int, default=0,
                        help='it specifies how many noise words are used')
    parser.add_argument('--lda_alpha', type=float, default=0.625,
                        help='a hyperparameter of LDA')
    parser.add_argument('--lda_beta', type=float, default=0.1,
                        help='a hyperparameter of LDA')
    parser.add_argument('--lda_iter_num', type=int, default=2000,
                        help='the number of iterations for GibbsLDA++')
    parser.add_argument('--emb', type=str, default='all',
                        help='specifies the output embedding concatenation method')

    """
    return parser.parse_args()



def process(args):

    params = {}
    params['directed'] = args.directed


    if args.method == "classification":

        training_ratios = []
        if args.train_test_ratio == 'small':
            training_ratios = np.arange(0.01, 0.1, 0.01).tolist()
        if args.train_test_ratio == 'large':
            training_ratios = np.arange(0.1, 1, 0.1).tolist()

        if args.train_test_ratio == 'all':
            training_ratios = np.arange(0.01, 0.1, 0.01).tolist() + np.arange(0.1, 1, 0.1).tolist()

        nc = NodeClassification(embedding_file=args.emb, graph_path=args.graph, params=params)
        nc.evaluate(number_of_shuffles=args.num_of_shuffles, training_ratios=training_ratios)

        if args.output_file is None:
            nc.print_results(detailed=args.detailed, shuffle_var=args.shuffle_var)
        else:
            nc.save_results(args.output_file, shuffle_var=args.shuffle_var, detailed=args.detailed, save_format=args.save_format)

    elif args.method == "edge_prediction":

        ep = EdgePrediction()

        if args.option == 'split':

            ratio = args.split_ratio
            graph_path = args.graph
            target_folder = args.output_folder
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            ep.read_graph(graph_path=graph_path)
            print("The network has been read!")
            ep.split_network(train_test_ratio=ratio, target_folder=target_folder)
            print("The network has been partitioned!")

        elif args.option == 'predict':

            samples_file_path = args.split_file
            embedding_file = args.emb
            graph_path = args.graph

            ep.read_graph(graph_path=graph_path)
            train_samples, train_labels, test_samples, test_labels = ep.read_samples(samples_file_path)
            scores = ep.predict(embedding_file_path=embedding_file,
                                train_samples=train_samples, train_labels=train_labels,
                                test_samples=test_samples, test_labels=test_labels)

            if args.output_file is not None:
                with open(args.output_file, "wb") as fp:
                    pickle.dump(scores, fp)
            else:
                print(scores)

        else:
            raise ValueError("Not implemented!")

    if args.method == "community_detection":

        params = {}
        params['directed'] = args.directed

        commdetect = CommunityDetection(embedding_file=args.emb, graph_path=args.graph, params=params)
        score = commdetect.evaluate(num_of_communities=args.k)

        print("Score: {}".format(score))


if __name__ == "__main__":
    args = parse_arguments()

    process(args)
