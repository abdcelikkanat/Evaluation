import networkx as nx
import numpy as np
from utils import get_file_extension
from classification.multi_label import NodeClassification
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_arguments():
    parser = ArgumentParser(description="Evaluate the methods",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', type=str, required=True,
                        choices=['classification', 'edge-prediction', 'community-detection'],
                        help='the name of the evaluation method')
    parser.add_argument('--emb', type=str, required=False,
                        help='the embedding file path')
    parser.add_argument('--graph', type=str, required=True,
                        help='the path of the graph, .gml or .mat files')
    parser.add_argument('--output', type=str, default=None, required=False,
                        help='the path of the output folder')
    parser.add_argument('--directed', type=bool, default=False, required=False,
                        help='the number of clusters')
    parser.add_argument('--ratios', type=str, default='some', required=False,
                        help='the ratios of the training set')
    parser.add_argument('--num_shuffles', type=int, default=25, required=False,
                        help='the number of shuffles')
    parser.add_argument('--detailed', type=bool, default=False, required=False,
                        help='indicates the format of the output')

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

    if args.method == "classification":

        params = {}
        params['directed'] = args.directed

        if args.ratios == 'some':
            training_ratios = np.arange(1, 10) * 0.1

        nc = NodeClassification(embedding_file=args.emb, graph_path=args.graph, params=params)
        nc.evaluate(number_of_shuffles=args.num_shuffles, training_ratios=training_ratios)

        if args.output is None:
            nc.print_results(detailed=args.detailed)
        else:
            nc.save_results(args.output, detailed=args.detailed)


if __name__ == "__main__":
    args = parse_arguments()

    process(args)
