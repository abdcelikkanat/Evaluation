import numpy as np
import pickle


# facebook_combined_deepwalk

generic_name = "facebook_combined_gcc" #"p2p-Gnutella08_gcc" #"facebook_combined"
method_name = "deepwalk"

for var in ['node', 'final']:
    print("--------< {} >----------".format(var))
    file_path = "./scores/" + generic_name + "/" + generic_name + "_" + method_name + "_{}.score".format(var)

    pf = pickle.load(open(file_path, 'rb'))
    total_scores = pf[0]

    for metric in total_scores:
        print("{}: {}".format(metric, total_scores[metric]))
