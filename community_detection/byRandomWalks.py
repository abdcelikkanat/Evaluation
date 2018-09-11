import constants as c
import os
import time
import numpy as np
import networkx as nx
from detection import *
from graphbase.graphbase import *


class ByRandomWalks(GraphBase):
    def __init__(self):
        GraphBase.__init__(self)
        self._walks = []

    def set_walks(self, walks):
        self._walks = walks

    def read_walks_file(self, file_path):
        walks = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                walk = line.strip().split()
                walks.append(walk)

        self._walks = walks

    def get_community_assignments(self, method=None, temp_dfile_file="gibbsldapp.dfile", params={}):

        if method == "LDA":

            # Run GibbsLDA++

            lda_exe_path = c._GIBBSLDA_PATH

            if not os.path.exists(lda_exe_path):
                raise ValueError("Invalid path of GibbsLDA++!")


            temp_lda_folder = "./temp"
            if not os.path.exists(temp_lda_folder):
                os.makedirs(temp_lda_folder)

            temp_dfile_path = os.path.join(temp_lda_folder, temp_dfile_file)

            if not os.path.exists(temp_dfile_path):
                # Save the walks into the dfile
                n = len(self._walks)
                with open(temp_dfile_path, 'w') as f:
                    f.write("{}\n".format(n))
                    for walk in self._walks:
                        f.write("{}\n".format(" ".join(str(w) for w in walk)))

            initial_time = time.time()

            cmd = "{} -est ".format(lda_exe_path)
            cmd += "-alpha {} ".format(params['lda_alpha'])
            cmd += "-beta {} ".format(params['lda_beta'])
            cmd += "-ntopics {} ".format(params['number_of_topics'])
            cmd += "-niters {} ".format(params['lda_number_of_iters'])
            cmd += "-savestep {} ".format(params['lda_number_of_iters'] + 1)
            cmd += "-dfile {} ".format(temp_dfile_path)
            os.system(cmd)

            print("-> The LDA algorithm run in {:.2f} secs".format(time.time() - initial_time))

            # Read wordmap file
            id2node = {}
            temp_wordmap_path = os.path.join(temp_lda_folder, "wordmap.txt")
            with open(temp_wordmap_path, 'r') as f:
                f.readline()  # skip the first line
                for line in f.readlines():
                    tokens = line.strip().split()
                    id2node[int(tokens[1])] = tokens[0]

            # Read phi file
            phi = np.zeros(shape=(params['number_of_topics'], len(id2node)), dtype=np.float)
            temp_phi_path = os.path.join(temp_lda_folder, "model-final.phi")
            with open(temp_phi_path, 'r') as f:
                for topicId, line in enumerate(f.readlines()):
                    phi[topicId, :] = [float(value) for value in line.strip().split()]

            max_topics = np.argmax(phi, axis=0)
            print(max_topics)
            node2comm = {}
            for nodeId in id2node:
                node2comm[id2node[nodeId]] = max_topics[int(nodeId)-1]

            return node2comm


params = {'lda_alpha': 0.5, 'lda_beta': 0.8, 'number_of_topics': 2, 'lda_number_of_iters': 1000}

brw = ByRandomWalks()
brw.read_walks_file("./karate.walks")
node2comm = brw.get_community_assignments(method="LDA", params=params)

print(node2comm)