import constants as c
import os
import time
import numpy as np
import networkx as nx
from community_detection.detection import *
from graphbase.graphbase import *
#from hmmlearn import hmm


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

    def get_community_assignments_by(self, method=None, temp_dfile_file="gibbsldapp.dfile", params={}):

        if method == "HMM":
            """
            model = hmm.MultinomialHMM(n_components=3)
            model.startprob_ = np.array([0.6, 0.3, 0.1])
            model.transmat_ = np.array([[0.7, 0.2, 0.1],
                                             [0.3, 0.5, 0.2],
                                             [0.3, 0.3, 0.4]])
            model.emissionprob_ = np.array([[0.4, 0.2, 0.1, 0.3],
                                        [0.3, 0.4, 0.1, 0.2],
                                        [0.1, 0.3, 0.5, 0.1]])

            X, Z = model.sample(1000)

            print(np.asarray(X).T)
            print(Z)
            """

            """
            remodel = hmm.MultinomialHMM(n_components=3, n_iter=100)
            remodel.fit(X)
            Z2 = remodel.predict(X)
            print(Z2)
            """

            """
            seqs = []
            lens = []
            for walk in self._walks:
                s = [[int(w)-1] for w in walk]
                seqs.extend(s)
                lens.append(len(s))

            model = hmm.MultinomialHMM(n_components=params['number_of_topics'], tol=0.001, n_iter=5000)
            model.fit(seqs, lens)

            posteriors = model.predict_proba(np.asarray([[i] for i in range(self.g.number_of_nodes())]))
            comms = np.argmax(posteriors, 1)

            node2comm = {}
            for id in range(len(comms)):
                node2comm[str(id+1)] = comms[id]

            return node2comm
            """
            seqs = []
            lens = []
            for walk in self._walks:
                s = [int(w) - 1 for w in walk]
                seqs.append(s)
                lens.append(len(s))

            pipi = np.asarray([0.5, 0.5], dtype=np.float)
            AA = np.asarray([[0.2, 0.8], [0.5, 0.5]], dtype=np.float)
            OO = np.asarray([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]], dtype=np.float)

            seqs = []
            for i in range(31):
                seq = []

                s = np.random.choice(range(2), p=pipi)
                o = np.random.choice(range(3), p=OO[s, :])
                seq.append(o)
                for _ in range(59):
                    s = np.random.choice(range(2), p=AA[s, :])
                    o = np.random.choice(range(3), p=OO[s, :])
                    seq.append(o)

                seqs.append(seq)

            seqs = np.vstack(seqs)

            #print(seqs)

            from bayespy.nodes import Categorical, Mixture
            from bayespy.nodes import CategoricalMarkovChain
            from bayespy.nodes import Dirichlet
            from bayespy.inference import VB
            K = params['number_of_topics'] # the number of hidden states
            N = self.g.number_of_nodes()  # the number of observations

            #p0 = np.ones(K) / K

            D = 31 #len(lens)
            states = 60

            a0 = Dirichlet(1e+1 * np.ones(K), plates=())
            A = Dirichlet(1e+1 * np.ones(K), plates=(2, ), name='A')
            P = Dirichlet(1e+1 * np.ones((K, N)))
            Z = CategoricalMarkovChain(a0, A, states=states, plates=(D,))
            Y = Mixture(Z, Categorical, P)

            Y.observe(seqs)

            #a0.random()
            #A.random()
            #P.random()

            Ainit = np.random.random((2,2))
            Ainit = np.divide(Ainit.T, np.sum(Ainit, 1)).T

            #A.initialize_from_value(Ainit)
            #print(Ainit)
            Q = VB(Y, Z, P, A, a0)

            Q.update(repeat=1000, plot=False, verbose=True)

            #print(Z.random())
            print(Q['A'])


            return {}

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

            node2comm = {}
            for nodeId in id2node:
                node2comm[id2node[nodeId]] = max_topics[int(nodeId)]

            return node2comm


