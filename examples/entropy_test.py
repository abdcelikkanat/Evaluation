from community_detection.detection import CommunityDetection
import scipy
import numpy as np

comdetect = CommunityDetection(embedding_file="", graph_path="", params={'directed': False})

labels = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
print(np.bincount(labels).astype(np.float))
s = comdetect._compute_entropy(labels=labels)
print(s)

t = scipy.stats.entropy(pk=np.bincount(labels).astype(np.float))
print(t)