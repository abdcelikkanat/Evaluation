import os
from os.path import dirname


_BASE_FOLDER = dirname(os.path.abspath(__file__))
_EXTERNAL_FOLDER = "external"
_GIBBSLDA_PATH = os.path.join(_BASE_FOLDER, _EXTERNAL_FOLDER, "GibbsLDA++-0.2", "lda")