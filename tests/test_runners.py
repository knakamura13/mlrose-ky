"""Unit tests for runners/"""

# Author: Kyle Nakamura
# License: BSD 3 clause

import numpy as np

try:
    import mlrose_hiive
except ImportError:
    import sys

    sys.path.append("..")
