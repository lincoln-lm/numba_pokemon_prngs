"""Package for numba @jitclass implementations
of the Pseudo-Random Number Generators used in the Pokemon series"""
# pylint: disable=wrong-import-position,wrong-import-order
from .patch import patch_numba_8e2063b

patch_numba_8e2063b()

import numpy as np

np.seterr(over="ignore")
del np

# pylint: disable=wrong-import-position,wrong-import-order
