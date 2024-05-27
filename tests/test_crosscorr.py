from pathlib import Path

import numpy as np
import pandas as pd

from raccoon import crosscorr
from raccoon import ccflibfort


# Test
def test_get_obj_info():
    obj = crosscorr.get_obj_info()
    assert obj == 'obj'

# Test funciton runs
def test_ccflibfort_ccfcompute():
    w = np.linspace(5000, 5100, 100)
    f = np.random.rand(len(w))
    c = np.ones_like(w)
    wm = np.linspace(5003, 5075, 25)
    fm = np.random.rand(len(wm))
    rv = np.arange(-100, 100, 10)
    ccf = ccflibfort.computeccf(w, f, c, wm, fm, rv)
    assert len(rv) == len(ccf)