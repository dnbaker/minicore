import numpy as np


def idnc(e, v=None):
    if isinstance(v, int):
        v = np.arange(0, v).astype(np.uint32)
    sp = np.random.choice(range(len(v)), replace=False)
