import random

import numpy as np
import torch as th


def set_seed(seed: int = 42):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    return None
