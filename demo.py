import warnings

import numpy as np

from thurstat import *

warnings.filterwarnings("ignore")

@formula
def exp(x):
    return np.exp(x)

X = (
    ContinuousDistribution
    .from_pfunc("cdf", -1 / (np.e ** x + 1) + 1, a=-np.inf, b=np.inf)
    .apply_func(1 / (1 + exp(-x)))
)
print(X.support)
print(X.mean)
X.display("pdf")