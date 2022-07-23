import warnings

import numpy as np

from thurstat import *

warnings.filterwarnings("ignore")

X = (
    ContinuousDistribution
    .from_pfunc("cdf", lambda x: -1 / (np.exp(x) + 1) + 1, a=-np.inf, b=np.inf)
    .apply_function(lambda x: 1 / (1 + np.exp(-x)))
)
print(X.support)
print(X.mean)
X.display("pdf")