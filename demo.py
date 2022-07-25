import warnings

import numpy as np

from thurstat import *

warnings.filterwarnings("ignore")

U = UniformDiscreteDistribution(a=0, b=3)
V = UniformDiscreteDistribution(a=2, b=5)
W = 3 - U * V
print(W.mean)
W.display("pmf")

@formula
def exp(x):
    return np.exp(x)

x = FormulaVariable()
X = (
    ContinuousDistribution
    .from_pfunc("cdf", -1 / (exp(x) + 1) + 1, a=-np.inf, b=np.inf)
    .apply_func(1 / (1 + exp(-x)))
)
print(X.support)
print(X.mean)
X.display("pdf")