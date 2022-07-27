import numpy as np

from thurstat import *

update_defaults(warnings="ignore")

N = BinomialDistribution(n=10, p=0.2)
print(P(N == 3))
print(P(N != 3))
N.display("pmf")

O = UniformContinuousDistribution(a=0, b=1)
print(P(0.25 < O))
print(P(O > 0.5))
print(P(0.25 < O < 0.5))

Q = UniformContinuousDistribution(a=0, b=20)
Q.discretize().display("pmf")

R = UniformContinuousDistribution(a=0, b=5)
S = UniformContinuousDistribution(a=2, b=6)
T = R + S
print(T.variance)
T.display("pdf")

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