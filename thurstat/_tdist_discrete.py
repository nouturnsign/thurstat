from typing import Dict, Union

import scipy.stats
from scipy.stats._distn_infrastructure import rv_frozen

from ._tdist_base import *

__all__ = [
    "BinomialDistribution", "BernoulliDistribution", "BetaBinomialDistribution", "BinomialDistribution",
    "GeometricDistribution", "HypergeometricDistribution", "NegativeBinomialDistribution",
    "NegativeHypergeometricDistribution", "PoissonDistribution", "SkellamDistribution",
    "UniformDiscreteDistribution", "YuleSimonDistribution", "ZipfDistribution", "ZipfianDistribution",
]

class BernoulliDistribution(DiscreteDistribution):
    """A Bernoulli distribution."""
    
    options = [
        ["p"],
        ["q"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "p" in parameters:
            p = parameters.pop("p")
        elif "q" in parameters:
            p = 1 - parameters.pop("q")
        self.p = p
        return scipy.stats.bernoulli(p)
    
    def __add__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, BernoulliDistribution) and self.p == other.p:
            return BinomialDistribution(n=2, p=self.p)
        if isinstance(other, BinomialDistribution) and self.p == other.p:
            return BinomialDistribution(n=1 + other.n, p=self.p)
        return super().__add__(other)

class BetaBinomialDistribution(DiscreteDistribution):
    """A beta-binomial distribution."""
    
    options = [
        ["n", "a", "b"],
        ["n", "alpha", "beta"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "n" in parameters:
            n = parameters.pop("n")
            if "a" in parameters and "b" in parameters:
                a = parameters.pop("a")
                b = parameters.pop("b")
            elif "alpha" in parameters and "beta" in parameters:
                a = parameters.pop("alpha")
                b = parameters.pop("beta")
        return scipy.stats.betabinom(n, a, b)

class BinomialDistribution(DiscreteDistribution):
    """A binomial distribution."""
    
    options = [
        ["n", "p"],
        ["n", "q"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "n" in parameters:
            n = parameters.pop("n")
            if "p" in parameters:
                p = parameters.pop("p")
            elif "q" in parameters:
                p = 1 - parameters.pop("q")
        self.n = n
        self.p = p
        return scipy.stats.binom(n, p)
    
    def __add__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, BernoulliDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n + 1, p=self.p)
        if isinstance(other, BinomialDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n + other.n, p=self.p)
        return super().__add__(other)
    
    def __sub__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, BernoulliDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n - 1, p=self.p)
        if isinstance(other, BinomialDistribution) and self.p == other.p:
            n = self.n - other.n
            if n == 1:
                return BernoulliDistribution(p=self.p)
            elif n > 1:
                return BinomialDistribution(n=n, p=self.p)
        return super().__sub__(other)

class GeometricDistribution(DiscreteDistribution):
    """A geometric distribution."""
    
    options = [
        ["p"],
        ["q"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "p" in parameters:
            p = parameters.pop("p")
        elif "q" in parameters:
            p = 1 - parameters.pop("q")
        self. p =p
        return scipy.stats.geom(p)
    
    def __add__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, GeometricDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=2, p=self.p)
        if isinstance(other, NegativeBinomialDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=1 + other.n, p=self.p)
        return super().__add__(other)

class HypergeometricDistribution(DiscreteDistribution):
    """A hypergeometric distribution."""
    
    options = [
        ["M", "n", "N"],
        ["N", "K", "n"],
        ["N", "m", "n"],
        ["N", "N1", "n"],
        ["N1", "N2", "n"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "M" in parameters and "n" in parameters and "N" in parameters:
            M = parameters.pop("M")
            n = parameters.pop("n")
            N = parameters.pop("N")
        elif "n" in parameters:
            N = parameters.pop("n")
            if "N1" in parameters:
                n = parameters.pop("N1")
                if "N" in parameters:
                    M = parameters.pop("N")
                elif "N2" in parameters:
                    M = n + parameters.pop("N2")
            elif "N" in parameters:
                M = parameters.pop("N")
                if "K" in parameters:
                    n = parameters.pop("K")
                elif "n" in parameters:
                    n = parameters.pop("m")
        return scipy.stats.hypergeom(M, n, N)

class NegativeBinomialDistribution(DiscreteDistribution):
    """A negative binomial distribution."""
    
    options = [
        ["n", "p"],
        ["n", "q"],
        ["r", "p"],
        ["r", "q"],
        ["mu", "sigma"],
        ["mean", "variance"],
        ["mu", "n"],
        ["mu", "r"],
        ["mu", "alpha"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "mean" in parameters and "variance" in parameters:
            mean = parameters.pop("mean")
            variance = parameters.pop("variance")
            p = mean / variance
            n = mean ** 2 / (variance - mean)
        elif "mu" in parameters:
            mu = parameters.pop("mu")
            if "n" in parameters:
                n = parameters.pop("n")
                p = n / (n + mu)
            elif "sigma" in parameters:
                sigma = parameters.pop("sigma")
                p = mu / sigma ** 2
                n = mu ** 2 / (sigma ** 2 - mu)
            elif "r" in parameters:
                n = parameters.pop("r")
                p = n / (mu + n)
            elif "alpha" in parameters:
                n = int(1 / parameters.pop("alpha"))
                p = n / (mu + n)
        elif "n" in parameters:
            n = parameters.pop("n")
            if "p" in parameters:
                p = parameters.pop("p")
            elif "q" in parameters:
                p = 1 - parameters.pop("q")
        elif "r" in parameters:
            n = parameters.pop("r")
            if "p" in parameters:
                p = parameters.pop("p")
            elif "q" in parameters:
                p = 1 - parameters.pop("q")
        self.n = n
        self.p = p
        return scipy.stats.nbinom(n, p)
    
    def __add__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, GeometricDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=self.n + 1, p=self.p)
        if isinstance(other, NegativeBinomialDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=self.n + other.n, p=self.p)
        return super().__add__(other)
    
    def __sub__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, GeometricDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=self.n - 1, p=self.p)
        if isinstance(other, NegativeBinomialDistribution) and self.p == other.p:
            return NegativeBinomialDistribution(n=self.n - other.n, p=self.p)
        return super().__sub__(other)

class NegativeHypergeometricDistribution(DiscreteDistribution):
    """A negative hypergeometric distribution."""
    
    options = [
        ["M", "n", "r"],
        ["N", "K", "r"],
        ["N", "m", "r"],
        ["N", "N1", "r"],
        ["N1", "N2", "r"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "r" in parameters:
            r = parameters.pop("r")
            if "M" in parameters and "n" in parameters:
                M = parameters.pop("M")
                n = parameters.pop("n")
            elif "N1" in parameters:
                n = parameters.pop("N1")
                if "N" in parameters:
                    M = parameters.pop("N")
                elif "N2" in parameters:
                    M = n + parameters.pop("N2")
            elif "N" in parameters:
                M = parameters.pop("N")
                if "K" in parameters:
                    n = parameters.pop("K")
                elif "n" in parameters:
                    n = parameters.pop("m")
        return scipy.stats.nhypergeom(M, n, r)

class PoissonDistribution(DiscreteDistribution):
    """A Poisson distribution."""
    
    options = [
        ["mu"],
        ["lambda_"],
        ["r", "t"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "mu" in parameters:
            mu = parameters.pop("mu")
        elif "lambda_" in parameters:
            mu = parameters.pop("lambda_")
        elif "r" in parameters and "t" in parameters:
            mu = parameters.pop("r") * parameters.pop("t")
        self.mu = mu
        return scipy.stats.poisson(mu)
    
    def __add__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, PoissonDistribution):
            return PoissonDistribution(mu=self.mu + other.mu)
        return super().__add__(other)
    
    def __sub__(self, other: Union[float, DiscreteDistribution]) -> DiscreteDistribution:
        if isinstance(other, PoissonDistribution):
            return SkellamDistribution(mu1=self.mu, mu2=other.mu)
        return super().__sub__(other)

class SkellamDistribution(DiscreteDistribution):
    """A Skellam distribution."""
    
    options = [
        ["mu1", "mu2"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "mu1" in parameters and "mu2" in parameters:
            mu1 = parameters.pop("mu1")
            mu2 = parameters.pop("mu2")
        return scipy.stats.skellam(mu1, mu2)

class UniformDiscreteDistribution(DiscreteDistribution):
    """A random integer or uniform discrete distribution."""
    
    options = [
        ["a", "b"],
        ["low", "high"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters and "b" in parameters:
            low = parameters.pop("a")
            high = parameters.pop("b") + 1
        elif "low" in parameters and "high" in parameters:
            low = parameters.pop("low")
            high = parameters.pop("high")
        return scipy.stats.randint(low, high)

class YuleSimonDistribution(DiscreteDistribution):
    """A Yule-Simon distribution."""
    
    options = [
        ["alpha"],
        ["rho"],
        ["a"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "alpha" in parameters:
            alpha = parameters.pop("alpha")
        elif "rho" in parameters:
            alpha = parameters.pop("rho")
        elif "a" in parameters:
            alpha = parameters.pop("a") - 1
        return scipy.stats.yulesimon(alpha)

class ZipfDistribution(DiscreteDistribution):
    """A Zipf or zeta distribution."""
    
    options = [
        ["a"],
        ["s"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters:
            a = parameters.pop("a")
        elif "s" in parameters:
            a = parameters.pop("s")
        return scipy.stats.zipf(a)

class ZipfianDistribution(DiscreteDistribution):
    """A Zipfian distribution."""
    
    options = [
        ["a", "n"],
        ["s", "N"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters and "n" in parameters:
            a = parameters.pop("a")
            n = parameters.pop("n")
        elif "s" in parameters:
            a = parameters.pop("s")
            n = parameters.pop("N")
        return scipy.stats.zipfian(a, n)