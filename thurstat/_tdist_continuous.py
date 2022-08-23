from typing import Dict, Union

import numpy as np
import scipy.stats
from scipy.stats._distn_infrastructure import rv_frozen

from ._tdist_base import *

__all__ = [
    "BetaDistribution", "BetaPrimeDistribution", "CauchyDistribution", "ChiDistribution", 
    "ChiSquaredDistribution", "CosineDistribution", "ErlangDistribution", "ExponentialDistribution", 
    "FDistribution", "GammaDistribution", "GompertzDistribution", "LaplaceDistribution", 
    "LogisticDistribution", "NormalDistribution", "TDistribution", "TrapezoidalDistribution",
    "TriangularDistribution", "UniformContinuousDistribution", "WeibullDistribution",
]

class BetaDistribution(ContinuousDistribution):
    """A beta distribution."""
    
    options = [
        ["a", "b"],
        ["alpha", "beta"],
        ["mu", "nu"],
        ["omega", "kappa"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters and "b" in parameters:
            a = parameters.pop("a")
            b = parameters.pop("b")
        elif "alpha" in parameters and "beta" in parameters:
            a = parameters.pop("alpha")
            b = parameters.pop("beta")
        elif "mu" in parameters and "nu" in parameters:
            mu = parameters.pop("mu")
            nu = parameters.pop("nu")
            a = mu * nu
            b = (1 - mu) * nu
        elif "omega" in parameters and "kappa" in parameters:
            omega = parameters.pop("omega")
            kappa = parameters.pop("kappa")
            a = omega * (kappa - 2) + 1
            b = (1 - omega) * (kappa - 2) + 1
        return scipy.stats.beta(a, b)

class BetaPrimeDistribution(ContinuousDistribution):
    """A beta prime or inverted beta distribution."""
    
    options = [
        ["a", "b"],
        ["alpha", "beta"],
        ["mu", "nu"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters and "b" in parameters:
            a = parameters.pop("a")
            b = parameters.pop("b")
        elif "alpha" in parameters and "beta" in parameters:
            a = parameters.pop("alpha")
            b = parameters.pop("beta")
        elif "mu" in parameters and "nu" in parameters:
            mu = parameters.pop("mu")
            nu = parameters.pop("nu")
            a = mu * (1 + nu)
            b = 2 + nu
        return scipy.stats.betaprime(a, b)

class CauchyDistribution(ContinuousDistribution):
    """A Cauchy distribution."""
    
    options = [
        ["loc", "scale"],
        ["x0", "gamma"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "x0" in parameters and "gamma" in parameters:
            loc = parameters.pop("x0")
            scale = parameters.pop("gamma")
        return scipy.stats.cauchy(loc, scale)

class ChiDistribution(ContinuousDistribution):
    """A chi distribution."""
    
    options = [
        ["k"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "k" in parameters:
            df = parameters.pop("k")
        return scipy.stats.chi(df)

class ChiSquaredDistribution(ContinuousDistribution):
    """A chi-squared distribution."""
    
    options = [
        ["k"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "k" in parameters:
            df = parameters.pop("k")
        self.df = df
        return scipy.stats.chi2(df)
    
    def __add__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, ChiSquaredDistribution):
            return ChiSquaredDistribution(df=self.df + other.df)
        return super().__add__(other)

class CosineDistribution(ContinuousDistribution):
    """A cosine approximation to the normal distribution."""
    
    options = [
        ["loc", "scale"],
        ["mu", "s"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters and "s" in parameters:
            loc = parameters.pop("mu")
            scale = parameters.pop("s")
        return scipy.stats.cosine(loc, scale)
    
class ErlangDistribution(ContinuousDistribution):
    """An Erlang distribution."""
    
    options = [
        ["a", "beta"],
        ["a", "scale"],
        ["k", "lambda_"],
        ["k", "beta"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters:
            k = parameters.pop("a")
            if "beta" in parameters:
                beta = parameters.pop("beta")
            elif "scale" in parameters:
                beta = 1 / parameters.pop("scale")
        elif "k" in parameters:
            k = parameters.pop("k")
            if "lambda_" in parameters:
                beta  = parameters.pop("lambda_")
            elif "beta" in parameters:
                beta = 1 / parameters.pop("beta")
        self.k = k
        self.beta = beta
        return scipy.stats.erlang(k, scale=1 / beta)
    
    def __add__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, ErlangDistribution) and self.beta == other.beta:
            return ErlangDistribution(a=self.k + other.k, beta=self.beta)
        if isinstance(other, GammaDistribution) and self.beta == other.beta:
            return GammaDistribution(alpha=self.k + other.k, beta=self.beta)
        return super().__add__(other)

class ExponentialDistribution(ContinuousDistribution):
    """An exponential continuous random variable."""
    
    options = [
        ["loc", "scale"],
        ["beta"],
        ["lambda_"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "beta" in parameters:
            loc = 0
            scale = parameters.pop("beta")
        elif "lambda_" in parameters:
            loc = 0
            scale = parameters.pop("lambda_")
        self.lambda_ = scale
        return scipy.stats.expon(loc, scale)
    
    def __sub__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, ExponentialDistribution) and self.lambda_ == other.lambda_:
            return LaplaceDistribution(mu=0, b=1 / self.lambda_)
        return super().__sub__(other)

class FDistribution(ContinuousDistribution):
    """An F continuous random variable"""
    
    options = [
        ["dfn", "dfd"],
        ["df1", "df2"],
        ["d1", "d2"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "dfn" in parameters and "dfd" in parameters:
            dfn = parameters.pop("dfn")
            dfd = parameters.pop("dfd")
        elif "df1" in parameters and "df2" in parameters:
            dfn = parameters.pop("df1")
            dfd = parameters.pop("df2")
        elif "d1" in parameters and "d2" in parameters:
            dfn = parameters.pop("d1")
            dfd = parameters.pop("d2")
        self.dfn = dfn
        self.dfd = dfd
        return scipy.stats.f(dfn, dfd)
    
    def __mul__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if other == self.dfn / self.dfd:
            return BetaPrimeDistribution(a=self.dfn / 2, b=self.dfd / 2)
        return super().__mul__(other)
    
class GammaDistribution(ContinuousDistribution):
    """A gamma distribution."""
    
    options = [
        ["alpha", "beta"],
        ["alpha", "theta"],
        ["k", "beta"],
        ["k", "theta"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "alpha" in parameters:
            k = parameters.pop("alpha")
        elif "k" in parameters:
            k = parameters.pop("k")        
        if "beta" in parameters:
            beta  = parameters.pop("beta")
        elif "theta" in parameters:
            beta = 1 / parameters.pop("theta")
        self.k = k
        self.beta = beta
        return scipy.stats.gamma(k, scale=1 / beta)
    
    def __add__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, (ErlangDistribution, GammaDistribution)) and self.beta == other.beta:
            return GammaDistribution(alpha=self.k + other.k, beta=self.beta)
        return super().__add__(other)
    
    def __truediv__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, (ErlangDistribution, GammaDistribution)) and self.beta == other.beta:
            return BetaPrimeDistribution(self.k, other.k)
        return super().__truediv__(other)
    
class GompertzDistribution(ContinuousDistribution):
    """A Gompertz distribution."""
    
    options = [
        ["c"],
        ["b", "eta"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "c" in parameters:
            c = parameters.pop("c")
            scale = 1
        elif "b" in parameters and "eta" in parameters:
            c = parameters.pop("eta")
            scale = 1 / parameters.pop("b")
        return scipy.stats.gompertz(c, scale=scale)
    
class LaplaceDistribution(ContinuousDistribution):
    """A Laplace distribution."""
    
    options = [
        ["loc", "scale"],
        ["mu", "b"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters and "b" in parameters:
            loc = parameters.pop("mu")
            scale = parameters.pop("b")
        return scipy.stats.laplace(loc, scale)
    
class LogisticDistribution(ContinuousDistribution):
    """A logistic distribution."""
    
    options = [
        ["loc", "scale"],
        ["mu", "s"],
        ["mu", "sigma"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters:
            loc = parameters.pop("mu")
            if "s" in parameters:
                scale = parameters.pop("s")
            elif "sigma" in parameters:
                scale = np.sqrt(3) / np.pi * parameters.pop("sigma")
        return scipy.stats.logistic(loc, scale)

class NormalDistribution(ContinuousDistribution):
    """A normal continuous random variable"""
    
    options = [
        ["loc", "scale"],
        ["mu", "sigma"],
        ["mean", "variance"],
        ["mu", "tau"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters:
            loc = parameters.pop("mu")
            if "sigma" in parameters:
                scale = parameters.pop("sigma")
            elif "tau" in parameters:
                scale = 1 / np.sqrt(parameters.pop("tau"))
        elif "mean" in parameters and "variance" in parameters:
            loc = parameters.pop("mean")
            scale = np.sqrt(parameters.pop("variance"))
        self.mu = loc
        self.sigma = scale
        return scipy.stats.norm(loc,scale)
    
    def __add__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, NormalDistribution):
            return NormalDistribution(mean=self.mu + other.mu, variance=self.sigma ** 2 + other.sigma ** 2)
        return super().__add__(other)
    
    def __sub__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, NormalDistribution):
            return NormalDistribution(mean=self.mu - other.mu, variance=self.sigma ** 2 + other.sigma ** 2)
        return super().__add__(other)
    
    def __truediv__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, NormalDistribution) and self.mu == 0 and other.mu == 0 and self.sigma == 1 and other.sigma == 1:
            return CauchyDistribution(x0=0, gamma=1)
        return super().__truediv__(other)

class TDistribution(ContinuousDistribution):
    """A Student's continuous t random variable"""
    
    options = [
        ["nu"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "nu" in parameters:
            df = parameters.pop("nu")
        return scipy.stats.t(df)
    
class TrapezoidalDistribution(ContinuousDistribution):
    """A trapezoidal continuous random variable"""
    
    options = [
        ["c", "d", "loc", "scale"],
        ["a", "b", "c", "d"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "c" in parameters and "d" in parameters and "loc" in parameters and "scale" in parameters:
            c = parameters.pop("c")
            d = parameters.pop("d")
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "a" in parameters and "b" in parameters and "c" in parameters and "d" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("d") - loc
            c = (parameters.pop("b") - loc) / scale
            d = (parameters.pop("c") - loc) / scale
        return scipy.stats.trapezoid(c, d, loc, scale)
    
class TriangularDistribution(ContinuousDistribution):
    """A triangular continuous random variable"""
    
    options = [
        ["c", "loc", "scale"],
        ["a", "b", "c"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "c" in parameters and "loc" in parameters and "scale" in parameters:
            c = parameters.pop("c")
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "a" in parameters and "b" in parameters and "c" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("c") - loc
            c = (parameters.pop("b") - loc) / scale
        return scipy.stats.triang(c, loc, scale)

class UniformContinuousDistribution(ContinuousDistribution):
    """A uniform continuous distribution."""
    
    options = [
        ["a", "b"],
        ["loc", "scale"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "a" in parameters and "b" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("b") - loc
        elif "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        self.a = loc
        self.b = loc + scale
        return scipy.stats.uniform(loc, scale)
    
    def __add__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, UniformContinuousDistribution):
            a, b, c, d = sorted([self.a + other.a, self.a + other.b, self.b + other.a, self.b + other.b])
            if b == c:
                return TriangularDistribution(a=a, b=b, c=d)
            return TrapezoidalDistribution(a=a, b=b, c=c, d=d)
        return super().__add__(other)
    
    def __sub__(self, other: Union[float, ContinuousDistribution]) -> ContinuousDistribution:
        if isinstance(other, UniformContinuousDistribution):
            a, b, c, d = sorted([self.a - other.a, self.a - other.b, self.b - other.a, self.b - other.b])
            if b == c:
                return TriangularDistribution(a=a, b=b, c=d)
            return TrapezoidalDistribution(a=a, b=b, c=c, d=d)
        return super().__sub__(other)
    
class WeibullDistribution(ContinuousDistribution):
    """A Weibull distribution."""
    
    options = [
        ["c"],
        ["k", "lambda_"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> rv_frozen:
        if "c" in parameters:
            c = parameters.pop("c")
            scale = 1
        elif "k" in parameters and "lambda_" in parameters:
            c = parameters.pop("k")
            scale = parameters.pop("lambda_")
        return scipy.stats.weibull_min(c, scale=scale)