from __future__ import annotations

import abc
import operator
import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
import portion
import scipy.stats
from scipy.misc import derivative
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize_scalar
from scipy.stats._distn_infrastructure import rv_frozen

__all__ = [
    # global
    "PFUNC", "update_defaults", "formula",
    # base classes
    "Distribution", "DiscreteDistribution", "ContinuousDistribution",
    # custom equivalents of the base classes
    "CustomDistribution", "CustomDiscreteDistribution", "CustomContinuousDistribution",
    # events and probability
    "P", "probability_of",
    # predefined discrete distributions
    "BinomialDistribution", "BernoulliDistribution", "BetaBinomialDistribution", "BinomialDistribution",
    "GeometricDistribution", "HypergeometricDistribution", "NegativeBinomialDistribution",
    "NegativeHypergeometricDistribution", "PoissonDistribution", "SkellamDistribution",
    "UniformDiscreteDistribution", "YuleSimonDistribution", "ZipfDistribution", "ZipfianDistribution",
    # predefined continuous distributions
    "BetaDistribution", "BetaPrimeDistribution", "CauchyDistribution", "ChiDistribution", 
    "ChiSquaredDistribution", "CosineDistribution", "ErlangDistribution", "ExponentialDistribution", 
    "FDistribution", "GammaDistribution", "GompertzDistribution", "LaplaceDistribution", 
    "LogisticDistribution", "NormalDistribution", "TDistribution", "TrapezoidalDistribution",
    "TriangularDistribution", "UniformContinuousDistribution", "WeibullDistribution",
]

T = TypeVar('T')
NumericFunction = Callable[[float], float]
InfixOperator = Callable[[float, float], float]
ProbabilityFunction = Union[Literal["pdf", "pmf", "cdf", "sf", "ppf", "isf"], Enum]

DEFAULTS = {
    "infinity_approximation": 1e6,
    "exact": False,
    "ratio": 200,
    "buffer": 0.2,
    "default_color": "C0",
    "local_seed": None,
    "global_seed": None,
    "warnings": "default",
}

def update_defaults(**kwargs: Any) -> None:
    """
    Update the global defaults.
    
    Parameters
    ----------
    infinity_approximation: float
        Large enough to be considered a finite infinity, defaults to `1e6`
    exact: bool
        Whether or not to use approximations in continuous random variable arithmetic, defaults to `False`
    ratio: int
        The ratio of points plotted to distance between endpoints when displaying, defaults to `200`
    buffer: float
        The additional percent of the width to be plotted to both the right and left.
    default_color: str
        default matplotlib color to be used when plotting, defaults to `"C0"`
    local_seed: int
        The numeric value of the seed when calling any function or None if no local seed, defaults to `None`
    global_seed: int
        The numeric value of the seed singleton to be set at the beginning or None if no global seed, defaults to `None`
    warnings: str
        The warning level to be displayed according to Python's `warning` module, defaults to `default`

    Returns
    -------
    None
    
    Notes
    -----
    As an example for the seeds, setting local seed will mean that calling `generate_random_values` on a distribution will result in the same sequence of values. Setting global seed will mean that calling `generate_random_values` on a distribution will start from the same sequence of values but keep progressing through the seed.
    Using an invalid keyword will not raise an error; make sure to spell correctly.
    """
    
    DEFAULTS.update(kwargs)
    if "global_seed" in kwargs:
        DEFAULTS["global_seed"] = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(kwargs["global_seed"])))
    if "warnings" in kwargs:
        warnings.filterwarnings(DEFAULTS["warnings"])
        
class PFUNC(Enum):
    """Acceptable probability functions."""
    
    PDF: str = "pdf"
    PROBABILITY_DENSITY_FUNCTION: str = "pdf"
    DENSITY_FUNCTION: str = "pdf"
    
    PMF: str = "pmf"
    PROBABILITY_MASS_FUNCTION: str = "pmf"
    MASS_FUNCTION: str = "pmf"
    
    CDF: str = "cdf"
    CUMULATIVE_DISTRIBUTION_FUNCTION: str = "cdf"
    DISTRIBUTION_FUNCTION: str = "cdf"
    
    SF : str = "sf"
    SURVIVAL_FUNCTION: str = "sf"
    SURVIVOR_FUNCTION: str = "sf"
    RELIABILITY_FUNCTION: str = "sf"
    
    PPF: str = "ppf"
    PERCENT_POINT_FUNCTION: str = "ppf"
    PERCENTILE_FUNCTION: str = "ppf"
    QUANTILE_FUNCTION: str = "ppf"
    INVERSE_CUMULATIVE_DISTRIBUTION_FUNCTION: str = "ppf"
    INVERSE_DISTRIBUTION_FUNCTION: str = "ppf"
    
    ISF: str = "isf"
    INVERSE_SURVIVAL_FUNCTION: str = "isf"
    INVERSE_SURVIVOR_FUNCTION: str = "isf"
    INVERSE_RELIABILITY_FUNCTION: str = "isf"

class ParameterValidationError(Exception):
    """Raised when an invalid set of parameters are used to instantiate a `Distribution`."""
    
    def __init__(self, given: List[str], options: Optional[List[List[str]]]=None) -> None:
        """Create the error with the given parameters and optional parameters."""
        message = f"Failed to validate parameters. Given: {given}."
        if options is not None:
            message += f" Options: {options}."
        super().__init__(message)

class Distribution(abc.ABC):
    """The base class for a distribution. Do not instantiate this class."""
    
    options: Optional[List[List[str]]]
    
    def __init__(self, **parameters: float) -> None:
        """Create an independent random variable given the parameters as named keyword arguments."""
        given = list(parameters.keys())
        try:
            self._dist = self.interpret_parameterization(parameters)
        except (UnboundLocalError, ParameterValidationError):
            raise ParameterValidationError(given, self.options)
        if len(parameters) > 0:
            raise ParameterValidationError(given, self.options)
    
    @abc.abstractmethod
    def interpret_parameterization(self, parameters: Dict[str, float]) -> Optional[rv_frozen]:
        pass
    
    @property
    def support(self) -> portion.interval.Interval.Interval:
        if not hasattr(self, "_support"):
            self._support = portion.closed(*self._dist.support())
        return self._support
    
    @property
    def median(self) -> float:
        if not hasattr(self, "_median"):
            self._median = self._dist.median()
        return self._median
    
    @property
    def mean(self) -> float:
        if not hasattr(self, "_mean"):
            self._mean, self._variance, self._skewness, self._excess_kurtosis = self._dist.stats(moments="mvsk")
            self._standard_deviation = self._variance ** 0.5
        return self._mean
    
    @property
    def variance(self) -> float:
        if not hasattr(self, "_variance"):
            self._mean, self._variance, self._skewness, self._excess_kurtosis = self._dist.stats(moments="mvsk")
            self._standard_deviation = self._variance ** 0.5
        return self._variance
    
    @property
    def skewness(self) -> float:
        if not hasattr(self, "_skewness"):
            self._mean, self._variance, self._skewness, self._excess_kurtosis = self._dist.stats(moments="mvsk")
            self._standard_deviation = self._variance ** 0.5
        return self._skewness
    
    @property
    def excess_kurtosis(self) -> float:
        if not hasattr(self, "_excess_kurtosis"):
            self._mean, self._variance, self._skewness, self._excess_kurtosis = self._dist.stats(moments="mvsk")
            self._standard_deviation = self._variance ** 0.5
        return self._excess_kurtosis
    
    @property
    def standard_deviation(self) -> float:
        if not hasattr(self, "_standard_deviation"):
            self._standard_deviation = self._dist.std()
            self._variance = self._standard_deviation ** 2
        return self._standard_deviation
    
    def generate_random_values(self, n: int) -> np.ndarray:
        """Generate n random values."""
        if DEFAULTS["local_seed"] is not None:
            seed = DEFAULTS["local_seed"]
        else:
            seed = DEFAULTS["global_seed"]
        return self._dist.rvs(size=n, random_state=seed)
    
    def evaluate(self, pfunc: ProbabilityFunction, at: float) -> float:
        """
        Evaluate a probability function at some value.
        
        Parameters
        ----------
        pfunc: ProbabilityFunction
            One of the supported probability function abbreviations. See `thurstat.pfunc`.
        at: float
            The value to evaluate at.
        
        Returns
        -------
        float
            pfunc(at)
        """
        if isinstance(pfunc, Enum):
            pfunc = pfunc.value
        try:
            pfunc = getattr(self._dist, pfunc)
        except AttributeError:
            msg = "Invalid pfunc abbreviation. See PFUNC."
            if pfunc == "pmf":
                msg += " Did you mean pdf instead of pmf?"
            elif pfunc == "pdf":
                msg += " Did you mean pmf instead of pdf?"
            raise ValueError(msg)
        return pfunc(at)
    
    def expected_value(self, func: Optional[NumericFunction]=None) -> float:
        """
        Get the expected value of a function on the distribution.
        
        Parameters
        ----------
        func: Optional[NumericFunction], default=None
            A function that accepts a numeric input and returns a numeric output.
        
        Returns
        -------
        float
            E[func(X)]
        """
        return self._dist.expect(func)
    
    def moment(self, n: int) -> float:
        """Return the nth moment."""
        return self._dist.moment(n)
    
    @classmethod
    def to_alias(cls, *characterization: str) -> Alias:
        return Alias(cls, *characterization)
    
    @abc.abstractmethod
    def probability_between(self, a: float, b: float) -> float:
        pass
    
    @abc.abstractmethod
    def probability_at(self, a: float) -> float:
        pass
        
    @abc.abstractmethod
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction) -> CustomDistribution:
        pass
        
    @abc.abstractmethod
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs: Any) -> None:
        pass
        
    @abc.abstractmethod
    def apply_infix_operator(self, other: Union[float, Distribution], op: InfixOperator, inv_op: InfixOperator) -> Distribution:
        pass
    
    def __add__(self, other: Union[float, Distribution]) -> Distribution:
        return self.apply_infix_operator(other, operator.add, operator.sub)

    def __radd__(self, other: Union[float, Distribution]) -> Distribution:
        return self + other
    
    def __sub__(self, other: Union[float, Distribution]) -> Distribution:
        return self.apply_infix_operator(other, operator.sub, operator.add)
    
    def __rsub__(self, other: Union[float, Distribution]) -> Distribution:
        return -(self - other)
    
    def __mul__(self, other: Union[float, Distribution]) -> Distribution:
        return self.apply_infix_operator(other, operator.mul, operator.truediv)
    
    def __rmul__(self, other: Union[float, Distribution]) -> Distribution:
        return self * other
    
    def __neg__(self) -> Distribution:
        return self * -1
    
    def __truediv__(self, other: Union[float, Distribution]) -> Distribution:
        raise NotImplementedError("Division is currently not implemented for distributions.")
    
    def __rtruediv__(self, other: Union[float, Distribution]) -> Distribution:
        raise NotImplementedError("Division is currently not implemented for distributions.")
    
    def __pow__(self, other: Union[float, Distribution]) -> Distribution:
        if isinstance(other, int) and other > 0:
            if other % 2 == 0:
                return self.apply_func(lambda x: x ** other, lambda x: x ** (1 / other), lambda x: - (x ** (1 / other)))
            else:
                return self.apply_func(lambda x: x ** other, lambda x: x ** (1 / other))
        raise NotImplementedError("Exponentiation is currently not implemented between distributions or for non-integer or non-positive powers.")
    
    def __rpow__(self, other: Union[float, Distribution]) -> Distribution:
        if isinstance(other, int) and other > 0:
            return self.apply_func(lambda x: other ** x, lambda x: np.log(x) / np.log(other))
        raise NotImplementedError("Exponentiation is currently not implemented between distributions or for non-integer or non-positive bases.")
    
    def __lt__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.open(-np.inf, other))
        elif isinstance(other, Distribution):
            return self - other < 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __le__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.openclosed(-np.inf, other))
        elif isinstance(other, Distribution):
            return self - other <= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __gt__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.open(other, np.inf))
        elif isinstance(other, Distribution):
            return self - other > 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ge__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.closedopen(other, np.inf))
        elif isinstance(other, Distribution):
            return self - other >= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ne__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.open(-np.inf, other) | portion.open(other, np.inf))
        elif isinstance(other, Distribution):
            return self - other != 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __eq__(self, other: Union[float, Distribution]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, portion.singleton(other))
        elif isinstance(other, Distribution):
            return self - other == 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
class CustomDistribution(Distribution):
    """The base class for custom distributions, defined by a scipy rv. Do not instantiate this class."""
    
    options = None
    
    @abc.abstractmethod
    def __init__(self, dist: rv_frozen) -> None:
        self._dist = dist
    
    def interpret_parameterization(self) -> None:
        return
    
    @classmethod
    @abc.abstractmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> Union[CustomDiscreteDistribution, CustomContinuousDistribution]:
        pass
    
class DiscreteDistribution(Distribution):
    """The base class for discrete distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a - 1)
    
    def probability_at(self, k: float) -> float:
        """Calculate the probability P(X == k)."""
        return self.evaluate("pmf", k)
    
    def apply_func(self, func: NumericFunction, *, infinity_approximation: Optional[int]=None, a: Optional[int]=None, b: Optional[int]=None) -> CustomDiscreteDistribution:
        """
        Apply a function to the distribution to create a new distribution.
        
        Parameters
        ----------
        func: NumericFunction
            The function to apply to the distribution.
        infinity_approximation: Optional[int], default=None
            What value to use as an approximation for infinity when the original distribution's support is unbounded and the new distribution's support is not explicitly defined.
        a, b: Optional[int], default=None
            What values to use as the support of the new distribution.
        
        Returns
        -------
        CustomDiscreteDistriibution
            The new distribution.
        """
        a0, b0 = self.support.lower, self.support.upper
        
        if (infinity_approximation is None) and (a is None) and (b is None):
            infinity_approximation = DEFAULTS["infinity_approximation"]
        
        if infinity_approximation is not None:
            if a0 == -np.inf:
                a0 = -infinity_approximation
            if b0 == np.inf:
                b0 = infinity_approximation
        
        if a is None:
            a = a0
        if b is None:
            b = b0
        
        x = np.arange(a, b + 1)
        y = self.evaluate("pmf", x)
        
        x_transform = func(x)
        pmf = {}
        for e, p in zip(x_transform, y):
            pmf[e] = pmf.get(e, 0) + p
        
        a = min(x_transform)
        b = max(x_transform)
        
        return CustomDiscreteDistribution.from_pfunc("pmf", np.vectorize(lambda a: pmf.get(a, 0)), a, b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs: Any) -> None:
        """
        Display a probability function.
        
        Parameters
        ----------
        pfunc: ProbabilityFunction
            One of the supported probability function abbreviations. See `thurstat.pfunc`.
        add: bool, default=False
            Whether or not to start or add to an existing plot. `plt.show()` must be called later.
        color: Optional[str], default=None
            What color to use. Defaults to matplotlib's default blue color.
        **kwargs
            Additional keyword arguments to pass to `plt.stem`.
        
        Returns
        -------
        None
        """
        a, b = self.support.lower, self.support.upper
        if a == -np.inf:
            a = self.evaluate("ppf", 1 / DEFAULTS["infinity_approximation"])
        if b == np.inf:
            b = self.evaluate("ppf", 1 - 1 / DEFAULTS["infinity_approximation"])
        x = np.arange(a, b + 1)
        y = self.evaluate(pfunc, x)
        markerline, stemlines, baseline = plt.stem(x, y, basefmt=" ", use_line_collection=True, **kwargs)
        if color is None:
            color = DEFAULTS["default_color"]
        markerline.set_color(color)
        stemlines.set_color(color)
        if not add:
            plt.show()

    def apply_infix_operator(self, other: Union[float, DiscreteDistribution], op: InfixOperator, inv_op: InfixOperator=None) -> CustomDiscreteDistribution:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return CustomDiscreteDistribution.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
        elif isinstance(other, DiscreteDistribution):
            a0, b0 = self.support.lower, self.support.upper
            if a0 == -np.inf:
                a0 = -DEFAULTS["infinity_approximation"]
            if b0 == np.inf:
                b0 = DEFAULTS["infinity_approximation"]
            a1, b1 = other.support.lower, other.support.upper
            if a1 == -np.inf:
                a1 = -DEFAULTS["infinity_approximation"]
            if b1 == np.inf:
                b1 = DEFAULTS["infinity_approximation"]
            a, b = np.arange(a0, b0 + 1), np.arange(a1, b1 + 1)
            pmf = {}
            for x, y in np.nditer(np.array(np.meshgrid(a, b)), flags=['external_loop'], order='F'):
                pmf[op(x, y)] = pmf.get(op(x, y), 0) + self.evaluate("pmf", x) * other.evaluate("pmf", y)
            a2, b2 = min(pmf.keys()), max(pmf.keys())
            return CustomDiscreteDistribution.from_pfunc("pmf", np.vectorize(lambda a: pmf.get(a, 0)), a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomDiscreteDistribution(CustomDistribution, DiscreteDistribution):
    """A custom discrete distribution."""
    
    def __init__(self, dist: rv_frozen) -> None:
        """Create a `CustomDiscreteDistribution` object given a frozen `scipy.stats.rv_discrete` object."""
        self._dist = dist
    
    @classmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> Self:
        """
        Create a distribution from a probability function.
        
        Parameters
        ----------
        pfunc: ProbabilityFunction
            One of the supported probability function abbreviations. See `thurstat.pfunc`.
        func: NumericFunction
            The function itself.
        a, b: Numeric
            The support of the distribution.
        
        Returns
        -------
        CustomDiscreteDistribution
            The new distribution.
        """
        class NewScipyDiscreteDistribution(scipy.stats.rv_discrete): pass
        if isinstance(pfunc, Enum):
            pfunc = pfunc.value
        setattr(NewScipyDiscreteDistribution, "_" + pfunc, staticmethod(func))
        return CustomDiscreteDistribution(NewScipyDiscreteDistribution(a=a, b=b))
    
class ContinuousDistribution(Distribution):
    """The base class for continuous distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a)
    
    def probability_at(self, x: float) -> float:
        """Calculate the probability P(X == x). Always returns 0."""
        warnings.warn("Trying to calculate the point probability of a continuous distribution.")
        return 0
    
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: Optional[float]=None, a: Optional[float]=None, b: Optional[float]=None) -> CustomContinuousDistribution:
        """
        Apply a function to the distribution to create a new distribution.
        
        Parameters
        ----------
        func: NumericFunction
            The function to apply to the distribution.
        *inverse_funcs: NumericFunction
            The branches of the inverse of `func`. Currently, multiple branches are not implemented.
        infinity_approximation: Optional[float], default=None
            What value to use as an approximation for infinity when the original distribution's support is unbounded and the new distribution's support is not explicitly defined.
        a, b: Optional[float], default=None
            What values to use as the support of the new distribution.
        
        Returns
        -------
        CustomContinuousDistribution
            The new distribution.
        """
        a0, b0 = self.support.lower, self.support.upper
        
        if (infinity_approximation is None) and (a is None) and (b is None):
            infinity_approximation = DEFAULTS["infinity_approximation"]
        
        if infinity_approximation is not None:
            if a0 == -np.inf:
                a0 = -infinity_approximation
            if b0 == np.inf:
                b0 = infinity_approximation
        
        if a is None:
            result = minimize_scalar(func, bounds=(a0, b0), method="bounded")
            a = func(result.x)
        if b is None:
            result = minimize_scalar(lambda x: -func(x), bounds=(a0, b0), method="bounded")
            b = func(result.x)
        
        if len(inverse_funcs) == 0:
            inverse_func = np.vectorize(lambda y: brentq(lambda x: func(x) - y, a=a0, b=b0))
            return CustomContinuousDistribution.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
        elif len(inverse_funcs) == 1:
            inverse_func = inverse_funcs[0]
            return CustomContinuousDistribution.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
        else:
            warnings.warn("Multiple branched inverse functions are currently questionably implemented. Use 1 to 1 functions when possible.")
            return CustomContinuousDistribution.from_pfunc("pdf", lambda y: sum(self.evaluate("pdf", inverse_func(y)) * np.absolute(derivative(inverse_func, y, dx=1 / infinity_approximation)) for inverse_func in inverse_funcs), a=a, b=b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs: Any) -> None:
        """
        Display a probability function.
        
        Parameters
        ----------
        pfunc: ProbabilityFunction
            One of the supported probability function abbreviations. See `thurstat.pfunc`.
        add: bool, default=False
            Whether or not to start or add to an existing plot. `plt.show()` must be called later.
        color: Optional[str], default=None
            What color to use. Defaults to matplotlib's default blue color.
        **kwargs
            Additional keyword arguments to pass to `plt.stem`.
        
        Returns
        -------
        None
        """
        a, b = self.support.lower, self.support.upper
        if a == -np.inf:
            a = self.evaluate("ppf", 1 / DEFAULTS["infinity_approximation"])
        if b == np.inf:
            b = self.evaluate("ppf", 1 - 1 / DEFAULTS["infinity_approximation"])
        diff = b - a
        x = np.linspace(a - diff * DEFAULTS["buffer"], b + diff * DEFAULTS["buffer"], int(diff * DEFAULTS["ratio"]))
        y = self.evaluate(pfunc, x)
        if color is None:
            color = DEFAULTS["default_color"]
        lines = plt.plot(x, y, color=color, **kwargs)
        if not add:
            plt.show()
    
    def discretize(self) -> DiscreteDistribution:
        """Approximate the continuous distribution with a discrete distribution using the correction for continuity."""
        return CustomDiscreteDistribution.from_pfunc("pmf", lambda x: self.probability_between(x - 0.5, x + 0.5), self.support.lower, self.support.upper)
    
    def apply_infix_operator(self, other: Union[float, ContinuousDistribution], op: InfixOperator, inv_op: InfixOperator) -> CustomContinuousDistribution:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return CustomContinuousDistribution.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
        elif isinstance(other, ContinuousDistribution):
            a0, b0 = self.support.lower, self.support.upper
            if a0 == -np.inf:
                a0 = -DEFAULTS["infinity_approximation"]
            if b0 == np.inf:
                b0 = DEFAULTS["infinity_approximation"]
            a1, b1 = other.support.lower, other.support.upper
            if a1 == -np.inf:
                a1 = -DEFAULTS["infinity_approximation"]
            if b1 == np.inf:
                b1 = DEFAULTS["infinity_approximation"]
            values = [op(a0, a1), op(a0, b1), 
                      op(a1, a0), op(a1, b0),
                      op(b0, a1), op(b0, b1),
                      op(b1, a0), op(b1, b0)]
            a2, b2 = min(values), max(values)
            
            exact_pdf = lambda z: quad_vec(lambda x: other.evaluate("pdf", inv_op(z, x)) * self.evaluate("pdf", x) * abs(1 if op != operator.mul and op != operator.truediv else inv_op(1, x + 1 / DEFAULTS["infinity_approximation"])), a=-np.inf, b=np.inf)[0]
            if DEFAULTS["exact"]:
                return CustomContinuousDistribution.from_pfunc("pdf", exact_pdf, a2, b2)
            
            diff = b2 - a2
            x = np.linspace(a2, b2, int(diff) * DEFAULTS["ratio"])
            
            approximate_pdf = exact_pdf(x)
            @np.vectorize
            def approximate_cdf(z):
                i = np.searchsorted(x, z, side="right")
                res = np.trapz(approximate_pdf[:i], x[:i])
                return res
            y = approximate_cdf(x)
            cdf = interp1d(x[:-1], y[:-1], bounds_error=False, fill_value=(0, 1), assume_sorted=True)
            
            return CustomContinuousDistribution.from_pfunc("cdf", cdf, a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomContinuousDistribution(CustomDistribution, ContinuousDistribution):
    """A custom continuous distribution."""

    def __init__(self, dist: rv_frozen) -> None:
        """Create a `CustomContinuousDistribution` object given a frozen `scipy.stats.rv_continuous` object."""
        self._dist = dist
    
    @classmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> Self:
        """
        Create a distribution from a probability function.
        
        Parameters
        ----------
        pfunc: ProbabilityFunction
            One of the supported probability function abbreviations. See `thurstat.pfunc`.
        func: NumericFunction
            The function itself.
        a, b: Numeric
            The support of the distribution.
        
        Returns
        -------
        CustomContinuousDistribution
            The new distribution.
        """
        class NewScipyContinuousDistribution(scipy.stats.rv_continuous): pass
        if isinstance(pfunc, Enum):
            pfunc = pfunc.value
        setattr(NewScipyContinuousDistribution, "_" + pfunc, staticmethod(func))
        return CustomContinuousDistribution(NewScipyContinuousDistribution(a=a, b=b))
    
class Alias(object):
    """An alias for a distribution given the parameter convention."""
    
    def __init__(self, tdist: Type[Distribution], *characterization: str) -> None:
        """Create an alias given the class of distribution and parameter names in the order they will be passed."""
        self._tdist = tdist
        self._characterization = characterization
    
    def __call__(self, *parameters: float) -> Distribution:
        """Return a distribution interpreted by the alias."""
        return self._tdist(**{k: v for k, v in zip(self._characterization, parameters)})
    
class formula(object):
    """A formula-like variable that can be passed as a function. Can be used as a decorator on functions."""
    
    def __init__(self, func: Optional[NumericFunction]=None) -> None:
        """Create a formula variable, optionally with a func. Defaults to the identity function."""
        if func is None:
            func = lambda x: x
        self.func = func
    
    def __call__(self, other: T) -> T:
        if isinstance(other, formula):
            return formula(lambda x: self.func(other(x)))
        elif isinstance(other, Distribution):
            return other.apply_func(self.func)
        else:
            return self.func(other)
    
    def __add__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: self.func(x) + other.func(x))
        return formula(lambda x: self.func(x) + other)
    
    def __radd__(self, other: Union[float, Self]) -> Self:
        return self + other
    
    def __sub__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: self.func(x) - other.func(x))
        return formula(lambda x: self.func(x) - other)
    
    def __rsub__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: other.func(x) - self.func(x))
        return formula(lambda x: other - self.func(x))
    
    def __mul__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: self.func(x) * other.func(x))
        return formula(lambda x: self.func(x) * other)
    
    def __rmul__(self, other: Union[float, Self]) -> Self:
        return self * other
    
    def __neg__(self) -> Self:
        return self * -1
    
    def __truediv__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: self.func(x) / other.func(x))
        return formula(lambda x: self.func(x) / other)
    
    def __rtruediv__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: other.func(x) / self.func(x))
        return formula(lambda x: other / self.func(x))
    
    def __pow__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: self.func(x) ** other.func(x))
        return formula(lambda x: self.func(x) ** other)
    
    def __rpow__(self, other: Union[float, Self]) -> Self:
        if isinstance(other, formula):
            return formula(lambda x: other.func(x) ** self.func(x))
        return formula(lambda x: other ** self.func(x))
    
class Event(object):
    """An event described by a distribution and interval."""
    
    _last = None
    
    def __init__(self, tdist: Distribution, interval: portion.interval.Interval):
        """Create an event object."""
        self._tdist = tdist
        self._interval = interval
        if Event._last is not None:
            self._interval = self._interval & Event._last
            Event._last = None
    
    def __bool__(self) -> bool:
        if Event._last is None:
            Event._last = self._interval
        return True
    
def probability_of(evt: Event) -> float:
    """Return the probability of an event."""
    probability = 0
    
    for atomic in evt._interval._intervals:
        if atomic.lower > atomic.upper:
            continue
        elif atomic.lower == atomic.upper:
            probability += evt._tdist.probability_at(atomic.lower)
            continue
        lower = atomic.lower
        upper = atomic.upper
        if isinstance(evt._tdist, DiscreteDistribution):
            if atomic.left == portion.OPEN:
                lower += 1
            if atomic.right == portion.OPEN:
                upper -= 1
        probability += evt._tdist.probability_between(lower, upper)
    return probability

def P(evt: Event) -> float:
    """Return the probability of an event."""
    return probability_of(evt)

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

del abc, operator, warnings
del annotations, Enum, Literal, Self
del Any, Dict, List, Optional, Type, Union
del T, NumericFunction, ProbabilityFunction
del plt, np, portion, scipy.stats
del derivative, quad_vec, interp1d, brentq, minimize_scalar, rv_frozen