from __future__ import annotations

import abc
import operator
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from typing_extensions import Self

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_frozen

import portion
from ._utils import DEFAULTS, ParameterValidationError, NumericFunction, InfixOperator, ProbabilityFunction

__all__ = [
    "Distribution", "CustomDistribution",
    "DiscreteDistribution", "CustomDiscreteDistribution",
    "ContinuousDistribution", "CustomContinuousDistribution",
]

T = TypeVar('T')

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
    def support(self) -> portion.interval.Interval:
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
            A function that accepts a numeric input and returns a numeric output. Defaults to the identity function.
        
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
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: Optional[float]=None, a: Optional[float]=None, b: Optional[float]=None) -> CustomDistribution:
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
    
    def apply_func(self, func: NumericFunction, *inverse_funcs, infinity_approximation: Optional[int]=None, a: Optional[int]=None, b: Optional[int]=None) -> CustomDiscreteDistribution:
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
        CustomDiscreteDistribution
            The new distribution.
            
        Notes
        -----
        Passing additional arguments into func will raise a warning. Inverse functions are not considered in this implementation.
        """
        if len(inverse_funcs) > 0:
            warnings.warn(f"Passed {len(inverse_funcs)} inverse functions to apply_func on a DiscreteDistribution. These will be ignored.")
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
        class NewScipyDiscreteDistribution(rv_discrete): pass
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
            Additional keyword arguments to pass to `plt.plot`.
        
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
        class NewScipyContinuousDistribution(rv_continuous): pass
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
        return self._tdist(**dict(zip(self._characterization, parameters)))
    
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
