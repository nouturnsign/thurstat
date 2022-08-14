from __future__ import annotations as _annotations

import abc as _abc
import operator as _operator
import typing as _typing
import warnings as _warnings
from enum import Enum as _Enum
from types import BuiltinFunctionType as _BuiltinFunctionType
from typing_extensions import Literal as _Literal, Self as _Self

import matplotlib.pyplot as _plt
import numpy as _np
import portion as _portion
import scipy.stats as _stats
from scipy.misc import derivative as _derivative
from scipy.integrate import quad_vec as _quad_vec
from scipy.interpolate import interp1d as _interp1d
from scipy.optimize import brentq as _brentq, minimize_scalar as _minimize_scalar
from scipy.stats._distn_infrastructure import rv_frozen as _rv_frozen

__all__ = [
    # global
    "pfunc", "update_defaults", "formula",
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
    "CauchyDistribution", "ChiDistribution", "ChiSquaredDistribution", "CosineDistribution",
    "ExponentialDistribution", "FDistribution", "NormalDistribution", "TDistribution",
    "UniformContinuousDistribution",
]

class pfunc(_Enum):
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

T = _typing.TypeVar('T')
NumericFunction = _typing.Callable[[float], float]
ProbabilityFunction = _typing.Union[_Literal["pdf", "pmf", "cdf", "sf", "ppf", "isf"], pfunc]

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

def update_defaults(**kwargs) -> None:
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
    warnings: the warning level to be displayed according to Python's `warning` module, defaults to `default`

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
        DEFAULTS["global_seed"] = _np.random.RandomState(_np.random.MT19937(_np.random.SeedSequence(kwargs["global_seed"])))
    if "warnings" in kwargs:
        _warnings.filterwarnings(DEFAULTS["warnings"])

class ParameterValidationError(Exception):
    """Raised when an invalid set of parameters are used to instantiate a `Distribution`."""
    
    def __init__(self, given: _typing.List[str], options: _typing.Optional[_typing.List[_typing.List[str]]]=None) -> None:
        """Create the error with the given parameters and optional parameters."""
        message = f"Failed to validate parameters. Given: {given}."
        if options is not None:
            message += f" Options: {options}."
        super().__init__(message)

class Distribution(_abc.ABC):
    """The base class for a distribution. Do not instantiate this class."""
    
    options: _typing.Optional[_typing.List[_typing.List[str]]]
    
    def __init__(self, **parameters: float) -> None:
        """Create an independent random variable given the parameters as named keyword arguments."""
        given = list(parameters.keys())
        try:
            self._dist = self.interpret_parameterization(parameters)
        except (UnboundLocalError, ParameterValidationError):
            raise ParameterValidationError(given, self.options)
        if len(parameters) > 0:
            raise ParameterValidationError(given, self.options)
    
    @_abc.abstractmethod
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _typing.Optional[_rv_frozen]:
        pass
    
    @property
    def support(self) -> _portion.Interval:
        if not hasattr(self, "_support"):
            self._support = _portion.closed(*self._dist.support())
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
    
    def generate_random_values(self, n: int) -> _np.ndarray:
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
        if isinstance(pfunc, _Enum):
            pfunc = pfunc.value
        return getattr(self._dist, pfunc)(at)
    
    def expected_value(self, func: _typing.Optional[NumericFunction]=None) -> float:
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
    def to_alias(cls, *interpret_parameters: str) -> Alias:
        return Alias(cls, *interpret_parameters)
    
    @_abc.abstractmethod
    def probability_between(self, a: float, b: float) -> float:
        pass
    
    @_abc.abstractmethod
    def probability_at(self, a: float) -> float:
        pass
        
    @_abc.abstractmethod
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction) -> _typing.Union[CustomDiscreteDistribution, CustomContinuousDistribution]:
        pass
        
    @_abc.abstractmethod
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _typing.Optional[str]=None, **kwargs) -> None:
        pass
        
    @_abc.abstractmethod
    def apply_infix_operator(self, other: _typing.Union[float, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType) -> _Self:
        pass
    
    def __add__(self, other: _typing.Union[float, _Self]) -> _Self:
        return self.apply_infix_operator(other, _operator.add, _operator.sub)

    def __radd__(self, other: _typing.Union[float, _Self]) -> _Self:
        return self + other
    
    def __sub__(self, other: _typing.Union[float, _Self]) -> _Self:
        return self.apply_infix_operator(other, _operator.sub, _operator.add)
    
    def __rsub__(self, other: _typing.Union[float, _Self]) -> _Self:
        return -(self - other)
    
    def __mul__(self, other: _typing.Union[float, _Self]) -> _Self:
        return self.apply_infix_operator(other, _operator.mul, _operator.truediv)
    
    def __rmul__(self, other: _typing.Union[float, _Self]) -> _Self:
        return self * other
    
    def __neg__(self) -> _Self:
        return self * -1
    
    def __truediv__(self, other: _typing.Union[float, _Self]) -> _Self:
        raise NotImplementedError("Division is currently not implemented for distributions.")
    
    def __rtruediv__(self, other: _typing.Union[float, _Self]) -> _Self:
        raise NotImplementedError("Division is currently not implemented for distributions.")
    
    def __pow__(self, other: _typing.Union[float, _Self]) -> _Self:
        if isinstance(other, int) and other > 0:
            if other % 2 == 0:
                return self.apply_func(lambda x: x ** other, lambda x: x ** (1 / other), lambda x: - (x ** (1 / other)))
            else:
                return self.apply_func(lambda x: x ** other, lambda x: x ** (1 / other))
        raise NotImplementedError("Exponentiation is currently not implemented between distributions or for non-integer or non-positive powers.")
    
    def __rpow__(self, other: _typing.Union[float, _Self]) -> _Self:
        if isinstance(other, int) and other > 0:
            return self.apply_func(lambda x: other ** x, lambda x: _np.log(x) / _np.log(other))
        raise NotImplementedError("Exponentiation is currently not implemented between distributions or for non-integer or non-positive bases.")
    
    def __lt__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(-_np.inf, other))
        elif isinstance(other, Distribution):
            return self - other < 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __le__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.openclosed(-_np.inf, other))
        elif isinstance(other, Distribution):
            return self - other <= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __gt__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other > 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ge__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.closedopen(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other >= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ne__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(-_np.inf, other) | _portion.open(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other != 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __eq__(self, other: _typing.Union[float, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.singleton(other))
        elif isinstance(other, Distribution):
            return self - other == 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
class formula(object):
    """A formula-like variable that can be passed as a function. Can be used as a decorator on functions."""
    
    def __init__(self, func: _typing.Optional[NumericFunction]=None) -> None:
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
    
    def __add__(self, other: float) -> _Self:
        return formula(lambda x: self.func(x) + other)
    
    def __radd__(self, other: float) -> _Self:
        return self + other
    
    def __sub__(self, other: float) -> _Self:
        return formula(lambda x: self.func(x) - other)
    
    def __rsub__(self, other: float) -> _Self:
        return formula(lambda x: other - self.func(x))
    
    def __mul__(self, other: float) -> _Self:
        return formula(lambda x: self.func(x) * other)
    
    def __rmul__(self, other: float) -> _Self:
        return self * other
    
    def __neg__(self) -> _Self:
        return self * -1
    
    def __truediv__(self, other: float) -> _Self:
        return formula(lambda x: self.func(x) / other)
    
    def __rtruediv__(self, other: float) -> _Self:
        return formula(lambda x: other / self.func(x))
    
    def __pow__(self, other: float) -> _Self:
        return formula(lambda x: self.func(x) ** other)
    
    def __rpow__(self, other: float) -> _Self:
        return formula(lambda x: other ** self.func(x))
    
class CustomDistribution(Distribution):
    """The base class for custom distributions, defined by a scipy rv. Do not instantiate this class."""
    
    options = None
    
    @_abc.abstractmethod
    def __init__(self, dist: _rv_frozen) -> None:
        self._dist = dist
    
    def interpret_parameterization(self) -> None:
        return
    
    @classmethod
    @_abc.abstractmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> _typing.Union[CustomDiscreteDistribution, CustomContinuousDistribution]:
        pass
    
class DiscreteDistribution(Distribution):
    """The base class for discrete distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a - 1)
    
    def probability_at(self, k: float) -> float:
        """Calculate the probability P(X == k)."""
        return self.evaluate("pmf", k)
    
    def apply_func(self, func: NumericFunction, *, infinity_approximation: _typing.Optional[int]=None, a: _typing.Optional[int]=None, b: _typing.Optional[int]=None) -> CustomDiscreteDistribution:
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
            if a0 == -_np.inf:
                a0 = -infinity_approximation
            if b0 == _np.inf:
                b0 = infinity_approximation
        
        if a is None:
            a = a0
        if b is None:
            b = b0
        
        x = _np.arange(a, b + 1)
        y = self.evaluate("pmf", x)
        
        x_transform = func(x)
        pmf = {}
        for e, p in zip(x_transform, y):
            pmf[e] = pmf.get(e, 0) + p
        
        a = min(x_transform)
        b = max(x_transform)
        
        return CustomDiscreteDistribution.from_pfunc("pmf", _np.vectorize(lambda a: pmf.get(a, 0)), a, b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _typing.Optional[str]=None, **kwargs) -> None:
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
        if a == -_np.inf:
            a = self.evaluate("ppf", 1 / DEFAULTS["infinity_approximation"])
        if b == _np.inf:
            b = self.evaluate("ppf", 1 - 1 / DEFAULTS["infinity_approximation"])
        x = _np.arange(a, b + 1)
        y = self.evaluate(pfunc, x)
        markerline, stemlines, baseline = _plt.stem(x, y, basefmt=" ", use_line_collection=True, **kwargs)
        if color is None:
            color = DEFAULTS["default_color"]
        markerline.set_color(color)
        stemlines.set_color(color)
        if not add:
            _plt.show()

    def apply_infix_operator(self, other: _typing.Union[float, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType = None) -> _Self:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return CustomDiscreteDistribution.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
        elif isinstance(other, DiscreteDistribution):
            a0, b0 = self.support.lower, self.support.upper
            if a0 == -_np.inf:
                a0 = -DEFAULTS["infinity_approximation"]
            if b0 == _np.inf:
                b0 = DEFAULTS["infinity_approximation"]
            a1, b1 = other.support.lower, other.support.upper
            if a1 == -_np.inf:
                a1 = -DEFAULTS["infinity_approximation"]
            if b1 == _np.inf:
                b1 = DEFAULTS["infinity_approximation"]
            a, b = _np.arange(a0, b0 + 1), _np.arange(a1, b1 + 1)
            pmf = {}
            for x, y in _np.nditer(_np.array(_np.meshgrid(a, b)), flags=['external_loop'], order='F'):
                pmf[op(x, y)] = pmf.get(op(x, y), 0) + self.evaluate("pmf", x) * other.evaluate("pmf", y)
            a2, b2 = min(pmf.keys()), max(pmf.keys())
            return CustomDiscreteDistribution.from_pfunc("pmf", _np.vectorize(lambda a: pmf.get(a, 0)), a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomDiscreteDistribution(CustomDistribution, DiscreteDistribution):
    """A custom discrete distribution."""
    
    def __init__(self, dist: _rv_frozen) -> None:
        """Create a `CustomDiscreteDistribution` object given a frozen `scipy.stats.rv_discrete` object."""
        self._dist = dist
    
    @classmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> _Self:
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
        class NewScipyDiscreteDistribution(_stats.rv_discrete): pass
        if isinstance(pfunc, _Enum):
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
        _warnings.warn("Trying to calculate the point probability of a continuous distribution.")
        return 0
    
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: _typing.Optional[float]=None, a: _typing.Optional[float]=None, b: _typing.Optional[float]=None) -> CustomContinuousDistribution:
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
            if a0 == -_np.inf:
                a0 = -infinity_approximation
            if b0 == _np.inf:
                b0 = infinity_approximation
        
        if a is None:
            result = _minimize_scalar(func, bounds=(a0, b0), method="bounded")
            a = func(result.x)
        if b is None:
            result = _minimize_scalar(lambda x: -func(x), bounds=(a0, b0), method="bounded")
            b = func(result.x)
        
        if len(inverse_funcs) == 0:
            inverse_func = _np.vectorize(lambda y: _brentq(lambda x: func(x) - y, a=a0, b=b0))
            return CustomContinuousDistribution.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
        elif len(inverse_funcs) == 1:
            inverse_func = inverse_funcs[0]
            return CustomContinuousDistribution.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
        else:
            _warnings.warn("Multiple branched inverse functions are currently questionably implemented. Use 1 to 1 functions when possible.")
            return CustomContinuousDistribution.from_pfunc("pdf", lambda y: sum(self.evaluate("pdf", inverse_func(y)) * _np.absolute(_derivative(inverse_func, y, dx=1 / infinity_approximation)) for inverse_func in inverse_funcs), a=a, b=b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _typing.Optional[str]=None, **kwargs) -> None:
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
        if a == -_np.inf:
            a = self.evaluate("ppf", 1 / DEFAULTS["infinity_approximation"])
        if b == _np.inf:
            b = self.evaluate("ppf", 1 - 1 / DEFAULTS["infinity_approximation"])
        diff = b - a
        x = _np.linspace(a - diff * DEFAULTS["buffer"], b + diff * DEFAULTS["buffer"], int(diff * DEFAULTS["ratio"]))
        y = self.evaluate(pfunc, x)
        if color is None:
            color = DEFAULTS["default_color"]
        lines = _plt.plot(x, y, color=color)
        if not add:
            _plt.show()
    
    def discretize(self) -> DiscreteDistribution:
        """Approximate the continuous distribution with a discrete distribution using the correction for continuity."""
        return CustomDiscreteDistribution.from_pfunc("pmf", lambda x: self.probability_between(x - 0.5, x + 0.5), self.support.lower, self.support.upper)
    
    def apply_infix_operator(self, other: _typing.Union[float, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType) -> _Self:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return CustomContinuousDistribution.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
        elif isinstance(other, ContinuousDistribution):
            a0, b0 = self.support.lower, self.support.upper
            if a0 == -_np.inf:
                a0 = -DEFAULTS["infinity_approximation"]
            if b0 == _np.inf:
                b0 = DEFAULTS["infinity_approximation"]
            a1, b1 = other.support.lower, other.support.upper
            if a1 == -_np.inf:
                a1 = -DEFAULTS["infinity_approximation"]
            if b1 == _np.inf:
                b1 = DEFAULTS["infinity_approximation"]
            values = op(a0, a1), op(a0, b1), op(b0, a0), op(b0, b1)
            a2, b2 = min(values), max(values)
            
            exact_pdf = lambda z: _quad_vec(lambda x: other.evaluate("pdf", inv_op(z, x)) * self.evaluate("pdf", x) * abs(1 if op != _operator.mul and op != _operator.truediv else inv_op(1, x + 1 / DEFAULTS["infinity_approximation"])), a=-_np.inf, b=_np.inf)[0]
            if DEFAULTS["exact"]:
                return CustomContinuousDistribution.from_pfunc("pdf", exact_pdf, a2, b2)
            
            diff = b2 - a2
            x = _np.linspace(a2, b2, int(diff) * DEFAULTS["ratio"])
            
            approximate_pdf = exact_pdf(x)
            @_np.vectorize
            def approximate_cdf(z):
                i = _np.searchsorted(x, z, side="right")
                res = _np.trapz(approximate_pdf[:i], x[:i])
                return res
            y = approximate_cdf(x)
            cdf = _interp1d(x[:-1], y[:-1], bounds_error=False, fill_value=(0, 1), assume_sorted=True)
            
            return CustomContinuousDistribution.from_pfunc("cdf", cdf, a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomContinuousDistribution(CustomDistribution, ContinuousDistribution):
    """A custom continuous distribution."""

    def __init__(self, dist: _rv_frozen) -> None:
        """Create a `CustomContinuousDistribution` object given a frozen `scipy.stats.rv_continuous` object."""
        self._dist = dist
    
    @classmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> _Self:
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
        class NewScipyContinuousDistribution(_stats.rv_continuous): pass
        if isinstance(pfunc, _Enum):
            pfunc = pfunc.value
        setattr(NewScipyContinuousDistribution, "_" + pfunc, staticmethod(func))
        return CustomContinuousDistribution(NewScipyContinuousDistribution(a=a, b=b))
    
class Alias(object):
    """An alias for a distribution given the parameter convention."""
    
    def __init__(self, tdist: _typing.Type[Distribution], *interpret_parameters: str) -> None:
        """Create an alias given the class of distribution and parameter names in the order they will be passed."""
        self._tdist = tdist
        self._interpret_parameters = interpret_parameters
    
    def __call__(self, *value_parameters: float) -> Distribution:
        """Return a distribution interpreted by the alias."""
        return self._tdist(**{k: v for k, v in zip(self._interpret_parameters, value_parameters)})
    
class Event(object):
    """An event described by a distribution and interval."""
    
    _last = None
    
    def __init__(self, tdist: Distribution, interval: _portion.Interval):
        """Create an event object."""
        self._tdist = tdist
        self._interval = interval
    
    def __bool__(self) -> bool:
        if Event._last is None:
            Event._last = self._interval
        return True
    
def probability_of(evt: Event) -> float:
    """Return the probability of an event."""
    probability = 0
    if Event._last is not None:
        evt._interval = evt._interval & Event._last
        Event._last = None
    
    for atomic in evt._interval._intervals:
        if atomic.lower > atomic.upper:
            continue
        elif atomic.lower == atomic.upper:
            probability += evt._tdist.probability_at(atomic.lower)
            continue
        lower = atomic.lower
        upper = atomic.upper
        if isinstance(evt._tdist, DiscreteDistribution):
            if atomic.left == _portion.OPEN:
                lower += 1
            if atomic.right == _portion.OPEN:
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
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "p" in parameters:
            p = parameters.pop("p")
        elif "q" in parameters:
            p = 1 - parameters.pop("q")
        self.p = p
        return _stats.bernoulli(p)
    
    def __add__(self, other: _typing.Union[float, _Self]) -> _Self:
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
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "n" in parameters:
            n = parameters.pop("n")
            if "a" in parameters and "b" in parameters:
                a = parameters.pop("a")
                b = parameters.pop("b")
            elif "alpha" in parameters and "beta" in parameters:
                a = parameters.pop("alpha")
                b = parameters.pop("beta")
        return _stats.betabinom(n, a, b)

class BinomialDistribution(DiscreteDistribution):
    """A binomial distribution."""
    
    options = [
        ["n", "p"],
        ["n", "q"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "n" in parameters:
            n = parameters.pop("n")
            if "p" in parameters:
                p = parameters.pop("p")
            elif "q" in parameters:
                p = 1 - parameters.pop("q")
        self.n = n
        self.p = p
        return _stats.binom(n, p)
    
    def __add__(self, other: _typing.Union[float, _Self]) -> _Self:
        if isinstance(other, BernoulliDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n + 1, p=self.p)
        if isinstance(other, BinomialDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n + other.n, p=self.p)
        return super().__add__(other)
    
    def __sub__(self, other: _typing.Union[float, _Self]) -> _Self:
        if isinstance(other, BernoulliDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n - 1, p=self.p)
        if isinstance(other, BinomialDistribution) and self.p == other.p:
            return BinomialDistribution(n=self.n - other.n, p=self.p)
        return super().__sub__(other)

class GeometricDistribution(DiscreteDistribution):
    """A geometric distribution."""
    
    options = [
        ["p"],
        ["q"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "p" in parameters:
            p = parameters.pop("p")
        elif "q" in parameters:
            p = 1 - parameters.pop("q")
        return _stats.geom(p)

class HypergeometricDistribution(DiscreteDistribution):
    """A hypergeometric distribution."""
    
    options = [
        ["M", "n", "N"],
        ["N", "K", "n"],
        ["N", "m", "n"],
        ["N", "N1", "n"],
        ["N1", "N2", "n"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
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
        return _stats.hypergeom(M, n, N)

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
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
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
        return _stats.nbinom(n, p)

class NegativeHypergeometricDistribution(DiscreteDistribution):
    """A negative hypergeometric distribution."""
    
    options = [
        ["M", "n", "r"],
        ["N", "K", "r"],
        ["N", "m", "r"],
        ["N", "N1", "r"],
        ["N1", "N2", "r"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
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
        return _stats.nhypergeom(M, n, r)

class PoissonDistribution(DiscreteDistribution):
    """A Poisson distribution."""
    
    options = [
        ["mu"],
        ["lambda_"],
        ["r", "t"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "mu" in parameters:
            mu = parameters.pop("mu")
        elif "lambda_" in parameters:
            mu = parameters.pop("lambda_")
        elif "r" in parameters and "t" in parameters:
            mu = parameters.pop("r") * parameters.pop("t")
        return _stats.poisson(mu)

class SkellamDistribution(DiscreteDistribution):
    """A Skellam distribution."""
    
    options = [
        ["mu1", "mu2"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "mu1" in parameters and "mu2" in parameters:
            mu1 = parameters.pop("mu1")
            mu2 = parameters.pop("mu2")
        return _stats.skellam(mu1, mu2)

class UniformDiscreteDistribution(DiscreteDistribution):
    """A random integer or uniform discrete distribution."""
    
    options = [
        ["a", "b"],
        ["low", "high"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "a" in parameters and "b" in parameters:
            low = parameters.pop("a")
            high = parameters.pop("b") + 1
        elif "low" in parameters and "high" in parameters:
            low = parameters.pop("low")
            high = parameters.pop("high")
        return _stats.randint(low, high)

class YuleSimonDistribution(DiscreteDistribution):
    """A Yule-Simon distribution."""
    
    options = [
        ["alpha"],
        ["rho"],
        ["a"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "alpha" in parameters:
            alpha = parameters.pop("alpha")
        elif "rho" in parameters:
            alpha = parameters.pop("rho")
        elif "a" in parameters:
            alpha = parameters.pop("a") - 1
        return _stats.yulesimon(alpha)

class ZipfDistribution(DiscreteDistribution):
    """A Zipf or zeta distribution."""
    
    options = [
        ["a"],
        ["s"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "a" in parameters:
            a = parameters.pop("a")
        elif "s" in parameters:
            a = parameters.pop("s")
        return _stats.zipf(a)

class ZipfianDistribution(DiscreteDistribution):
    """A Zipfian distribution."""
    
    options = [
        ["a", "n"],
        ["s", "N"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "a" in parameters and "n" in parameters:
            a = parameters.pop("a")
            n = parameters.pop("n")
        elif "s" in parameters:
            a = parameters.pop("s")
            n = parameters.pop("N")
        return _stats.zipfian(a, n)

class CauchyDistribution(ContinuousDistribution):
    """A Cauchy distribution."""
    
    options = [
        ["loc", "scale"],
        ["x0", "gamma"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "x0" in parameters and "gamma" in parameters:
            loc = parameters.pop("x0")
            scale = parameters.pop("gamma")
        return _stats.cauchy(loc, scale)

class ChiDistribution(ContinuousDistribution):
    """A chi distribution."""
    
    options = [
        ["k"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "k" in parameters:
            df = parameters.pop("k")
        return _stats.chi(df)

class ChiSquaredDistribution(ContinuousDistribution):
    """A chi-squared distribution."""
    
    options = [
        ["k"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "k" in parameters:
            df = parameters.pop("k")
        return _stats.chi2(df)

class CosineDistribution(ContinuousDistribution):
    """A cosine approximation to the normal distribution."""
    
    options = [
        ["loc", "scale"],
        ["mu", "s"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters and "s" in parameters:
            loc = parameters.pop("mu")
            scale = parameters.pop("s")
        return _stats.cosine(loc, scale)

class ExponentialDistribution(ContinuousDistribution):
    """An exponential continuous random variable."""
    
    options = [
        ["loc", "scale"],
        ["beta"],
        ["lambda_"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "beta" in parameters:
            loc = 0
            scale = parameters.pop("beta")
        elif "lambda_" in parameters:
            loc = 0
            scale = parameters.pop("lambda_")
        return _stats.expon(loc, scale)

class FDistribution(ContinuousDistribution):
    """An F continuous random variable"""
    
    options = [
        ["dfn", "dfd"],
        ["df1", "df2"],
        ["d1", "d2"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "dfn" in parameters and "dfd" in parameters:
            dfn = parameters.pop("dfn")
            dfd = parameters.pop("dfd")
        elif "df1" in parameters and "df2" in parameters:
            dfn = parameters.pop("df1")
            dfd = parameters.pop("df2")
        elif "d1" in parameters and "d2" in parameters:
            dfn = parameters.pop("d1")
            dfd = parameters.pop("d2")
        return _stats.f(dfn, dfd)

class NormalDistribution(ContinuousDistribution):
    """A normal continuous random variable"""
    
    options = [
        ["loc", "scale"],
        ["mu", "sigma"],
        ["mean", "variance"],
        ["mu", "tau"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        elif "mu" in parameters:
            loc = parameters.pop("mu")
            if "sigma" in parameters:
                scale = parameters.pop("sigma")
            elif "tau" in parameters:
                scale = 1 / _np.sqrt(parameters.pop("tau"))
        elif "mean" in parameters and "variance" in parameters:
            loc = parameters.pop("mean")
            scale = _np.sqrt(parameters.pop("variance"))
        return _stats.norm(loc,scale)

class TDistribution(ContinuousDistribution):
    """A Student's continuous t random variable"""
    
    options = [
        ["nu"],
        ["df"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "df" in parameters:
            df = parameters.pop("df")
        elif "nu" in parameters:
            df = parameters.pop("nu")
        return _stats.t(df)

class UniformContinuousDistribution(ContinuousDistribution):
    """A uniform continuous distribution."""
    
    options = [
        ["a", "b"],
        ["loc", "scale"],
    ]
    
    def interpret_parameterization(self, parameters: _typing.Dict[str, float]) -> _rv_frozen:
        if "a" in parameters and "b" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("b") - loc
        elif "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        return _stats.uniform(loc, scale)
