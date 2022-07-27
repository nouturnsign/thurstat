from __future__ import annotations as _annotations

import abc as _abc
import operator as _operator
import warnings as _warnings
from types import BuiltinFunctionType as _BuiltinFunctionType
from typing import (
    Callable as _Callable, 
    Dict as _Dict, 
    List as _List, 
    NamedTuple as _NamedTuple, 
    Optional as _Optional, 
    Type as _Type, 
    Union as _Union
)
from typing_extensions import Literal as _Literal, Self as _Self

import matplotlib.pyplot as _plt
import numpy as _np
import portion as _portion
import scipy.stats as _stats
from scipy.integrate import quad_vec as _quad_vec
from scipy.interpolate import interp1d as _interp1d
from scipy.optimize import brentq as brentq, minimize_scalar as _minimize_scalar

__all__ = [
    # global
    "pfunc", "update_defaults",
    # formula
    "FormulaVariable", "formula",
    # base classes
    "Distribution", "DiscreteDistribution", "ContinuousDistribution",
    # instantiable equivalents of the base classes
    "CustomDistribution", "CustomDiscreteDistribution", "CustomContinuousDistribution",
    # events and probability
    "P",
    # predefined discrete distributions
    "UniformDiscreteDistribution", "BinomialDistribution",
    # predefined continuous distributions
    "UniformContinuousDistribution",
]

Numeric = _Union[int, float]
NumericFunction = _Callable[[Numeric], Numeric]
ProbabilityFunction = _Literal["pdf", "pmf", "cdf", "sf", "ppf", "isf"]

class pfunc(_NamedTuple):
    """Acceptable pfunc abbreviations."""
    PDF: str = "pdf"
    PMF: str = "pmf"
    CDF: str = "cdf"
    SF : str = "sf"
    PPF: str = "ppf"
    ISF: str = "isf"

DEFAULTS = {
    "infinity_approximation": 1e6,
    "exact": False,
    "ratio": 200,
    "default_color": "C0",
    "local_seed": None,
    "global_seed": None,
    "warnings": "default",
}

def update_defaults(**kwargs):
    """Update the global defaults."""
    
    DEFAULTS.update(kwargs)
    if "global_seed" in kwargs:
        DEFAULTS["global_seed"] = _np.random.RandomState(_np.random.MT19937(_np.random.SeedSequence(kwargs["global_seed"])))
    if "warnings" in kwargs:
        _warnings.filterwarnings(DEFAULTS["warnings"])

class ParameterValidationError(Exception):
    """Raised when an invalid set of parameters are used to instantiate a Distribution."""
    
    def __init__(self, given: _List[str], options: _Optional[_List[_List[str]]]=None):
        """Create the error with the given parameters and optional parameters."""
        message = f"Failed to validate parameters. Given: {given}."
        if options is not None:
            message += f" Options: {options}."
        super().__init__(message)

class Distribution(_abc.ABC):
    """The base class for a distribution. Do not instantiate this class."""
    
    options: _Optional[_List[_List[str]]]
    
    def __init__(self, **parameters: float) -> None:
        """Create a distribution object given the parameters as named keyword arguments."""
        given = list(parameters.keys())
        try:
            self._dist = self.interpret_parameterization(parameters)
        except UnboundLocalError:
            raise ParameterValidationError(given, self.options)
        if len(parameters) > 0:
            raise ParameterValidationError(given, self.options)
    
    @_abc.abstractmethod  
    def interpret_parameterization(self, parameters: _Dict[str, float]) -> _Union[_stats.rv_continuous, _stats.rv_discrete, None]:
        pass
    
    @property
    def support(self):
        if not hasattr(self, "_support"):
            self._support = _portion.closed(*self._dist.support())
        return self._support
    
    @property
    def median(self):
        if not hasattr(self, "_median"):
            self._median = self._dist.median()
        return self._median
    
    @property
    def mean(self):
        if not hasattr(self, "_mean"):
            self._mean, self._variance, self._skewness, self._kurtosis = self._dist.stats(moments="mvsk")
        return self._mean
    
    @property
    def variance(self):
        if not hasattr(self, "_variance"):
            self._mean, self._variance, self._skewness, self._kurtosis = self._dist.stats(moments="mvsk")
        return self._variance
    
    @property
    def skewness(self):
        if not hasattr(self, "_skewness"):
            self._mean, self._variance, self._skewness, self._kurtosis = self._dist.stats(moments="mvsk")
        return self._skewness
    
    @property
    def kurtosis(self):
        if not hasattr(self, "_kurtosis"):
            self._mean, self._variance, self._skewness, self._kurtosis = self._dist.stats(moments="mvsk")
        return self._kurtosis
    
    @property
    def standard_deviation(self):
        if not hasattr(self, "_standard_deviation"):
            self._standard_deviation = self._dist.std()
        return self._standard_deviation
    
    def generate_random_values(self, n: int) -> _np.ndarray:
        """Generate n random values."""
        if DEFAULTS["local_seed"] is not None:
            seed = DEFAULTS["local_seed"]
        else:
            seed = DEFAULTS["global_seed"]
        return self._dist.rvs(n, random_state=seed)
    
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
        return getattr(self._dist, pfunc)(at)
    
    def expected_value(self, func: _Optional[NumericFunction]=None) -> float:
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
    
    @_abc.abstractmethod
    def probability_between(self, a: float, b: float) -> float:
        pass
    
    @_abc.abstractmethod
    def probability_at(self, a: float) -> float:
        pass
    
    @_abc.abstractmethod
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction) -> _Union[CustomDiscreteDistribution, CustomContinuousDistribution]:
        pass
    
    @_abc.abstractmethod
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _Optional[str]=None, **kwargs) -> None:
        pass
    
    @classmethod
    @_abc.abstractmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Numeric, b: Numeric) -> _Union[CustomDiscreteDistribution, CustomContinuousDistribution]:
        pass
    
    @_abc.abstractmethod
    def apply_infix_operator(self, other: _Union[Numeric, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType) -> _Self:
        pass
    
    def __add__(self, other: _Union[Numeric, _Self]) -> _Self: 
        return self.apply_infix_operator(other, _operator.add, _operator.sub)

    def __radd__(self, other: _Union[Numeric, _Self]) -> _Self: 
        return self + other
    
    def __sub__(self, other: _Union[Numeric, _Self]) -> _Self:
        return self.apply_infix_operator(other, _operator.sub, _operator.add)
    
    def __rsub__(self, other: _Union[Numeric, _Self]) -> _Self:
        return -(self - other)
    
    def __mul__(self, other: _Union[Numeric, _Self]) -> _Self: 
        return self.apply_infix_operator(other, _operator.mul, _operator.truediv)
    
    def __rmul__(self, other: _Union[Numeric, _Self]) -> _Self: 
        return self * other
    
    def __neg__(self) -> _Self:
        return self * -1
    
    def __truediv__(self, other: _Union[Numeric, _Self]) -> _Self:
        raise NotImplementedError("Division is currently not implemented for distributions.")
    
    def __rtruediv__(self, other: _Union[Numeric, _Self]) -> _Self:
        raise other * self ** -1
    
    def __pow__(self, other: _Union[Numeric, _Self]) -> _Self:
        raise NotImplementedError("Exponentiation is currently not implemented for distributions.")
     
    def __rpow__(self, other: _Union[Numeric, _Self]) -> _Self:
        raise NotImplementedError("Exponentiation is currently not implemented for distributions.")
    
    def __lt__(self, other: _Union[Numeric, _Self]) -> Event:        
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(-_np.inf, other))
        elif isinstance(other, Distribution):
            return self - other < 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __le__(self, other: _Union[Numeric, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.openclosed(-_np.inf, other))
        elif isinstance(other, Distribution):
            return self - other <= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __gt__(self, other: _Union[Numeric, _Self]) -> Event:
            
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other > 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ge__(self, other: _Union[Numeric, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.closedopen(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other >= 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __ne__(self, other: _Union[Numeric, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.open(-_np.inf, other) | _portion.open(other, _np.inf))
        elif isinstance(other, Distribution):
            return self - other != 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
    def __eq__(self, other: _Union[Numeric, _Self]) -> Event:
        if isinstance(other, (int, float)):
            return Event(self, _portion.singleton(other))
        elif isinstance(other, Distribution):
            return self - other == 0
        else:
            raise TypeError(f"Cannot compare objects of types {type(self)} and {type(other)}.")
    
class FormulaVariable(object):
    """A formula-like that supports formula writing."""
    
    def __init__(self, func: _Optional[NumericFunction]=None) -> None:
        """Create a formula variable, optionally with a func. Defaults to the identity function."""
        if func is None:
            func = lambda x: x
        self.func = func
           
    def __call__(self, other: _Self) -> NumericFunction:
        return self.func(other)
    
    def __add__(self, other: Numeric) -> _Self: 
        return FormulaVariable(lambda x: self.func(x) + other)
    
    def __radd__(self, other: Numeric) -> _Self: 
        return self + other
    
    def __sub__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: self.func(x) - other) 
    
    def __rsub__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: other - self.func(x)) 
    
    def __mul__(self, other: Numeric) -> _Self: 
        return FormulaVariable(lambda x: self.func(x) * other)
    
    def __rmul__(self, other: Numeric) -> _Self: 
        return self * other
    
    def __neg__(self) -> _Self:
        return self * -1
    
    def __truediv__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: self.func(x) / other)
    
    def __rtruediv__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: other / self.func(x))
    
    def __pow__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: self.func(x) ** other)
        
    def __rpow__(self, other: Numeric) -> _Self:
        return FormulaVariable(lambda x: other ** self.func(x))
    
class formula(object):
    """Decorator for converting functions to formulas."""
    
    def __init__(self, func: NumericFunction) -> None:
        """Convert func to a formula."""
        self.func = func
        
    def __call__(self, other: _Union[FormulaVariable, Distribution]) -> _Union[FormulaVariable, Distribution]:
        if isinstance(other, FormulaVariable):
            return FormulaVariable(lambda x: self.func(other(x)))
        elif isinstance(other, Distribution):
            return other.apply_func(self.func)
        else:
            raise TypeError(f"Formulas cannot be called on objects of class {type(other)}.")
    
class CustomDistribution(Distribution):
    """The base class for custom distributions, defined by a scipy rv. Do not instantiate this class."""
    
    options = None
    
    @_abc.abstractmethod
    def __init__(self, dist: _Union[_stats.rv_discrete, _stats.rv_continuous]) -> None:
        self._dist = dist
        
    def interpret_parameterization(self) -> None:
        pass
    
class DiscreteDistribution(Distribution):
    """The base class for discrete distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a - 1)
    
    def probability_at(self, a: float) -> float:
        """Calculate the probability P(X == a)."""
        return self.evaluate("pmf", a)
    
    def apply_func(self, func: NumericFunction, *, infinity_approximation: _Optional[int]=None, a: _Optional[int]=None, b: _Optional[int]=None) -> CustomDiscreteDistribution:
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
        
        return self.from_pfunc("pmf", _np.vectorize(lambda a: pmf.get(a, 0)), a, b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _Optional[str]=None, **kwargs) -> None:
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
    
    @classmethod    
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Numeric, b: Numeric) -> CustomDiscreteDistribution:
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
        setattr(NewScipyDiscreteDistribution, "_" + pfunc, staticmethod(func))
        return CustomDiscreteDistribution(NewScipyDiscreteDistribution(a=a, b=b))

    def apply_infix_operator(self, other: _Union[Numeric, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType = None) -> _Self:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return self.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
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
            return self.from_pfunc("pmf", _np.vectorize(lambda a: pmf.get(a, 0)), a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomDiscreteDistribution(CustomDistribution, DiscreteDistribution):
    """A custom discrete distribution."""
    
    def __init__(self, dist: _stats.rv_discrete) -> None:
        """Create a `CustomDiscreteDistribution` object given a `scipy.stats.rv_discrete` object."""
        self._dist = dist
    
class ContinuousDistribution(Distribution):
    """The base class for continuous distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a)
    
    def probability_at(self, a: float) -> float:
        """Calculate the probability P(X == a). Always returns 0."""
        _warnings.warn("Trying to calculate the point probability in a discrete distribution.")
        return 0
    
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: _Optional[float]=None, a: _Optional[float]=None, b: _Optional[float]=None) -> CustomContinuousDistribution:
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
        
        if len(inverse_funcs) == 0:
            inverse_func = _np.vectorize(lambda y: brentq(lambda x: func(x) - y, a=a0, b=b0))
        elif len(inverse_funcs) == 1:
            inverse_func = inverse_funcs[0]
        else:
            raise NotImplementedError("Multiple branched inverse functions not implemented yet. Use 1 to 1 functions.")
        
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
            
        return self.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: _Optional[str]=None, **kwargs) -> None:
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
        buffer = 0.2
        x = _np.linspace(a - diff * buffer, b + diff * buffer, int(diff * DEFAULTS["ratio"]))
        y = self.evaluate(pfunc, x)
        if color is None:
            color = DEFAULTS["default_color"]
        lines = _plt.plot(x, y, color=color)
        if not add:
            _plt.show()
    
    @classmethod    
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> CustomContinuousDistribution:
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
        setattr(NewScipyContinuousDistribution, "_" + pfunc, staticmethod(func))
        return CustomContinuousDistribution(NewScipyContinuousDistribution(a=a, b=b))
    
    def discretize(self) -> DiscreteDistribution:
        """Approximate the continuous distribution with a discrete distribution."""
        return DiscreteDistribution.from_pfunc("pmf", lambda x: self.probability_between(x - 0.5, x + 0.5), self.support.lower, self.support.upper)
    
    def apply_infix_operator(self, other: _Union[Numeric, _Self], op: _BuiltinFunctionType, inv_op: _BuiltinFunctionType) -> _Self:
        """Apply a binary infix operator. Avoid calling this function and use built-in operators instead."""
        if isinstance(other, (int, float)):
            a, b = self.support.lower, self.support.upper
            a2, b2 = sorted((op(a, other), op(b, other)))
            return self.from_pfunc("pmf", lambda x: self.evaluate("pmf", inv_op(x, other)), a2, b2)
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
                return self.from_pfunc("pdf", exact_pdf, a2, b2) 
            
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
            
            return self.from_pfunc("cdf", cdf, a2, b2)
        else:
            raise NotImplementedError(f"Binary operation between objects of type {type(self)} and {type(other)} is currently undefined.")
    
class CustomContinuousDistribution(CustomDistribution, ContinuousDistribution):
    """A custom custom distribution."""

    def __init__(self, dist: _stats.rv_continuous) -> None:
        """Create a `CustomContinuousDistribution` object given a `scipy.stats.rv_continuous` object."""
        self._dist = dist
        
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
    
def P(evt: Event) -> float:
    """Return the probability of an event occuring."""
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
        
        if isinstance(evt._tdist, ContinuousDistribution):
            probability += evt._tdist.probability_between(atomic.lower, atomic.upper)
        elif isinstance(evt._tdist, DiscreteDistribution):
            lower = atomic.lower
            if atomic.left == _portion.OPEN:
                lower += 1
            upper = atomic.upper
            if atomic.right == _portion.OPEN:
                upper -= 1
            probability += evt._tdist.probability_between(lower, upper)
    return probability
    
class UniformDiscreteDistribution(DiscreteDistribution):
    """A uniform discrete distribution."""
    
    options = [
        ["a", "b"], 
        ["low", "high"],
    ]
    
    def interpret_parameterization(self, parameters: _Dict[str, float]) -> _Type[_stats.randint]:
        if "a" in parameters and "b" in parameters:
            low = parameters.pop("a")
            high = parameters.pop("b") + 1
        elif "low" in parameters and "high" in parameters:
            low = parameters.pop("low")
            high = parameters.pop("high")
        return _stats.randint(low, high)
    
class BinomialDistribution(DiscreteDistribution):
    """A binomial distribution."""
    
    options = [
        ["n", "p"],
        ["n", "q"],
    ]
    
    def interpret_parameterization(self, parameters: _Dict[str, float]) -> _Union[_stats.rv_continuous, _stats.rv_discrete, None]:
        if "n" in parameters:
            n = parameters.pop("n")
            if "p" in parameters:
                p = parameters.pop("p")
            elif "q" in parameters:
                p = 1 - parameters.pop("q")
        return _stats.binom(n, p)
    
class UniformContinuousDistribution(ContinuousDistribution):
    """A uniform continuous distribution."""
    
    options = [
        ["a", "b"], 
        ["loc", "scale"],
    ]
    
    def interpret_parameterization(self, parameters: _Dict[str, float]) -> _Type[_stats.uniform]:
        if "a" in parameters and "b" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("b") - loc
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        return _stats.uniform(loc, scale)