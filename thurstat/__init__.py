import abc
from typing import Callable, Dict, List, NamedTuple, Optional, Type, Union
from typing_extensions import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import brentq, minimize_scalar

__all__ = [
    # global
    "pfunc",
    "update_config",
    # formula
    "Formula", "formula", "x",
    # base classes
    "Distribution", "DiscreteDistribution", "ContinuousDistribution",
    # instantiable equivalents of the base classes
    "CustomDistribution", "CustomDiscreteDistribution", "CustomContinuousDistribution",
    # predefined discrete distributions
    "UniformDiscreteDistribution",
    # predefined continuous distributions
    "UniformContinuousDistribution",
]

Numeric = Union[int, float]
NumericFunction = Callable[[Numeric], Numeric]
ProbabilityFunction = Literal["pdf", "pmf", "cdf", "sf", "ppf", "isf"]

class pfunc(NamedTuple):
    """Acceptable pfunc abbreviations."""
    PDF: str = "pdf"
    PMF: str = "pmf"
    CDF: str = "cdf"
    SF : str = "sf"
    PPF: str = "ppf"
    ISF: str = "isf"

class Formula(object):
    """A formula-like that supports formula writing."""
    
    def __init__(self, func: Optional[NumericFunction]=None) -> None:
        """Create a formula, optionally with a func. Defaults to the identity function."""
        if func is None:
            func = lambda x: x
        self.func = func
           
    def __call__(self, other: Self) -> NumericFunction:
        return self.func(other)
    
    def __add__(self, other: Numeric) -> Self: 
        return Formula(lambda x: self.func(x) + other)
    
    def __radd__(self, other: Numeric) -> Self: 
        return self + other
    
    def __sub__(self, other: Numeric) -> Self:
        return Formula(lambda x: self.func(x) - other) 
    
    def __rsub__(self, other: Numeric) -> Self:
        return Formula(lambda x: other - self.func(x)) 
    
    def __mul__(self, other: Numeric) -> Self: 
        return Formula(lambda x: self.func(x) * other)
    
    def __rmul__(self, other: Numeric) -> Self: 
        return self * other
    
    def __neg__(self) -> Self:
        return self * -1
    
    def __truediv__(self, other: Numeric) -> Self:
        return Formula(lambda x: self.func(x) / other)
    
    def __rtruediv__(self, other: Numeric) -> Self:
        return Formula(lambda x: other / self.func(x))
    
    def __pow__(self, other: Numeric) -> Self:
        return Formula(lambda x: self.func(x) ** other)
        
    def __rpow__(self, other: Numeric) -> Self:
        return Formula(lambda x: other ** self.func(x))
    
class formula(object):
    """Decorator for converting functions to formulas."""
    
    def __init__(self, func: NumericFunction) -> None:
        """Convert func to a formula."""
        self.func = func
        
    def __call__(self, other: Formula) -> Formula:
        return Formula(lambda x: self.func(other(x)))
    
x = Formula()

config = {
    "infinity_approximation": 1e6,
    "default_color": "C0",
    "local_seed": None,
    "global_seed": None,
}

def update_config(**kwargs):
    """Update the global config."""
    
    config.update(kwargs)
    if "global_seed" in kwargs:
        config["global_seed"] = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(kwargs["global_seed"])))

class ParameterValidationError(Exception):
    """Raised when an invalid set of parameters are used to instantiate a Distribution."""
    
    def __init__(self, given: List[str], options: Optional[List[List[str]]]=None):
        """Create the error with the given parameters and optional parameters."""
        message = f"Failed to validate parameters. Given: {given}."
        if options is not None:
            message += f" Options: {options}."
        super().__init__(message)

class Distribution(abc.ABC):
    """The base class for a distribution. Do not instantiate this class."""
    
    options: Optional[List[List[str]]]
    
    def __init__(self, **parameters: float) -> None:
        """Create a distribution object given the parameters as named keyword arguments."""
        given = list(parameters.keys())
        try:
            self._dist = self.interpret_parameterization(parameters)
        except UnboundLocalError:
            raise ParameterValidationError(given, self.options)
        if len(parameters) > 0:
            raise ParameterValidationError(given, self.options)
        self.support = self._dist.support()
        self.median = self._dist.median()
        self.mean, self.variance, self.skewness, self.kurtosis = self._dist.stats(moments="mvsk")
        self.standard_deviation = self._dist.std()
    
    @abc.abstractmethod  
    def interpret_parameterization(self, parameters: Dict[str, float]) -> Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, None]:
        pass
    
    def generate_random_values(self, n: int) -> np.ndarray:
        """Generate n random values."""
        if config["local_seed"] is not None:
            seed = config["local_seed"]
        else:
            seed = config["global_seed"]
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
    
    @abc.abstractmethod
    def probability_between(self, a: float, b: float) -> float:
        pass
    
    @abc.abstractmethod
    def probability_at(self, a: float) -> float:
        pass
    
    @abc.abstractmethod
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction) -> Union["CustomDiscreteDistribution", "CustomContinuousDistribution"]:
        pass
    
    @abc.abstractmethod
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs) -> None:
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Numeric, b: Numeric) -> Type[Self]:
        pass
    
class CustomDistribution(Distribution):
    """The base class for custom distributions, defined by a scipy rv. Do not instantiate this class."""
    
    options = None
    
    @abc.abstractmethod
    def __init__(self, dist: Union[scipy.stats.rv_discrete, scipy.stats.rv_continuous]) -> None:
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
    
    def apply_func(self, func: NumericFunction) -> "CustomDiscreteDistribution":
        """Apply a function to the distribution to create a new distribution."""
        a, b = self._dist.support()
        x = np.arange(a, b + 1)
        y = self.evaluate("pmf", x)
        
        x_transform = func(x)
        pmf = {}
        for e, p in zip(x_transform, y):
            pmf[e] = pmf.get(e, 0) + p
            
        a = min(x_transform)
        b = max(x_transform)
        
        return self.from_pfunc("pmf", np.vectorize(lambda a: pmf.get(a, 0)), a, b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs) -> None:
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
        a, b = self._dist.support()
        if a == -np.inf:
            a = self.evaluate("ppf", 1 / config["infinity_approximation"])
        if b == np.inf:
            b = self.evaluate("ppf", 1 - 1 / config["infinity_approximation"])
        x = np.arange(a, b + 1)
        y = self.evaluate(pfunc, x)
        markerline, stemlines, baseline = plt.stem(x, y, basefmt=" ", use_line_collection=True, **kwargs)
        if color is None:
            color = config["default_color"]
        markerline.set_color(color)
        stemlines.set_color(color)
        if not add:
            plt.show()
    
    @classmethod    
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Numeric, b: Numeric) -> "CustomDiscreteDistribution":
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
        setattr(NewScipyDiscreteDistribution, "_" + pfunc, staticmethod(func))
        return CustomDiscreteDistribution(NewScipyDiscreteDistribution(a=a, b=b))
    
class CustomDiscreteDistribution(CustomDistribution, DiscreteDistribution):
    """A custom discrete distribution."""
    
    def __init__(self, dist: scipy.stats.rv_discrete) -> None:
        """Create a `CustomDiscreteDistribution` object given a `scipy.stats.rv_discrete` object."""
        self._dist = dist
        self.support = self._dist.support()
        self.median = self._dist.median()
        self.mean, self.variance, self.skewness, self.kurtosis = self._dist.stats(moments="mvsk")
        self.standard_deviation = self._dist.std()
    
class ContinuousDistribution(Distribution):
    """The base class for continuous distributions. Do not instantiate this class."""
    
    def probability_between(self, a: float, b: float) -> float:
        """Calculate the probability P(a <= X <= b), including both a and b."""
        return self.evaluate("cdf", b) - self.evaluate("cdf", a)
    
    def probability_at(self, a: float) -> float:
        """Calculate the probability P(X == a). Always returns 0."""
        return 0
    
    def apply_func(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: Optional[float]=None, a: Optional[float]=None, b: Optional[float]=None) -> "CustomContinuousDistribution":
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
        a0, b0 = self._dist.support()
        
        if len(inverse_funcs) == 0:
            inverse_func = np.vectorize(lambda y: brentq(lambda x: func(x) - y, a=a0, b=b0))
        elif len(inverse_funcs) == 1:
            inverse_func = inverse_funcs[0]
        else:
            raise NotImplementedError("Multiple branched inverse functions not implemented yet. Use 1 to 1 functions.")
        
        if (infinity_approximation is None) and (a is None) and (b is None):
            infinity_approximation = config["infinity_approximation"]
        
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
            
        return self.from_pfunc("cdf", lambda y: self.evaluate("cdf", inverse_func(y)), a=a, b=b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs) -> None:
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
        a, b = self._dist.support()
        if a == -np.inf:
            a = self.evaluate("ppf", 1 / config["infinity_approximation"])
        if b == np.inf:
            b = self.evaluate("ppf", 1 - 1 / config["infinity_approximation"])
        diff = b - a
        buffer = 0.2
        x = np.linspace(a - diff * buffer, b + diff * buffer, int(diff * 200))
        y = self.evaluate(pfunc, x)
        if color is None:
            color = config["default_color"]
        lines = plt.plot(x, y, color=color)
        if not add:
            plt.show()
    
    @classmethod    
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: float, b: float) -> "CustomContinuousDistribution":
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
        setattr(NewScipyContinuousDistribution, "_" + pfunc, staticmethod(func))
        return CustomContinuousDistribution(NewScipyContinuousDistribution(a=a, b=b))
    
class CustomContinuousDistribution(CustomDistribution, ContinuousDistribution):
    """A custom custom distribution."""

    def __init__(self, dist: scipy.stats.rv_continuous) -> None:
        """Create a `CustomContinuousDistribution` object given a `scipy.stats.rv_continuous` object."""
        self._dist = dist
        self.support = self._dist.support()
        self.median = self._dist.median()
        self.mean, self.variance, self.skewness, self.kurtosis = self._dist.stats(moments="mvsk")
        self.standard_deviation = self._dist.std()
    
class UniformDiscreteDistribution(DiscreteDistribution):
    """A uniform discrete distribution."""
    
    options = [
        ["a", "b"], 
        ["low", "high"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> Type[scipy.stats.randint]:
        if "a" in parameters and "b" in parameters:
            low = parameters.pop("a")
            high = parameters.pop("b") + 1
        elif "low" in parameters and "high" in parameters:
            low = parameters.pop("low")
            high = parameters.pop("high")
        return scipy.stats.randint(low, high)
    
class UniformContinuousDistribution(ContinuousDistribution):
    """A uniform continuous distribution."""
    
    options = [
        ["a", "b"], 
        ["loc", "scale"],
    ]
    
    def interpret_parameterization(self, parameters: Dict[str, float]) -> Type[scipy.stats.uniform]:
        if "a" in parameters and "b" in parameters:
            loc = parameters.pop("a")
            scale = parameters.pop("b") - loc
        if "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        return scipy.stats.uniform(loc, scale)