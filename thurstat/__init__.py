# Base distribution classes: 
# Distribution
# DiscreteDistribution
# ContinuousDistribution

# Distribution
#
# implements methods:
# interpret_parameterization
# generate_random_values
# evaluate
# expected_value
# moment
# probability_between
# apply_transform
# display
#
# implements class methods:
# from_pfunc

# DiscreteDistribution
#
# implements methods:
# probability_at

# Future goals:
# basic arithmetic operations on distributions and equality
# classes Event, ProbabilityOf (callable object P)

# assuming Python version >= 3.7
import abc
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import brentq, minimize_scalar
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Type, Union
from typing_extensions import Literal, Self

pfunc = SimpleNamespace(PDF="pdf", PMF="pmf", CDF="cdf", SF="sf", PPF="ppf", ISF="ISF")
NumericFunction = Callable[[float], float]
ProbabilityFunction = Literal["pdf", "pmf", "cdf", "sf", "ppf", "isf"]

config = {
    "infinity_approximation": 1e6,
    "default_color": "C0",
    "local_seed": None,
    "global_seed": None,
}

def update_config(**kwargs):
    
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
    
    options: Optional[List[List[str]]]
    
    def __init__(self, **parameters: float) -> None:
        given = list(parameters.keys())
        self._dist = self.interpret_parameterization(parameters)
        if len(parameters) > 0:
            raise ParameterValidationError(given, self.options)
        self.median = self._dist.median()
        self.mean, self.variance, self.skewness, self.kurtosis = self._dist.stats(moments="mvsk")
        self.standard_deviation = self._dist.std()
    
    @abc.abstractmethod  
    def interpret_parameterization(self, parameters: Dict[str, float]) -> Union[scipy.stats.rv_continuous, scipy.stats.rv_discrete, None]:
        pass
    
    def generate_random_values(self, n: int) -> np.ndarray:
        if config["local_seed"] is not None:
            seed = config["local_seed"]
        else:
            seed = config["global_seed"]
        return self._dist.rvs(n, random_state=seed)
    
    def evaluate(self, pfunc: ProbabilityFunction, at: float) -> float:
        return getattr(self._dist, pfunc)(at)
    
    def expected_value(self, func: NumericFunction) -> float:
        return self._dist.expect(func)
    
    def moment(self, n: int) -> float:
        return self._dist.moment(n)
    
    @abc.abstractmethod
    def probability_between(self, a: float, b: float) -> float:
        pass
    
    @abc.abstractmethod
    def probability_at(self, a: float) -> float:
        pass
    
    @abc.abstractmethod
    def apply_transform(self, func: NumericFunction, *inverse_funcs: NumericFunction) -> Union["CustomDiscreteDistribution", "CustomContinuousDistribution"]:
        pass
    
    @abc.abstractmethod
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs) -> None:
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Union[int, float], b: Union[int, float]) -> Type[Self]:
        pass
    
class CustomDistribution(Distribution):
    
    options = None
    
    @abc.abstractmethod
    def __init__(self, dist: Union[scipy.stats.rv_discrete, scipy.stats.rv_continuous]) -> None:
        self._dist = dist
        
    def interpret_parameterization(self) -> None:
        pass
    
class DiscreteDistribution(Distribution):
    
    def probability_between(self, a: float, b: float) -> float:
        return self.evaluate("cdf", b) - self.evaluate("cdf", a - 1)
    
    def probability_at(self, a: float) -> float:
        return self.evaluate("pmf", a)
    
    def apply_transform(self, func: NumericFunction) -> "CustomDiscreteDistribution":
        a, b = self._dist.support()
        x = np.arange(a, b + 1)
        y = self.evaluate("pmf", x) # see display
        
        x_transform = func(x)
        pmf = {}
        for e, p in zip(x_transform, y):
            pmf[e] = pmf.get(e, 0) + p
            
        a = min(x_transform)
        b = max(x_transform)
        
        return self.from_pfunc("pmf", np.vectorize(lambda a: pmf.get(a, 0)), a, b)
    
    def display(self, pfunc: ProbabilityFunction, add: bool=False, color: Optional[str]=None, **kwargs) -> None:
        a, b = self._dist.support()
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
    def from_pfunc(cls, pfunc: ProbabilityFunction, func: NumericFunction, a: Union[int, float], b: Union[int, float]) -> "CustomDiscreteDistribution":
        class NewScipyDiscreteDistribution(scipy.stats.rv_discrete): pass
        setattr(NewScipyDiscreteDistribution, "_" + pfunc, staticmethod(func))
        return CustomDiscreteDistribution(NewScipyDiscreteDistribution(a=a, b=b))
    
class CustomDiscreteDistribution(CustomDistribution, DiscreteDistribution):
    
    def __init__(self, dist: scipy.stats.rv_discrete) -> None:
        self._dist = dist
    
class ContinuousDistribution(Distribution):
    
    def probability_between(self, a: float, b: float) -> float:
        return self.evaluate("cdf", b) - self.evaluate("cdf", a)
    
    def probability_at(self, a: float) -> float:
        return 0
    
    def apply_transform(self, func: NumericFunction, *inverse_funcs: NumericFunction, infinity_approximation: Optional[float]=None, a: Optional[float]=None, b: Optional[float]=None) -> "CustomContinuousDistribution":
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
        a, b = self._dist.support()
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
        class NewScipyContinuousDistribution(scipy.stats.rv_continuous): pass
        setattr(NewScipyContinuousDistribution, "_" + pfunc, staticmethod(func))
        return CustomContinuousDistribution(NewScipyContinuousDistribution(a=a, b=b))
    
class CustomContinuousDistribution(CustomDistribution, ContinuousDistribution):

    def __init__(self, dist: scipy.stats.rv_continuous) -> None:
        self._dist = dist
    
class UniformDiscreteDistribution(DiscreteDistribution):
    
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