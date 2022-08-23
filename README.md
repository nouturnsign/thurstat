# thurstat

thurstat is a WIP easy-to-use univariate probability distribution module, mainly convenient wrappers for matplotlib.pyplot. numpy, and scipy.stats. Sometimes sacrifices accuracy for time and convenience, written in Python.

## Installation

Python >= 3.7, specifically with Google Colab usage in mind.

```bash
pip install git+https://github.com/nouturnsign/thurstat.git
```

## Usage

```py
from thurstat import *

# create independent random variables according to a distribution
X = UniformContinuousDistribution(a=0, b=1)
Y = UniformContinuousDistribution(a=0, b=1)

# perform arithmetic on random variables
Z = X - Y

# get summary statistics
print(Z.mean)
print(Z.variance)

# get 10 random values
print(Z.generate_random_values(10))

# evaluate probabilities of events
print(P(0.25 < Z < 0.5))

# display probability functions
Z.display("pdf")
```

## Recipes

Throughout, probability function will often be abbreviated as `pfunc`, function as `func`, scipy.stats distribution as `dist`, and thurstat distribution as `tdist` in code.

### Useful constants

To get probability functions, use the enum `PFUNC` e.g. `PFUNC.PDF`. Alternatively, use the commonly accepted abbreviation according to `scipy.stats` e.g. `"pdf"`.

### Predefined discrete distributions

```py
discrete_tdist = [
    "BinomialDistribution", "BernoulliDistribution", "BetaBinomialDistribution", 
    "BinomialDistribution", "GeometricDistribution", "HypergeometricDistribution", 
    "NegativeBinomialDistribution", "NegativeHypergeometricDistribution", 
    "PoissonDistribution", "SkellamDistribution", "UniformDiscreteDistribution", 
    "YuleSimonDistribution", "ZipfDistribution", "ZipfianDistribution"
]
```

### Predefined continuous distributions

```py
continuous_tdist = [
    "BetaDistribution", "BetaPrimeDistribution", "CauchyDistribution", 
    "ChiDistribution", "ChiSquaredDistribution", "CosineDistribution", 
    "ErlangDistribution", "ExponentialDistribution", "FDistribution", 
    "GammaDistribution", "GompertzDistribution", "LaplaceDistribution", 
    "LogisticDistribution", "NormalDistribution", "TDistribution", 
    "TrapezoidalDistribution", "TriangularDistribution", 
    "UniformContinuousDistribution", "WeibullDistribution"
]
```

### Creating distribution objects

When creating a new distribution, each distribution is assumed to be univariate, independent, and not mixed (i.e. either only discrete or only continuous). 

Use common names for parameters; all parameters must be named. Any parameters that are keywords (e.g. lambda) should be appended with an underscore (e.g. lambda_). Any parameters with subscripts (x_0) should be input without an underscore in between the variable name and subscript (e.g. x0). Parameters with superscripts (e.g. sigma^2) are not accepted; other names in place of these superscript parameters might be (e.g. variance). If the parameters are invalid, a `ParameterValidationError` will be raised, and should display acceptable options. When in doubt, consult `scipy.stats` or Wikipedia. If not `None`, the `options` class attribute will contain acceptable parameterizations. 

### Useful attributes

Some distributions will have additional attributes. Those are meant for internal use, representing the parameters based on one of the characterizations. The attributes given below are for all distributions.

```
support
median
mean
variance
skewness
kurtosis
standard_deviation
```

### Useful methods

See docstrings for additional arguments. The methods given below are for all distributions except `discretize`.

```
generate_random_values(n)
probability_between(a, b)
probability_at(a)
evaluate(pfunc, at)
expected_value(func)
moment(n)
apply_func(func)
display(pfunc)
discretize() # ContinuousDistribution only
```

### Useful classmethods

Use `to_alias` with any distribution class, and use `from_pfunc` on `Custom...Distribution`. Note that inputting a `"pdf"` for a `CustomContinuousDistribution` will mean that scipy has to integrate the `"pdf"` to calculate the `"cdf"`. If inputting the `"pdf"` is too slow, try integrating by hand or with other software first, then using `from_pfunc` on the `"cdf"`. See below for more on aliases.

```
to_alias(*interpret_parameters)
from_pfunc(pfunc, func, a, b)
```

### Useful constructors

```
formula()
...Distribution(**parameters)
Custom...Distribution(dist)
Alias(Type[tdist], *interpret_parameters)
Event(tdist, interval)
```

### Custom distributions

Use the constructor or classmethod `from_pfunc` with the custom type of desired distribution. Alternatively, create a new class inheriting from the desired type of distribution and implement method `interpret_parameters` to return a `rv_frozen` object. This allows for creating a general class of distribution with acceptable parameters instead of a new custom distribution every time the parameters are changed. Try to minimize the number of infinity-like expressions as these will cause floating point overflow (i.e. the number is too large).

e.g.
```py
import scipy.stats
from numpy import pi

# constructor
X = CustomContinuousDistribution(scipy.stats.cauchy())

# from_pfunc
X = CustomContinuousDistribution.from_pfunc("pdf", lambda x: 1 / (pi * (1 + x ** 2)), a=0, b=1)

# inheritance
class CauchyDistribution(ContinuousDistribution):
    
    # optionally define options as a list of acceptable parameter names; used in ParameterValidationError
    options = [
        ["x0", "gamma"],
        ["loc", "scale"],
    ]
    
    # implement this method
    # pop parameters and assign to values for parameter validation
    def interpret_parameterization(self, parameters):
        if "x0" in parameters and "gamma" in parameters:
            loc = parameters.pop("x0")
            scale = parameters.pop("gamma")
        elif "loc" in parameters and "scale" in parameters:
            loc = parameters.pop("loc")
            scale = parameters.pop("scale")
        return scipy.stats.cauchy(loc, scale)

X = CauchyDistribution(x0=0, gamma=1)
```

### Apply function

Use the method `apply_func` on the desired distribution. Add additional positional arguments as inverse functions for noninvertible functions on continuous distributions.

e.g.
```py
Y = UniformDiscreteDistribution(a=1, b=6).apply_func(lambda x: x ** 2)
Z = UniformContinuousDistribution(a=0, b=1).apply_func(abs, lambda x: x, lambda x: -x)
```

### formula

Write formulas without lambda expressions with formula-like notation. Formulas can be called on numeric inputs, distributions, and on each other. Note that formulas are immutable, so modifying formulas creates new `formula` objects instead of modifying the original object. 

e.g.
```py
# using formula constructor, only needed once
x = formula()

# using decorator to create functions
# the parameter can be named anything, does not have to be x
@formula
def sqrt(x):
    return x ** 0.5
X = CustomContinuousDistribution.from_pfunc("pdf", 1.5 * sqrt(x), a=0, b=1)

# using formulas on distributions
@formula
def square(x):
    return x ** 2
Y = square(UniformDiscreteDistribution(a=1, b=6))
```

### Random Variable Arithmetic

Perform addition, subtraction, and multiplication on distributions with other distributions and numerics. Limited support for exponentiation. Division only works between continuous distributions and numbers currently and must be implemented through `apply_func`. Note that for continuous distributions, the accuracy of the approximation is unknown but so far seems to not be too large; it is possible that the error be too large. Allow for as long as a minute, depending on the distribution.

Each distribution is assumed to be different, so `X + X != 2 * X`. 

e.g.
```py
X = UniformContinuousDistribution(a=0, b=1)
Y = UniformContinuousDistribution(a=0, b=1)

Z = 2 * (X - Y)
```

### Alias

Instead of forcibly naming arguments every time you wish to instantiate a class, you may also create an alias using positional arguments as the characterization. Creating an `Alias` does not check for the validity of the characterization, so it is possible to create an invalid Alias without error.

e.g.
```py
# using classmethod to_alias
B = BinomialDistribution.to_alias("n", "p")
# using Alias constructor
B = Alias(BinomialDistribution, "n", "p")

# creating an object using an Alias
X = B(10, 0.2)
```

### Events and probability

A function `P` is defined for probability-like notation. `P` aliases the function `probability_of` in case `P` is already defined as a constant. Comparisons with `Distribution` objects are implemented to return `Event` objects. Avoid using the `Event` constructor. Note that multiple inequality comparisons are not truly multiple inequality comparisons, so avoid comparing distributions within multiple inequalities and expressions with more than two inequalities. Below is an example of an acceptable multiple inequality (using <=, >, and >= are all acceptable as well).

e.g.
```py
X = UniformContinuousDistribution(a=0, b=1)
Y = UniformContinuousDistribution(a=0, b=1)

# simple comparisons are supported between floats and distributions
print(P(X < 0.4))
print(P(X < Y))

# multiple inequality comparisons are also supported
print(P(0.2 < X < 0.4))
```

### Modify defaults

For convenience, some defaults are assumed to be good values but can be changed using `update_defaults`. Namely, the following defaults can be changed:

- infinity_approximation: large enough to be considered infinity, defaults to `1e6`
- exact: whether or not to use approximations in continuous random variable arithmetic, defaults to `False`
- ratio: the ratio of points plotted to distance between endpoints when displaying, defaults to `200`
- buffer: the additional percent of the width to be plotted to both the left and right of `display`, defaults to `0.2`
- default_color: default [`matplotlib` color](https://matplotlib.org/stable/gallery/color/named_colors.html#list-of-named-colors) to be used when plotting, defaults to `"C0"`
- local_seed: the numeric value of the seed when calling any function or None if no local seed, defaults to `None`
- global_seed: the numeric value of the seed singleton to be set at the beginning or None if no global seed, defaults to `None`
- warnings: the warning level to be displayed according to [Python's `warning` module](https://docs.python.org/3/library/warnings.html#the-warnings-filter), defaults to `"default"`

Setting exact to `True` will make random variable arithmetic prohibitively slow on general continuous distributions.

Setting local seed will mean that calling `generate_random_values` on the same distribution will result in the same sequence of values. Setting global seed will mean that calling `generate_random_values` on the same distribution will start from the same sequence of values but keep progressing through the seed.

e.g.
```py
update_defaults(warnings="ignore", infinity_approximation=1e3)
```

## Future goals

- implement division and exponentiation (in the general case) on distributions
- implement faster addition, subtraction, and multiplication (in specific cases) on certain distributions
- implement multivariate distributions and create dependent distributions
- implement conditional probability and events
- assess and improve accuracy of approximations for continuous random variable arithmetic
- implement event arithmetic (intersection, union)
- implement model fitting
- add more distributions
- add tests

## License

Currently no license. `thurstat` is still very WIP and not meant for use yet.
