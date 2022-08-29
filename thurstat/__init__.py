"""
Name
----
thurstat

Description
-----------
thurstat is a WIP easy-to-use univariate probability distribution module, mainly convenient wrappers for 
matplotlib.pyplot. numpy, and scipy.stats. Sometimes sacrifices accuracy for time and convenience, written in Python. 

Documentation
-------------
See https://github.com/nouturnsign/thurstat for documentation in README and source code. There are likely to be bugs.
"""

from ._tdist_base import (
    ContinuousDistribution, 
    DiscreteDistribution,
    CustomContinuousDistribution,
    CustomDiscreteDistribution, 
    Alias,
    Event,
    probability_of,
    P,
    formula,
)
from ._tdist_continuous import __all__ as continuous_tdist
from ._tdist_continuous import *
from ._tdist_discrete import __all__ as discrete_tdist
from ._tdist_discrete import *
from ._utils import PFUNC, update_defaults, display_added

__all__ = [
    # global
    "PFUNC", "update_defaults", 
    # base classes
    "DiscreteDistribution", "ContinuousDistribution",
    # custom equivalents of the base classes
    "CustomDiscreteDistribution", "CustomContinuousDistribution",
    # events and probability
    "Event", "probability_of", "P",
    # miscellaneous
    "Alias", "formula", "display_added",
] + continuous_tdist + discrete_tdist
