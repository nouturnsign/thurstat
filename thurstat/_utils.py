import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

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
