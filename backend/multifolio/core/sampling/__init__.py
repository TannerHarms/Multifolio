"""Statistical parameter sampling module."""

from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions.base import Distribution
from multifolio.core.sampling.correlation import (
    CorrelationStructure,
    GaussianCopula,
    CorrelationManager
)

__all__ = [
    "Sampler",
    "Distribution",
    "CorrelationStructure",
    "GaussianCopula",
    "CorrelationManager"
]
