"""Base distribution class and protocol for sampling."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class Distribution(ABC):
    """
    Abstract base class for probability distributions.
    
    All distribution implementations must inherit from this class and implement
    the sample() method. This provides a consistent interface for generating
    samples from various probability distributions.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize distribution with optional random seed.
        
        Parameters
        ----------
        random_seed : int or None, optional
            Random seed for reproducibility. If None, results will vary.
        """
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
    
    @abstractmethod
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from the distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate. Can be:
            - int: generate 1D array of samples
            - tuple: generate multi-dimensional array
        
        Returns
        -------
        np.ndarray
            Array of samples from the distribution
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the distribution."""
        pass
    
    def reset_seed(self, random_seed: Optional[int] = None) -> None:
        """
        Reset the random number generator with a new seed.
        
        Parameters
        ----------
        random_seed : int or None, optional
            New random seed. If None, generator will produce different results.
        """
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
