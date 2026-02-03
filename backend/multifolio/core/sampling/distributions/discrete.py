"""Discrete probability distributions."""

from typing import Union, Optional
import numpy as np

from multifolio.core.sampling.distributions.base import Distribution


class ConstantDistribution(Distribution):
    """
    Constant (deterministic) distribution.
    
    Always returns the same value. Useful for fixed parameters in
    experimental designs.
    """
    
    def __init__(self, value: Union[int, float], random_seed: Optional[int] = None):
        """
        Initialize constant distribution.
        
        Parameters
        ----------
        value : int or float
            The constant value to return
        random_seed : int or None, optional
            Random seed (not used, but kept for interface consistency)
        """
        super().__init__(random_seed)
        self.value = value
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate constant values.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Array filled with the constant value
        """
        return np.full(size, self.value)
    
    def __repr__(self) -> str:
        return f"ConstantDistribution(value={self.value})"


class PoissonDistribution(Distribution):
    """
    Poisson distribution.
    
    Discrete distribution expressing probability of a given number of events
    occurring in a fixed interval. Common in count data.
    """
    
    def __init__(self, lam: float = 1.0, random_seed: Optional[int] = None):
        """
        Initialize Poisson distribution.
        
        Parameters
        ----------
        lam : float, optional
            Expected number of events (λ), by default 1.0
            Also equals both the mean and variance
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If lam <= 0
        
        Notes
        -----
        Returns non-negative integers. The probability of k events is:
        P(X=k) = (λ^k * e^(-λ)) / k!
        """
        super().__init__(random_seed)
        
        if lam <= 0:
            raise ValueError(f"lam ({lam}) must be positive")
        
        self.lam = lam
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from Poisson distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Integer counts from Poisson distribution
        """
        return self._rng.poisson(lam=self.lam, size=size)
    
    def __repr__(self) -> str:
        return f"PoissonDistribution(lam={self.lam})"


class UniformDiscreteDistribution(Distribution):
    """
    Discrete uniform distribution.
    
    Each integer value in [low, high] has equal probability.
    Useful for random selection from a finite set of integers.
    """
    
    def __init__(
        self,
        low: int = 0,
        high: int = 10,
        random_seed: Optional[int] = None
    ):
        """
        Initialize discrete uniform distribution.
        
        Parameters
        ----------
        low : int, optional
            Lower bound (inclusive), by default 0
        high : int, optional
            Upper bound (inclusive), by default 10
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If low >= high
        
        Notes
        -----
        Unlike continuous uniform, the upper bound is INCLUSIVE.
        So UniformDiscreteDistribution(0, 2) can return 0, 1, or 2.
        """
        super().__init__(random_seed)
        
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        self.low = low
        self.high = high
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from discrete uniform distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Integer samples uniformly distributed in [low, high]
        """
        # np.random.integers is inclusive of low, exclusive of high+1
        return self._rng.integers(low=self.low, high=self.high + 1, size=size)
    
    def __repr__(self) -> str:
        return f"UniformDiscreteDistribution(low={self.low}, high={self.high})"
