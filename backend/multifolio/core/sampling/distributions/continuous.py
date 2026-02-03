"""Continuous probability distributions."""

from typing import Union, Optional
import numpy as np
from scipy import stats

from multifolio.core.sampling.distributions.base import Distribution


class UniformDistribution(Distribution):
    """
    Continuous uniform distribution over [low, high).
    
    All values in the interval have equal probability.
    """
    
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize uniform distribution.
        
        Parameters
        ----------
        low : float, optional
            Lower bound (inclusive), by default 0.0
        high : float, optional
            Upper bound (exclusive), by default 1.0
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If low >= high
        """
        super().__init__(random_seed)
        
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        self.low = low
        self.high = high
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from uniform distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Samples uniformly distributed in [low, high)
        """
        return self._rng.uniform(low=self.low, high=self.high, size=size)
    
    def __repr__(self) -> str:
        return f"UniformDistribution(low={self.low}, high={self.high})"


class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution.
    
    Bell-shaped probability distribution characterized by mean and
    standard deviation.
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize normal distribution.
        
        Parameters
        ----------
        mean : float, optional
            Mean (center) of the distribution, by default 0.0
        std : float, optional
            Standard deviation (spread), by default 1.0
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If std <= 0
        """
        super().__init__(random_seed)
        
        if std <= 0:
            raise ValueError(f"std ({std}) must be positive")
        
        self.mean = mean
        self.std = std
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from normal distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Normally distributed samples
        """
        return self._rng.normal(loc=self.mean, scale=self.std, size=size)
    
    def __repr__(self) -> str:
        return f"NormalDistribution(mean={self.mean}, std={self.std})"


class TruncatedNormalDistribution(Distribution):
    """
    Truncated normal distribution.
    
    Normal distribution with specified bounds. Values outside [low, high]
    are never sampled.
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        low: float = -np.inf,
        high: float = np.inf,
        random_seed: Optional[int] = None
    ):
        """
        Initialize truncated normal distribution.
        
        Parameters
        ----------
        mean : float, optional
            Mean of the underlying normal distribution, by default 0.0
        std : float, optional
            Standard deviation of underlying normal, by default 1.0
        low : float, optional
            Lower bound (inclusive), by default -inf
        high : float, optional
            Upper bound (inclusive), by default inf
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If std <= 0 or low >= high
        """
        super().__init__(random_seed)
        
        if std <= 0:
            raise ValueError(f"std ({std}) must be positive")
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        
        # Calculate standardized bounds for scipy
        self._a = (low - mean) / std
        self._b = (high - mean) / std
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from truncated normal distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Samples from truncated normal distribution
        """
        # Use scipy's truncnorm which handles the truncation properly
        samples = stats.truncnorm.rvs(
            self._a,
            self._b,
            loc=self.mean,
            scale=self.std,
            size=size,
            random_state=self._rng
        )
        return samples
    
    def __repr__(self) -> str:
        return (f"TruncatedNormalDistribution(mean={self.mean}, std={self.std}, "
                f"low={self.low}, high={self.high})")


class BetaDistribution(Distribution):
    """
    Beta distribution over [low, high].
    
    Continuous distribution with shape parameters alpha and beta.
    Useful for modeling probabilities, proportions, and bounded continuous values.
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 2.0,
        low: float = 0.0,
        high: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize beta distribution.
        
        Parameters
        ----------
        alpha : float, optional
            First shape parameter, by default 2.0
        beta : float, optional
            Second shape parameter, by default 2.0
        low : float, optional
            Lower bound of distribution, by default 0.0
        high : float, optional
            Upper bound of distribution, by default 1.0
        random_seed : int or None, optional
            Random seed for reproducibility
        
        Raises
        ------
        ValueError
            If alpha <= 0, beta <= 0, or low >= high
        
        Notes
        -----
        Shape characteristics:
        - alpha = beta = 1: uniform distribution over [low, high]
        - alpha = beta > 1: symmetric bell-shaped
        - alpha < beta: skewed toward low
        - alpha > beta: skewed toward high
        
        The distribution is generated on [0, 1] and then scaled to [low, high].
        """
        super().__init__(random_seed)
        
        if alpha <= 0:
            raise ValueError(f"alpha ({alpha}) must be positive")
        if beta <= 0:
            raise ValueError(f"beta ({beta}) must be positive")
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        self.alpha = alpha
        self.beta = beta
        self.low = low
        self.high = high
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """
        Generate samples from beta distribution.
        
        Parameters
        ----------
        size : int or tuple of ints
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Samples in [low, high] from beta distribution
        """
        # Generate samples in [0, 1]
        samples_01 = self._rng.beta(a=self.alpha, b=self.beta, size=size)
        
        # Scale to [low, high]
        samples_scaled = self.low + samples_01 * (self.high - self.low)
        
        return samples_scaled
    
    def __repr__(self) -> str:
        return (f"BetaDistribution(alpha={self.alpha}, beta={self.beta}, "
                f"low={self.low}, high={self.high})")
