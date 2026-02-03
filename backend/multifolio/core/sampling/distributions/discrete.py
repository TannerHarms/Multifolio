"""Discrete probability distributions."""

from typing import Optional, Union, Dict, Callable, Literal
import numpy as np
from pathlib import Path
import pandas as pd

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


class CustomDiscreteDistribution(Distribution):
    """
    Custom discrete distribution from data, probabilities, or function.
    
    Can be created from:
    - Dict mapping values to probabilities
    - Array/list of observed values (counts frequencies)
    - CSV/text file with data or probability mapping
    - Custom function that generates samples
    
    Parameters
    ----------
    data : dict, np.ndarray, list, str, Path, or callable, optional
        Source of distribution:
        - dict: Maps values to probabilities (must sum to 1)
        - Array/list: Observed data to count frequencies
        - str/Path: Path to CSV file with data or probabilities
        - callable: Function that returns samples when called with size parameter
    negative_handling : {'truncate', 'shift', 'allow'}, optional
        How to handle negative values:
        - 'truncate': Replace negative values with 0
        - 'shift': Shift entire distribution so minimum is 0
        - 'allow': Allow negative values (default)
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> # From probability dict
    >>> probs = {1: 0.2, 2: 0.3, 3: 0.5}
    >>> dist = CustomDiscreteDistribution(data=probs)
    >>> samples = dist.sample(size=100)
    >>> 
    >>> # From observed data
    >>> data = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
    >>> dist = CustomDiscreteDistribution(data=data)
    >>> 
    >>> # From custom function
    >>> def my_sampler(size):
    ...     return np.random.choice([1, 2, 3], size=size, p=[0.2, 0.3, 0.5])
    >>> dist = CustomDiscreteDistribution(data=my_sampler)
    """
    
    def __init__(
        self,
        data: Union[Dict, np.ndarray, list, str, Path, Callable] = None,
        negative_handling: Literal['truncate', 'shift', 'allow'] = 'allow',
        random_seed: Optional[int] = None
    ):
        super().__init__(random_seed)
        
        if data is None:
            raise ValueError("Must provide data source")
        
        if negative_handling not in ['truncate', 'shift', 'allow']:
            raise ValueError(
                f"negative_handling must be 'truncate', 'shift', or 'allow', "
                f"got '{negative_handling}'"
            )
        
        self.negative_handling = negative_handling
        self._shift_amount = 0
        
        # Determine type of data source
        if callable(data):
            self._sample_function = data
            self._values = None
            self._probabilities = None
            self._source_type = 'function'
        else:
            self._sample_function = None
            self._source_type = 'data'
            
            # Load and process data
            if isinstance(data, dict):
                self._process_probability_dict(data)
            elif isinstance(data, (str, Path)):
                data = self._load_from_file(data)
                self._process_observed_data(data)
            else:
                self._process_observed_data(np.asarray(data))
    
    def _load_from_file(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load data from CSV or text file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Try reading as CSV
            df = pd.read_csv(filepath)
            
            # Check if it's a probability table (2 columns: value, probability)
            if len(df.columns) == 2:
                values = df.iloc[:, 0].values
                probs = df.iloc[:, 1].values
                
                # If looks like probability table
                if np.abs(probs.sum() - 1.0) < 0.01:
                    prob_dict = dict(zip(values, probs))
                    self._process_probability_dict(prob_dict)
                    return None
            
            # Otherwise treat as observed data
            data = df.iloc[:, 0].values
        except Exception:
            # Try reading as plain text
            data = np.loadtxt(filepath)
        
        return data
    
    def _process_probability_dict(self, prob_dict: Dict):
        """Process dictionary of value->probability mappings."""
        # Validate probabilities sum to 1
        total_prob = sum(prob_dict.values())
        if not np.isclose(total_prob, 1.0):
            raise ValueError(
                f"Probabilities must sum to 1, got {total_prob}. "
                f"Provide either exact probabilities or observed data."
            )
        
        values = np.array(list(prob_dict.keys()))
        probabilities = np.array(list(prob_dict.values()))
        
        # Handle negative values
        if self.negative_handling != 'allow':
            min_val = values.min()
            if min_val < 0:
                if self.negative_handling == 'truncate':
                    # Remove negative values and renormalize
                    mask = values >= 0
                    values = values[mask]
                    probabilities = probabilities[mask]
                    probabilities = probabilities / probabilities.sum()
                elif self.negative_handling == 'shift':
                    self._shift_amount = -min_val
                    values = values + self._shift_amount
        
        self._values = values
        self._probabilities = probabilities
    
    def _process_observed_data(self, data: np.ndarray):
        """Process observed data to compute frequencies."""
        if len(data) == 0:
            raise ValueError("Data array is empty")
        
        # Ensure integer values for discrete distribution
        data = np.round(data).astype(int)
        
        # Handle negative values
        if self.negative_handling == 'truncate':
            data = data[data >= 0]
            if len(data) == 0:
                raise ValueError("No non-negative values after truncation")
        elif self.negative_handling == 'shift':
            min_val = data.min()
            if min_val < 0:
                self._shift_amount = -min_val
                data = data + self._shift_amount
        
        # Count frequencies
        unique_values, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()
        
        self._values = unique_values
        self._probabilities = probabilities
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from custom discrete distribution."""
        if self._sample_function is not None:
            # Use custom function
            samples = self._sample_function(size)
            samples = np.asarray(samples)
        else:
            # Use probability distribution
            samples = self._rng.choice(
                self._values,
                size=size,
                p=self._probabilities
            )
        
        # Handle negatives in output
        if self.negative_handling == 'truncate':
            samples = np.maximum(samples, 0)
        
        return samples
    
    def __repr__(self) -> str:
        if self._source_type == 'function':
            return f"CustomDiscreteDistribution(source=function, negative_handling='{self.negative_handling}')"
        else:
            n_values = len(self._values) if self._values is not None else 0
            return f"CustomDiscreteDistribution(source=data, n_values={n_values}, negative_handling='{self.negative_handling}')"
