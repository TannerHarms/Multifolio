"""Multi-parameter sampler for experimental design."""

from typing import Dict, List, Optional, Union, Literal
import numpy as np
import pandas as pd
from scipy.stats import qmc

from multifolio.core.sampling.distributions.base import Distribution


class Sampler:
    """
    Multi-parameter sampler for generating experimental parameter sets.
    
    Allows defining multiple parameters, each with their own distribution,
    and generating coordinated samples across all parameters.
    
    Examples
    --------
    >>> from multifolio.core.sampling import Sampler
    >>> from multifolio.core.sampling.distributions import (
    ...     UniformDistribution, NormalDistribution
    ... )
    >>> 
    >>> sampler = Sampler(random_seed=42)
    >>> sampler.add_parameter('temperature', UniformDistribution(low=20, high=100))
    >>> sampler.add_parameter('pressure', NormalDistribution(mean=1.0, std=0.1))
    >>> samples = sampler.generate(n=100)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize sampler.
        
        Parameters
        ----------
        random_seed : int or None, optional
            Random seed for reproducibility. If provided, all added
            distributions will use derived seeds for reproducibility.
        """
        self.random_seed = random_seed
        self._parameters: Dict[str, Distribution] = {}
        self._parameter_order: List[str] = []
    
    def add_parameter(
        self,
        name: str,
        distribution: Distribution
    ) -> "Sampler":
        """
        Add a parameter with its distribution.
        
        Parameters
        ----------
        name : str
            Parameter name (must be unique)
        distribution : Distribution
            Distribution to sample from for this parameter
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Raises
        ------
        ValueError
            If parameter name already exists
        
        Examples
        --------
        >>> sampler = Sampler()
        >>> sampler.add_parameter('x', UniformDistribution(0, 1))
        >>> sampler.add_parameter('y', NormalDistribution(mean=5, std=2))
        """
        if name in self._parameters:
            raise ValueError(f"Parameter '{name}' already exists")
        
        self._parameters[name] = distribution
        self._parameter_order.append(name)
        
        return self  # Enable method chaining
    
    def remove_parameter(self, name: str) -> "Sampler":
        """
        Remove a parameter.
        
        Parameters
        ----------
        name : str
            Parameter name to remove
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Raises
        ------
        KeyError
            If parameter doesn't exist
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        
        del self._parameters[name]
        self._parameter_order.remove(name)
        
        return self
    
    def list_parameters(self) -> List[str]:
        """
        Get list of parameter names.
        
        Returns
        -------
        List[str]
            List of parameter names in the order they were added
        """
        return self._parameter_order.copy()
    
    def get_distribution(self, name: str) -> Distribution:
        """
        Get the distribution for a parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        
        Returns
        -------
        Distribution
            The distribution associated with the parameter
        
        Raises
        ------
        KeyError
            If parameter doesn't exist
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")
        
        return self._parameters[name]
    
    def generate(
        self,
        n: int = 1,
        return_type: str = 'dict',
        method: Literal['random', 'sobol', 'halton', 'lhs'] = 'random'
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
        """
        Generate parameter samples using specified sampling method.
        
        Parameters
        ----------
        n : int, optional
            Number of sample sets to generate, by default 1
        return_type : str, optional
            Format of returned samples:
            - 'dict': Dictionary mapping parameter names to arrays
            - 'dataframe': pandas DataFrame with columns for each parameter
            - 'array': 2D numpy array (n_samples, n_parameters)
            By default 'dict'
        method : {'random', 'sobol', 'halton', 'lhs'}, optional
            Sampling method to use:
            - 'random': Standard random sampling (default)
            - 'sobol': Sobol quasi-random sequence (better space-filling)
            - 'halton': Halton quasi-random sequence
            - 'lhs': Latin Hypercube Sampling (ensures coverage across each dimension)
            By default 'random'
        
        Returns
        -------
        Dict[str, np.ndarray] or pd.DataFrame or np.ndarray
            Generated samples in requested format
        
        Raises
        ------
        ValueError
            If no parameters defined, invalid return_type, or invalid method
        
        Notes
        -----
        Quasi-Monte Carlo (QMC) methods (Sobol, Halton) provide better space-filling
        properties than random sampling, which can be beneficial for:
        - Experimental design with limited samples
        - Sensitivity analysis
        - Parameter space exploration
        
        Latin Hypercube Sampling ensures that each parameter is sampled evenly
        across its range, providing good coverage with fewer samples.
        
        Examples
        --------
        >>> sampler = Sampler()
        >>> sampler.add_parameter('x', UniformDistribution(0, 1))
        >>> sampler.add_parameter('y', NormalDistribution(0, 1))
        >>> 
        >>> # Random sampling (default)
        >>> samples = sampler.generate(n=5, method='random')
        >>> 
        >>> # Sobol sequence for better coverage
        >>> samples = sampler.generate(n=100, method='sobol')
        >>> 
        >>> # Latin Hypercube Sampling
        >>> df = sampler.generate(n=50, return_type='dataframe', method='lhs')
        """
        if not self._parameters:
            raise ValueError("No parameters defined. Use add_parameter() first.")
        
        if return_type not in ['dict', 'dataframe', 'array']:
            raise ValueError(
                f"return_type must be 'dict', 'dataframe', or 'array', "
                f"got '{return_type}'"
            )
        
        if method not in ['random', 'sobol', 'halton', 'lhs']:
            raise ValueError(
                f"method must be 'random', 'sobol', 'halton', or 'lhs', "
                f"got '{method}'"
            )
        
        # Generate samples based on method
        if method == 'random':
            # Standard random sampling - each distribution samples independently
            samples_dict = {}
            for param_name in self._parameter_order:
                distribution = self._parameters[param_name]
                samples_dict[param_name] = distribution.sample(size=n)
        else:
            # QMC methods - generate uniform samples, then transform
            samples_dict = self._generate_qmc(n, method)
        
        # Return in requested format
        if return_type == 'dict':
            return samples_dict
        
        elif return_type == 'dataframe':
            return pd.DataFrame(samples_dict)
        
        elif return_type == 'array':
            # Stack arrays as columns
            arrays = [samples_dict[name] for name in self._parameter_order]
            return np.column_stack(arrays)
    
    def _generate_qmc(
        self,
        n: int,
        method: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate samples using Quasi-Monte Carlo methods.
        
        QMC methods generate uniform samples in [0, 1]^d space, which are then
        transformed through the inverse CDF of each distribution.
        
        Parameters
        ----------
        n : int
            Number of samples to generate
        method : str
            QMC method: 'sobol', 'halton', or 'lhs'
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping parameter names to sample arrays
        """
        d = len(self._parameters)  # Number of dimensions
        
        # Generate uniform samples in [0, 1]^d
        if method == 'sobol':
            sampler_qmc = qmc.Sobol(d=d, scramble=True, seed=self.random_seed)
            uniform_samples = sampler_qmc.random(n=n)
        elif method == 'halton':
            sampler_qmc = qmc.Halton(d=d, scramble=True, seed=self.random_seed)
            uniform_samples = sampler_qmc.random(n=n)
        elif method == 'lhs':
            sampler_qmc = qmc.LatinHypercube(d=d, seed=self.random_seed)
            uniform_samples = sampler_qmc.random(n=n)
        
        # Transform uniform samples through each distribution's inverse CDF
        samples_dict = {}
        for i, param_name in enumerate(self._parameter_order):
            distribution = self._parameters[param_name]
            uniform_col = uniform_samples[:, i]
            
            # Transform through distribution
            samples_dict[param_name] = self._transform_uniform_to_distribution(
                uniform_col, distribution
            )
        
        return samples_dict
    
    def _transform_uniform_to_distribution(
        self,
        uniform_samples: np.ndarray,
        distribution: Distribution
    ) -> np.ndarray:
        """
        Transform uniform [0, 1] samples to target distribution.
        
        Uses the inverse CDF (quantile function) method. For distributions
        without analytical inverse CDF, we use numerical approximation.
        
        Parameters
        ----------
        uniform_samples : np.ndarray
            Uniform samples in [0, 1]
        distribution : Distribution
            Target distribution
        
        Returns
        -------
        np.ndarray
            Samples from the target distribution
        """
        from multifolio.core.sampling.distributions.continuous import (
            UniformDistribution,
            NormalDistribution,
            TruncatedNormalDistribution,
            BetaDistribution,
            ExponentialDistribution,
            GammaDistribution,
            LogNormalDistribution,
            WeibullDistribution,
            TriangularDistribution,
            CustomContinuousDistribution,
        )
        from multifolio.core.sampling.distributions.discrete import (
            ConstantDistribution,
            PoissonDistribution,
            UniformDiscreteDistribution,
            CustomDiscreteDistribution,
        )
        from scipy import stats
        
        # Use inverse CDF (quantile function) to transform
        if isinstance(distribution, ConstantDistribution):
            return np.full_like(uniform_samples, distribution.value)
        
        elif isinstance(distribution, UniformDistribution):
            # Inverse CDF: F^(-1)(u) = low + u * (high - low)
            return distribution.low + uniform_samples * (distribution.high - distribution.low)
        
        elif isinstance(distribution, NormalDistribution):
            # Use scipy's inverse CDF
            return stats.norm.ppf(uniform_samples, loc=distribution.mean, scale=distribution.std)
        
        elif isinstance(distribution, TruncatedNormalDistribution):
            # Use scipy's truncated normal inverse CDF
            a = (distribution.low - distribution.mean) / distribution.std
            b = (distribution.high - distribution.mean) / distribution.std
            return stats.truncnorm.ppf(
                uniform_samples,
                a, b,
                loc=distribution.mean,
                scale=distribution.std
            )
        
        elif isinstance(distribution, BetaDistribution):
            # First transform through beta [0, 1], then scale
            beta_01 = stats.beta.ppf(uniform_samples, distribution.alpha, distribution.beta)
            return distribution.low + beta_01 * (distribution.high - distribution.low)
        
        elif isinstance(distribution, PoissonDistribution):
            # Poisson inverse CDF
            return stats.poisson.ppf(uniform_samples, mu=distribution.lam)
        
        elif isinstance(distribution, UniformDiscreteDistribution):
            # Discrete uniform: scale and round
            continuous = distribution.low + uniform_samples * (distribution.high - distribution.low + 1)
            return np.floor(continuous).astype(int)
        
        elif isinstance(distribution, ExponentialDistribution):
            # Exponential inverse CDF: -log(1-u) / rate
            return -np.log(1 - uniform_samples) / distribution.rate
        
        elif isinstance(distribution, GammaDistribution):
            # Use scipy's gamma inverse CDF
            return stats.gamma.ppf(uniform_samples, a=distribution.shape, scale=distribution.scale)
        
        elif isinstance(distribution, LogNormalDistribution):
            # Use scipy's lognormal inverse CDF
            return stats.lognorm.ppf(uniform_samples, s=distribution.sigma, scale=np.exp(distribution.mu))
        
        elif isinstance(distribution, WeibullDistribution):
            # Weibull inverse CDF: scale * (-log(1-u))^(1/shape)
            return distribution.scale * (-np.log(1 - uniform_samples)) ** (1 / distribution.shape)
        
        elif isinstance(distribution, TriangularDistribution):
            # Use scipy's triangular inverse CDF
            c = (distribution.mode - distribution.low) / (distribution.high - distribution.low)
            return stats.triang.ppf(uniform_samples, c, 
                                   loc=distribution.low, 
                                   scale=distribution.high - distribution.low)
        
        elif isinstance(distribution, (CustomContinuousDistribution, CustomDiscreteDistribution)):
            # Custom distributions - fallback to their own sampling
            # (loses QMC properties but maintains custom distribution)
            return distribution.sample(size=len(uniform_samples))
        
        else:
            # Fallback: use distribution's own sampling (loses QMC properties)
            # This shouldn't happen with built-in distributions
            return distribution.sample(size=len(uniform_samples))
    
    def __repr__(self) -> str:
        param_info = []
        for name in self._parameter_order:
            dist = self._parameters[name]
            param_info.append(f"  {name}: {dist}")
        
        params_str = "\n".join(param_info) if param_info else "  (none)"
        return f"Sampler(\n{params_str}\n)"
