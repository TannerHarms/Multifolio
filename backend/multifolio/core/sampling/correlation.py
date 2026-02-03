"""
Correlation and copula structures for multi-parameter sampling.

This module provides tools for modeling dependencies between random variables
using copulas, enabling realistic correlated sampling in Monte Carlo simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.linalg import cholesky


class CorrelationStructure:
    """
    Base class for correlation structures used in multi-parameter sampling.
    
    Correlation structures (copulas) separate the marginal distributions from
    the dependency structure, allowing flexible modeling of multivariate
    distributions.
    """
    
    def __init__(self, n_variables: int):
        """
        Initialize correlation structure.
        
        Parameters
        ----------
        n_variables : int
            Number of correlated variables
        """
        self.n_variables = n_variables
    
    def transform_uniform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform independent uniform samples to correlated uniform samples.
        
        Parameters
        ----------
        uniform_samples : np.ndarray
            Independent uniform [0,1] samples, shape (n_samples, n_variables)
        
        Returns
        -------
        np.ndarray
            Correlated uniform samples, shape (n_samples, n_variables)
        """
        raise NotImplementedError("Subclasses must implement transform_uniform")


class GaussianCopula(CorrelationStructure):
    """
    Gaussian copula for modeling correlations between variables.
    
    The Gaussian copula is the most commonly used copula in practice. It uses
    a correlation matrix to model dependencies while allowing each variable to
    have any marginal distribution.
    
    Mathematical Background
    -----------------------
    1. Start with independent uniform samples U ~ Uniform(0,1)
    2. Transform to standard normals: Z = Phi^(-1)(U)
    3. Apply correlation via Cholesky: Z_corr = Z * L^T, where L*L^T = Rho
    4. Transform back to correlated uniforms: U_corr = Phi(Z_corr)
    5. Apply inverse CDF of marginal distributions
    
    Properties
    ----------
    - Preserves rank correlation (Spearman's rho)
    - Linear correlation in normal space
    - No tail dependence (independence in extremes)
    - Computationally efficient
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix, shape (n_variables, n_variables)
        Must be symmetric positive semi-definite with ones on diagonal
    
    Attributes
    ----------
    correlation_matrix : np.ndarray
        The correlation matrix
    cholesky_factor : np.ndarray
        Lower triangular Cholesky factor of correlation matrix
    
    Examples
    --------
    >>> # Create 2D correlation
    >>> rho = np.array([[1.0, 0.7], [0.7, 1.0]])
    >>> copula = GaussianCopula(rho)
    >>> 
    >>> # Transform independent uniforms to correlated
    >>> u_indep = np.random.uniform(0, 1, (1000, 2))
    >>> u_corr = copula.transform_uniform(u_indep)
    >>> 
    >>> # Check correlation preserved
    >>> np.corrcoef(u_corr.T)  # Approximately [[1, 0.7], [0.7, 1]]
    """
    
    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialize Gaussian copula with correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix, must be symmetric positive definite
        
        Raises
        ------
        ValueError
            If correlation matrix is invalid
        """
        # Validate correlation matrix
        if correlation_matrix.ndim != 2:
            raise ValueError("Correlation matrix must be 2-dimensional")
        
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
        
        n_vars = correlation_matrix.shape[0]
        
        # Check diagonal is all ones
        if not np.allclose(np.diag(correlation_matrix), 1.0):
            raise ValueError("Correlation matrix diagonal must be all ones")
        
        # Check symmetric
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")
        
        super().__init__(n_vars)
        self.correlation_matrix = correlation_matrix
        
        # Compute Cholesky decomposition for efficient transformation
        try:
            self.cholesky_factor = cholesky(correlation_matrix, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite")
    
    def transform_uniform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform independent uniform samples to correlated uniform samples.
        
        Algorithm
        ---------
        1. Transform U ~ Uniform(0,1) to Z ~ Normal(0,1) via inverse normal CDF
        2. Apply correlation structure: Z_corr = Z @ L.T (Cholesky factor)
        3. Transform back: U_corr = Phi(Z_corr) via normal CDF
        
        Parameters
        ----------
        uniform_samples : np.ndarray
            Independent uniform [0,1] samples, shape (n_samples, n_variables)
        
        Returns
        -------
        np.ndarray
            Correlated uniform samples, shape (n_samples, n_variables)
            
        Notes
        -----
        - Input samples should be truly independent (from RNG or QMC)
        - Output preserves rank correlation from correlation matrix
        - Spearman's rho ≈ 6/pi * arcsin(rho/2) for Pearson's rho
        """
        if uniform_samples.shape[1] != self.n_variables:
            raise ValueError(
                f"Expected {self.n_variables} variables, got {uniform_samples.shape[1]}"
            )
        
        # Clip to avoid numerical issues at boundaries
        uniform_samples = np.clip(uniform_samples, 1e-10, 1 - 1e-10)
        
        # Step 1: Transform uniforms to standard normals
        standard_normals = stats.norm.ppf(uniform_samples)
        
        # Step 2: Apply correlation via Cholesky factor
        # Z_corr = Z @ L.T where L is lower triangular Cholesky factor
        correlated_normals = standard_normals @ self.cholesky_factor.T
        
        # Step 3: Transform back to correlated uniforms
        correlated_uniforms = stats.norm.cdf(correlated_normals)
        
        return correlated_uniforms
    
    def get_spearman_correlation(self) -> np.ndarray:
        """
        Get approximate Spearman rank correlation matrix.
        
        For Gaussian copula, the relationship between Pearson (rho) and
        Spearman (rho_s) correlation is approximately:
        rho_s ≈ (6/pi) * arcsin(rho/2)
        
        Returns
        -------
        np.ndarray
            Approximate Spearman correlation matrix
        """
        return (6 / np.pi) * np.arcsin(self.correlation_matrix / 2)


class CorrelationManager:
    """
    Manages correlations between parameters in multi-parameter sampling.
    
    This class provides a high-level interface for specifying and managing
    correlations between parameters. It handles:
    - Pairwise correlation specification
    - Building correlation matrices
    - Validating correlation structures
    - Applying correlations during sampling
    
    Parameters
    ----------
    parameter_names : List[str]
        Names of parameters in order
    
    Attributes
    ----------
    parameter_names : List[str]
        Names of parameters
    parameter_indices : Dict[str, int]
        Mapping from parameter names to indices
    correlation_matrix : np.ndarray
        Full correlation matrix
    copula : GaussianCopula
        Gaussian copula for correlation
    
    Examples
    --------
    >>> manager = CorrelationManager(['temp', 'pressure', 'yield'])
    >>> manager.set_correlation('temp', 'pressure', 0.8)
    >>> manager.set_correlation('pressure', 'yield', -0.5)
    >>> corr_matrix = manager.get_correlation_matrix()
    """
    
    def __init__(self, parameter_names: List[str]):
        """
        Initialize correlation manager.
        
        Parameters
        ----------
        parameter_names : List[str]
            Names of parameters in the order they appear in sampling
        """
        self.parameter_names = parameter_names
        self.parameter_indices = {name: i for i, name in enumerate(parameter_names)}
        self.n_parameters = len(parameter_names)
        
        # Initialize with identity matrix (no correlation)
        self.correlation_matrix = np.eye(self.n_parameters)
        self.copula: Optional[GaussianCopula] = None
    
    def set_correlation(self, param1: str, param2: str, correlation: float):
        """
        Set correlation between two parameters.
        
        Parameters
        ----------
        param1 : str
            Name of first parameter
        param2 : str
            Name of second parameter
        correlation : float
            Correlation coefficient, must be in [-1, 1]
        
        Raises
        ------
        ValueError
            If parameter names are invalid or correlation is out of range
        """
        if param1 not in self.parameter_indices:
            raise ValueError(f"Unknown parameter: {param1}")
        if param2 not in self.parameter_indices:
            raise ValueError(f"Unknown parameter: {param2}")
        
        if not -1 <= correlation <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")
        
        i = self.parameter_indices[param1]
        j = self.parameter_indices[param2]
        
        # Set symmetric entries
        self.correlation_matrix[i, j] = correlation
        self.correlation_matrix[j, i] = correlation
        
        # Rebuild copula
        self._rebuild_copula()
    
    def set_correlation_matrix(self, correlation_matrix: np.ndarray):
        """
        Set full correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix, shape (n_parameters, n_parameters)
        
        Raises
        ------
        ValueError
            If matrix is invalid
        """
        if correlation_matrix.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(
                f"Correlation matrix must be {self.n_parameters}x{self.n_parameters}"
            )
        
        # Validation will happen in GaussianCopula constructor
        self.correlation_matrix = correlation_matrix.copy()
        self._rebuild_copula()
    
    def _rebuild_copula(self):
        """Rebuild Gaussian copula with current correlation matrix."""
        try:
            self.copula = GaussianCopula(self.correlation_matrix)
        except ValueError as e:
            raise ValueError(f"Invalid correlation structure: {e}")
    
    def has_correlations(self) -> bool:
        """
        Check if any correlations are specified.
        
        Returns
        -------
        bool
            True if any off-diagonal correlations are non-zero
        """
        return not np.allclose(self.correlation_matrix, np.eye(self.n_parameters))
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get current correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        return self.correlation_matrix.copy()
    
    def get_correlation(self, param1: str, param2: str) -> float:
        """
        Get correlation between two parameters.
        
        Parameters
        ----------
        param1 : str
            Name of first parameter
        param2 : str
            Name of second parameter
        
        Returns
        -------
        float
            Correlation coefficient
        """
        i = self.parameter_indices[param1]
        j = self.parameter_indices[param2]
        return self.correlation_matrix[i, j]
    
    def transform_samples(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform independent uniform samples to correlated samples.
        
        Parameters
        ----------
        uniform_samples : np.ndarray
            Independent uniform samples, shape (n_samples, n_parameters)
        
        Returns
        -------
        np.ndarray
            Correlated uniform samples if correlations exist,
            otherwise returns input unchanged
        """
        if not self.has_correlations():
            return uniform_samples
        
        if self.copula is None:
            self._rebuild_copula()
        
        return self.copula.transform_uniform(uniform_samples)
    
    def __repr__(self) -> str:
        if not self.has_correlations():
            return f"CorrelationManager(n_parameters={self.n_parameters}, no correlations)"
        
        # Count non-zero correlations (excluding diagonal)
        n_corr = np.sum(np.abs(self.correlation_matrix - np.eye(self.n_parameters)) > 1e-10) // 2
        return f"CorrelationManager(n_parameters={self.n_parameters}, n_correlations={n_corr})"
