"""Multi-parameter sampler for experimental design."""

from typing import Dict, List, Optional, Union, Literal, Callable, Any, Generator
import numpy as np
import pandas as pd
from scipy.stats import qmc
import re
import json
import pickle
from pathlib import Path

from multifolio.core.sampling.distributions.base import Distribution
from multifolio.core.sampling.correlation import CorrelationManager


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
        self._correlation_manager: Optional[CorrelationManager] = None
        self._derived_parameters: Dict[str, Dict[str, Any]] = {}
        self._derived_order: List[str] = []
        self._constraints: List[str] = []
    
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
    
    def add_derived_parameter(
        self,
        name: str,
        formula: Optional[Union[str, Callable]] = None,
        depends_on: Optional[List[str]] = None
    ) -> "Sampler":
        """
        Add a derived parameter computed from other parameters.
        
        Derived parameters are computed after base parameters are sampled.
        They can be specified either as string formulas or callable functions.
        
        Parameters
        ----------
        name : str
            Name for the derived parameter (must be unique)
        formula : str or callable
            How to compute the parameter. Can be:
            - String formula: "X * Y", "X + Y", "X**2 + Y**2", etc.
              Supports: +, -, *, /, **, (), and numpy functions (np.sqrt, np.log, etc.)
            - Callable: Function that takes a dict/DataFrame of samples and returns array
              Example: lambda samples: samples["X"] * samples["Y"]
        depends_on : List[str], optional
            List of parameter names this depends on. If not provided, will be
            inferred from string formula or must be explicitly provided for callables.
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Raises
        ------
        ValueError
            If name already exists, dependencies missing, or formula invalid
        
        Examples
        --------
        >>> sampler = Sampler(n_samples=1000)
        >>> sampler.add_parameter("X", NormalDistribution(0, 1))
        >>> sampler.add_parameter("Y", NormalDistribution(5, 2))
        >>> 
        >>> # String formula (dependencies auto-detected)
        >>> sampler.add_derived_parameter("Z", formula="X * Y")
        >>> sampler.add_derived_parameter("W", formula="X**2 + Y**2")
        >>> sampler.add_derived_parameter("log_X", formula="np.log(np.abs(X) + 1)")
        >>> 
        >>> # Callable (must specify dependencies)
        >>> sampler.add_derived_parameter(
        ...     "ratio",
        ...     formula=lambda s: s["X"] / (s["Y"] + 0.001),
        ...     depends_on=["X", "Y"]
        ... )
        >>> 
        >>> samples = sampler.generate(as_dataframe=True)
        >>> # samples will have columns: X, Y, Z, W, log_X, ratio
        """
        # Validate name is unique
        if name in self._parameters or name in self._derived_parameters:
            raise ValueError(f"Parameter '{name}' already exists")
        
        if formula is None:
            raise ValueError("Formula must be provided")
        
        # Handle string formulas
        if isinstance(formula, str):
            # Infer dependencies from formula if not provided
            if depends_on is None:
                depends_on = self._infer_dependencies(formula)
            
            # Validate dependencies exist
            for dep in depends_on:
                if dep not in self._parameters and dep not in self._derived_parameters:
                    raise ValueError(f"Dependency '{dep}' not found in parameters")
            
            # Store formula as-is, will be evaluated later
            self._derived_parameters[name] = {
                'formula': formula,
                'depends_on': depends_on,
                'type': 'string'
            }
        
        # Handle callable formulas
        elif callable(formula):
            if depends_on is None:
                raise ValueError(
                    "For callable formulas, 'depends_on' must be explicitly provided"
                )
            
            # Validate dependencies exist
            for dep in depends_on:
                if dep not in self._parameters and dep not in self._derived_parameters:
                    raise ValueError(f"Dependency '{dep}' not found in parameters")
            
            self._derived_parameters[name] = {
                'formula': formula,
                'depends_on': depends_on,
                'type': 'callable'
            }
        
        else:
            raise ValueError("Formula must be a string or callable")
        
        # Add to derived order (will be sorted by dependencies later)
        self._derived_order.append(name)
        
        return self
    
    def _infer_dependencies(self, formula: str) -> List[str]:
        """
        Infer parameter dependencies from a string formula.
        
        Parameters
        ----------
        formula : str
            The formula string
        
        Returns
        -------
        List[str]
            List of parameter names found in the formula
        """
        # Find all valid Python identifiers that aren't numpy/builtin functions
        # Pattern: word boundaries around identifiers
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        candidates = set(re.findall(pattern, formula))
        
        # Filter to only known parameters (base or derived)
        all_params = set(self._parameters.keys()) | set(self._derived_parameters.keys())
        dependencies = [p for p in candidates if p in all_params]
        
        return dependencies
    
    def _sort_derived_parameters(self) -> List[str]:
        """
        Topologically sort derived parameters by dependencies.
        
        Returns
        -------
        List[str]
            Sorted list of derived parameter names
        
        Raises
        ------
        ValueError
            If circular dependencies are detected
        """
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for name in self._derived_order:
            deps = self._derived_parameters[name]['depends_on']
            # Only count derived parameters as dependencies (base params are already computed)
            derived_deps = [d for d in deps if d in self._derived_parameters]
            graph[name] = derived_deps
            in_degree[name] = len(derived_deps)
        
        # Kahn's algorithm for topological sort
        sorted_params = []
        queue = [name for name in self._derived_order if in_degree[name] == 0]
        
        while queue:
            current = queue.pop(0)
            sorted_params.append(current)
            
            # Reduce in-degree for dependents
            for name in self._derived_order:
                if current in graph[name]:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        # Check for cycles
        if len(sorted_params) != len(self._derived_order):
            raise ValueError("Circular dependencies detected in derived parameters")
        
        return sorted_params
    
    def set_correlation(
        self,
        param1: str,
        param2: str,
        correlation: float
    ) -> "Sampler":
        """
        Set correlation between two parameters.
        
        Uses Gaussian copula to model correlation between parameters while
        preserving their marginal distributions.
        
        Parameters
        ----------
        param1 : str
            Name of first parameter
        param2 : str
            Name of second parameter
        correlation : float
            Correlation coefficient in [-1, 1]
            - 1.0: Perfect positive correlation
            - 0.0: No correlation (independent)
            - -1.0: Perfect negative correlation
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Raises
        ------
        ValueError
            If parameter names are invalid or correlation is out of range
        
        Notes
        -----
        The correlation is preserved as rank correlation (Spearman's rho).
        For linear correlation (Pearson's rho), the relationship is approximately:
        rho_spearman ≈ (6/pi) * arcsin(rho_pearson / 2)
        
        Examples
        --------
        >>> sampler = Sampler(seed=42)
        >>> sampler.add_parameter('temp', NormalDistribution(100, 10))
        >>> sampler.add_parameter('pressure', NormalDistribution(50, 5))
        >>> sampler.set_correlation('temp', 'pressure', 0.8)
        >>> samples = sampler.generate(n=1000)
        """
        self._ensure_correlation_manager()
        self._correlation_manager.set_correlation(param1, param2, correlation)
        return self
    
    def set_correlation_matrix(self, correlation_matrix: np.ndarray) -> "Sampler":
        """
        Set full correlation matrix for all parameters.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix, shape (n_parameters, n_parameters)
            Must be symmetric positive semi-definite with ones on diagonal
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Raises
        ------
        ValueError
            If matrix is invalid or size doesn't match number of parameters
        
        Examples
        --------
        >>> import numpy as np
        >>> sampler = Sampler()
        >>> sampler.add_parameter('a', UniformDistribution(0, 1))
        >>> sampler.add_parameter('b', UniformDistribution(0, 1))
        >>> sampler.add_parameter('c', UniformDistribution(0, 1))
        >>> 
        >>> # Set correlations: a-b: 0.7, b-c: -0.5, a-c: 0.2
        >>> corr = np.array([[1.0, 0.7, 0.2],
        ...                  [0.7, 1.0, -0.5],
        ...                  [0.2, -0.5, 1.0]])
        >>> sampler.set_correlation_matrix(corr)
        """
        self._ensure_correlation_manager()
        self._correlation_manager.set_correlation_matrix(correlation_matrix)
        return self
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get current correlation matrix.
        
        Returns
        -------
        np.ndarray
            Correlation matrix. Returns identity matrix if no correlations set.
        """
        if self._correlation_manager is None:
            n = len(self._parameter_order)
            return np.eye(n)
        return self._correlation_manager.get_correlation_matrix()
    
    def has_correlations(self) -> bool:
        """
        Check if any correlations are specified.
        
        Returns
        -------
        bool
            True if correlations have been set
        """
        if self._correlation_manager is None:
            return False
        return self._correlation_manager.has_correlations()
    
    def _ensure_correlation_manager(self):
        """Create correlation manager if it doesn't exist."""
        if self._correlation_manager is None:
            self._correlation_manager = CorrelationManager(self._parameter_order)
    
    def add_constraint(self, constraint: str) -> "Sampler":
        """
        Add a constraint that samples must satisfy.
        
        Constraints are boolean expressions that reference parameter names.
        During generation with constraints, samples that don't satisfy ALL
        constraints are rejected and regenerated.
        
        Parameters
        ----------
        constraint : str
            Boolean expression using parameter names
            Examples: "X + Y <= 100", "X > 0", "X**2 + Y**2 < 1"
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        
        Examples
        --------
        >>> sampler = Sampler()
        >>> sampler.add_parameter("X", UniformDistribution(0, 10))
        >>> sampler.add_parameter("Y", UniformDistribution(0, 10))
        >>> sampler.add_constraint("X + Y <= 15")  # Budget constraint
        >>> sampler.add_constraint("X >= 0.5 * Y")  # Ratio constraint
        
        Notes
        -----
        Constraints use rejection sampling, which can be slow if the constraint
        region is very small compared to the parameter space. Consider:
        - Making constraints as broad as possible
        - Using conditional generation instead for complex conditions
        - Setting appropriate parameter bounds first
        """
        self._constraints.append(constraint)
        return self
    
    def clear_constraints(self) -> "Sampler":
        """
        Remove all constraints.
        
        Returns
        -------
        Sampler
            Returns self for method chaining
        """
        self._constraints = []
        return self
    
    def get_constraints(self) -> List[str]:
        """
        Get list of current constraints.
        
        Returns
        -------
        List[str]
            List of constraint expressions
        """
        return self._constraints.copy()
    
    def filter_samples(
        self,
        samples: Union[pd.DataFrame, Dict[str, np.ndarray]],
        condition: str
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Filter samples based on a condition.
        
        This performs post-generation filtering. For pre-filtering during
        generation, use add_constraint() or generate_conditional().
        
        Parameters
        ----------
        samples : DataFrame or dict
            Samples to filter
        condition : str
            Boolean expression using parameter names
            Examples: "X > 0", "X + Y < 100", "np.abs(X) < 2"
        
        Returns
        -------
        DataFrame or dict
            Filtered samples (same type as input)
        
        Examples
        --------
        >>> samples = sampler.generate(n=10000, return_type='dataframe')
        >>> filtered = sampler.filter_samples(samples, "X > 0 and Y < 10")
        >>> filtered = sampler.filter_samples(samples, "X**2 + Y**2 < 1")
        """
        # Convert to DataFrame for easier filtering
        if isinstance(samples, dict):
            df = pd.DataFrame(samples)
            return_dict = True
        else:
            df = samples
            return_dict = False
        
        # Evaluate condition
        namespace = {'np': np, 'numpy': np}
        namespace.update({col: df[col].values for col in df.columns})
        
        try:
            mask = eval(condition, {"__builtins__": {}}, namespace)
            filtered = df[mask]
        except Exception as e:
            raise ValueError(f"Error evaluating condition '{condition}': {str(e)}")
        
        if return_dict:
            return {col: filtered[col].values for col in filtered.columns}
        return filtered
    
    def generate_conditional(
        self,
        n: int,
        condition: str,
        max_attempts: int = None,
        return_type: str = 'dict',
        method: Literal['random', 'sobol', 'halton', 'lhs'] = 'random'
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
        """
        Generate samples that satisfy a condition using rejection sampling.
        
        This keeps generating samples until n valid samples are obtained.
        More efficient than generate() + filter_samples() for selective conditions.
        
        Parameters
        ----------
        n : int
            Number of valid samples to generate
        condition : str
            Boolean condition that samples must satisfy
            Examples: "X + Y <= 100", "X > 0 and Y > 0"
        max_attempts : int, optional
            Maximum total samples to generate before giving up.
            Default is n * 1000 (assumes ~0.1% acceptance rate minimum)
        return_type : str, optional
            Format: 'dict', 'dataframe', or 'array'
        method : str, optional
            Sampling method: 'random', 'sobol', 'halton', or 'lhs'
        
        Returns
        -------
        DataFrame, dict, or array
            n samples that satisfy the condition
        
        Raises
        ------
        RuntimeError
            If unable to generate n valid samples within max_attempts
        
        Examples
        --------
        >>> # Generate points inside unit circle
        >>> samples = sampler.generate_conditional(
        ...     n=1000,
        ...     condition="X**2 + Y**2 < 1"
        ... )
        
        Notes
        -----
        Uses rejection sampling: generates samples and keeps only those
        satisfying the condition. If the condition region is very small,
        this can be slow. Consider using add_constraint() if you need
        the condition applied consistently across multiple generations.
        
        For complex conditions, consider:
        1. Setting tighter parameter bounds first
        2. Using constraints instead of conditional generation
        3. Increasing max_attempts if needed
        """
        if max_attempts is None:
            max_attempts = n * 1000  # Default: assume at least 0.1% acceptance
        
        valid_samples = []
        total_generated = 0
        batch_size = min(n * 10, 10000)  # Generate in batches
        
        while len(valid_samples) < n and total_generated < max_attempts:
            # Generate a batch
            batch = self.generate(n=batch_size, return_type='dataframe', method=method)
            
            # Filter by condition
            filtered = self.filter_samples(batch, condition)
            
            if len(filtered) > 0:
                valid_samples.append(filtered)
            
            total_generated += batch_size
            
            # Adaptive batch sizing
            if len(valid_samples) > 0:
                acceptance_rate = sum(len(s) for s in valid_samples) / total_generated
                if acceptance_rate > 0:
                    # Estimate how many more we need
                    remaining = n - sum(len(s) for s in valid_samples)
                    if remaining > 0:
                        batch_size = max(min(int(remaining / acceptance_rate * 2), 10000), 10)
        
        # Check if we got enough samples
        total_valid = sum(len(s) for s in valid_samples) if valid_samples else 0
        
        if total_valid < n:
            raise RuntimeError(
                f"Unable to generate {n} valid samples. "
                f"Only got {total_valid} samples after {total_generated} attempts. "
                f"Condition: '{condition}'. "
                f"Consider: (1) relaxing the condition, (2) adjusting parameter bounds, "
                f"or (3) increasing max_attempts."
            )
        
        # Combine and trim to exactly n samples
        result_df = pd.concat(valid_samples, ignore_index=True).iloc[:n]
        
        # Convert to requested format
        if return_type == 'dataframe':
            return result_df
        elif return_type == 'dict':
            return {col: result_df[col].values for col in result_df.columns}
        elif return_type == 'array':
            ordered_names = self._parameter_order + self._derived_order
            return result_df[ordered_names].values
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
    
    def generate_batches(
        self,
        n_per_batch: int,
        n_batches: int,
        return_type: str = 'dict',
        method: Literal['random', 'sobol', 'halton', 'lhs'] = 'random',
        track_batch_id: bool = True
    ) -> Generator[Union[Dict[str, np.ndarray], pd.DataFrame, np.ndarray], None, None]:
        """
        Generate samples in batches (generator/iterator pattern).
        
        Useful for:
        - Processing large numbers of samples without holding all in memory
        - Tracking which batch each sample came from
        - Progressive computation or streaming scenarios
        
        Parameters
        ----------
        n_per_batch : int
            Number of samples per batch
        n_batches : int
            Number of batches to generate
        return_type : str, optional
            Format: 'dict', 'dataframe', or 'array'
        method : str, optional
            Sampling method: 'random', 'sobol', 'halton', or 'lhs'
        track_batch_id : bool, optional
            If True and return_type='dataframe' or 'dict', adds 'batch_id' column.
            Default is True.
        
        Yields
        ------
        DataFrame, dict, or array
            One batch of samples per iteration
        
        Examples
        --------
        >>> # Process batches one at a time
        >>> for batch in sampler.generate_batches(n_per_batch=1000, n_batches=10):
        ...     results = expensive_simulation(batch)
        ...     save_results(results)
        
        >>> # Collect all batches
        >>> all_batches = list(sampler.generate_batches(
        ...     n_per_batch=500,
        ...     n_batches=20,
        ...     return_type='dataframe'
        ... ))
        
        Notes
        -----
        This is a generator that yields batches one at a time, so memory
        usage stays constant regardless of n_batches.
        
        For QMC methods (sobol, halton), consecutive batches will continue
        the low-discrepancy sequence, maintaining good space-filling properties
        across all batches combined.
        """
        for batch_idx in range(n_batches):
            # Generate batch
            batch = self.generate(n=n_per_batch, return_type=return_type, method=method)
            
            # Add batch ID if requested
            if track_batch_id and return_type in ['dataframe', 'dict']:
                if isinstance(batch, pd.DataFrame):
                    batch['batch_id'] = batch_idx
                elif isinstance(batch, dict):
                    batch['batch_id'] = np.full(n_per_batch, batch_idx)
            
            yield batch
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """
        Save sampler configuration to file.
        
        Saves parameters, distributions, correlations, derived parameters,
        and constraints. Does NOT save generated data - use save_data() for that.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save configuration (JSON format)
        
        Examples
        --------
        >>> sampler.save_config("my_sampler.json")
        >>> # Later...
        >>> sampler2 = Sampler.load_config("my_sampler.json")
        
        Notes
        -----
        The configuration file is human-readable JSON. You can edit it
        directly if needed, but be careful with the distribution parameters.
        
        Custom distributions and callable derived parameters cannot be fully
        serialized - you'll need to recreate those manually after loading.
        """
        config = {
            'random_seed': self.random_seed,
            'parameters': {},
            'parameter_order': self._parameter_order,
            'correlations': None,
            'derived_parameters': {},
            'derived_order': self._derived_order,
            'constraints': self._constraints
        }
        
        # Serialize distributions
        for name, dist in self._parameters.items():
            dist_type = dist.__class__.__name__
            dist_dict = {'type': dist_type}
            
            # Extract parameters based on distribution type
            if dist_type == 'UniformDistribution':
                dist_dict['low'] = float(dist.low)
                dist_dict['high'] = float(dist.high)
            elif dist_type == 'NormalDistribution':
                dist_dict['mean'] = float(dist.mean)
                dist_dict['std'] = float(dist.std)
            elif dist_type == 'TruncatedNormalDistribution':
                dist_dict['mean'] = float(dist.mean)
                dist_dict['std'] = float(dist.std)
                dist_dict['low'] = float(dist.low)
                dist_dict['high'] = float(dist.high)
            elif dist_type == 'BetaDistribution':
                dist_dict['alpha'] = float(dist.alpha)
                dist_dict['beta'] = float(dist.beta)
            elif dist_type == 'ConstantDistribution':
                dist_dict['value'] = float(dist.value)
            elif dist_type == 'PoissonDistribution':
                dist_dict['lam'] = float(dist.lam)
            elif dist_type == 'UniformDiscreteDistribution':
                dist_dict['low'] = int(dist.low)
                dist_dict['high'] = int(dist.high)
            else:
                # For custom distributions, store placeholder
                dist_dict['note'] = '<custom distribution - not serialized>'
            
            config['parameters'][name] = dist_dict
        
        # Serialize correlations
        if self.has_correlations():
            config['correlations'] = self.get_correlation_matrix().tolist()
        
        # Serialize derived parameters
        for name, info in self._derived_parameters.items():
            if info['type'] == 'string':
                config['derived_parameters'][name] = {
                    'formula': info['formula'],
                    'depends_on': info['depends_on'],
                    'type': 'string'
                }
            else:
                # Callable - can't serialize, save placeholder
                config['derived_parameters'][name] = {
                    'formula': '<callable - not serialized>',
                    'depends_on': info['depends_on'],
                    'type': 'callable'
                }
        
        # Write to file
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> "Sampler":
        """
        Load sampler configuration from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to configuration file
        
        Returns
        -------
        Sampler
            New sampler instance with loaded configuration
        
        Examples
        --------
        >>> sampler = Sampler.load_config("my_sampler.json")
        
        Notes
        -----
        Callable derived parameters cannot be loaded and must be re-added
        manually after loading.
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Create sampler
        sampler = cls(random_seed=config.get('random_seed'))
        
        # Load parameters
        from multifolio.core.sampling.distributions import (
            UniformDistribution, NormalDistribution, TruncatedNormalDistribution,
            BetaDistribution, ConstantDistribution, PoissonDistribution,
            UniformDiscreteDistribution, CustomContinuousDistribution,
            CustomDiscreteDistribution
        )
        
        dist_classes = {
            'UniformDistribution': UniformDistribution,
            'NormalDistribution': NormalDistribution,
            'TruncatedNormalDistribution': TruncatedNormalDistribution,
            'BetaDistribution': BetaDistribution,
            'ConstantDistribution': ConstantDistribution,
            'PoissonDistribution': PoissonDistribution,
            'UniformDiscreteDistribution': UniformDiscreteDistribution,
            'CustomContinuousDistribution': CustomContinuousDistribution,
            'CustomDiscreteDistribution': CustomDiscreteDistribution,
        }
        
        for name in config['parameter_order']:
            dist_config = config['parameters'][name]
            dist_type = dist_config['type']
            
            # Create distribution based on type
            if dist_type == 'UniformDistribution':
                dist = UniformDistribution(dist_config['low'], dist_config['high'])
            elif dist_type == 'NormalDistribution':
                dist = NormalDistribution(dist_config['mean'], dist_config['std'])
            elif dist_type == 'TruncatedNormalDistribution':
                dist = TruncatedNormalDistribution(
                    dist_config['mean'], dist_config['std'], 
                    dist_config['low'], dist_config['high']
                )
            elif dist_type == 'BetaDistribution':
                dist = BetaDistribution(dist_config['alpha'], dist_config['beta'])
            elif dist_type == 'ConstantDistribution':
                dist = ConstantDistribution(dist_config['value'])
            elif dist_type == 'PoissonDistribution':
                dist = PoissonDistribution(dist_config['lam'])
            elif dist_type == 'UniformDiscreteDistribution':
                dist = UniformDiscreteDistribution(dist_config['low'], dist_config['high'])
            else:
                # Skip unsupported distributions
                continue
            
            sampler.add_parameter(name, dist)
        
        # Load correlations
        if config.get('correlations') is not None:
            corr_matrix = np.array(config['correlations'])
            sampler.set_correlation_matrix(corr_matrix)
        
        # Load derived parameters (string formulas only)
        for name in config['derived_order']:
            info = config['derived_parameters'][name]
            if info['type'] == 'string':
                sampler.add_derived_parameter(
                    name,
                    formula=info['formula'],
                    depends_on=info['depends_on']
                )
            # Skip callable - user must re-add manually
        
        # Load constraints
        for constraint in config.get('constraints', []):
            sampler.add_constraint(constraint)
        
        return sampler
    
    @staticmethod
    def save_data(
        data: Union[pd.DataFrame, Dict[str, np.ndarray]],
        filepath: Union[str, Path],
        format: str = 'auto',
        compression: str = None
    ) -> None:
        """
        Save generated sample data to file.
        
        Parameters
        ----------
        data : DataFrame or dict
            Data to save
        filepath : str or Path
            Where to save the data
        format : str, optional
            File format: 'csv', 'pickle', 'parquet', 'hdf5', or 'auto' (infer from extension).
            Default is 'auto'.
        compression : str, optional
            Compression to use. Options depend on format:
            - HDF5: 'gzip', 'lzf', 'szip' (default: 'gzip')
            - Parquet: 'gzip', 'snappy', 'brotli', 'zstd'
            Default is None (no compression, except HDF5 uses 'gzip' by default)
        
        Examples
        --------
        >>> samples = sampler.generate(n=10000, return_type='dataframe')
        >>> Sampler.save_data(samples, "samples.csv")
        >>> Sampler.save_data(samples, "samples.pkl", format='pickle')
        >>> Sampler.save_data(samples, "samples.h5", format='hdf5')
        >>> Sampler.save_data(samples, "samples.h5", compression='gzip')
        
        Notes
        -----
        - CSV: Human-readable, widely compatible, but slower for large data
        - Pickle: Fast, preserves dtypes, but Python-specific
        - Parquet: Efficient columnar format, good for large datasets
        - HDF5: Very fast, efficient, supports compression, great for large datasets
        """
        filepath = Path(filepath)
        
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Determine format
        if format == 'auto':
            ext = filepath.suffix.lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.h5', '.hdf5', '.hdf']:
                format = 'hdf5'
            else:
                format = 'csv'  # Default
        
        # Check for HDF5 dependencies
        if format == 'hdf5':
            try:
                import tables
                has_pytables = True
            except ImportError:
                has_pytables = False
            
            try:
                import h5py
                has_h5py = True
            except ImportError:
                has_h5py = False
            
            if not has_pytables and not has_h5py:
                raise ImportError(
                    "HDF5 format requires either 'pytables' or 'h5py'. "
                    "Install with: pip install tables (or pip install h5py)"
                )
        
        # Save
        if format == 'csv':
            data.to_csv(filepath, index=False)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'parquet':
            if compression:
                data.to_parquet(filepath, index=False, compression=compression)
            else:
                data.to_parquet(filepath, index=False)
        elif format == 'hdf5':
            # Try PyTables first (cleaner API for DataFrames)
            try:
                # Default to gzip compression for HDF5 (good balance of speed and size)
                if compression is None:
                    compression = 'gzip'
                
                # HDF5 requires a key for the dataset
                data.to_hdf(
                    filepath,
                    key='samples',
                    mode='w',
                    complevel=9 if compression else 0,
                    complib=compression if compression else None
                )
            except ImportError:
                # Fall back to h5py (more common)
                import h5py
                with h5py.File(filepath, 'w') as f:
                    # Store column names
                    f.attrs['columns'] = list(data.columns)
                    
                    # Store each column as a dataset
                    for col in data.columns:
                        col_data = data[col].values
                        if compression:
                            f.create_dataset(col, data=col_data, compression=compression)
                        else:
                            f.create_dataset(col, data=col_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_data(
        filepath: Union[str, Path],
        format: str = 'auto'
    ) -> pd.DataFrame:
        """
        Load sample data from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to data file
        format : str, optional
            File format: 'csv', 'pickle', 'parquet', 'hdf5', or 'auto'.
            Default is 'auto' (infer from extension).
        
        Returns
        -------
        DataFrame
            Loaded data
        
        Examples
        --------
        >>> samples = Sampler.load_data("samples.csv")
        >>> samples = Sampler.load_data("samples.pkl")
        >>> samples = Sampler.load_data("samples.h5")
        """
        filepath = Path(filepath)
        
        # Determine format
        if format == 'auto':
            ext = filepath.suffix.lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.h5', '.hdf5', '.hdf']:
                format = 'hdf5'
            else:
                format = 'csv'
        
        # Check for HDF5 dependencies
        if format == 'hdf5':
            try:
                import tables
                has_pytables = True
            except ImportError:
                has_pytables = False
            
            try:
                import h5py
                has_h5py = True
            except ImportError:
                has_h5py = False
            
            if not has_pytables and not has_h5py:
                raise ImportError(
                    "HDF5 format requires either 'pytables' or 'h5py'. "
                    "Install with: pip install tables (or pip install h5py)"
                )
        
        # Load
        if format == 'csv':
            return pd.read_csv(filepath)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                return data
        elif format == 'parquet':
            return pd.read_parquet(filepath)
        elif format == 'hdf5':
            # Try PyTables first
            try:
                return pd.read_hdf(filepath, key='samples')
            except ImportError:
                # Fall back to h5py
                import h5py
                with h5py.File(filepath, 'r') as f:
                    # Read column names
                    columns = [col.decode() if isinstance(col, bytes) else col 
                              for col in f.attrs['columns']]
                    
                    # Read each column
                    data = {col: f[col][:] for col in columns}
                    
                    return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate(
        self,
        n: int = 1,
        return_type: str = 'dict',
        method: Literal['random', 'sobol', 'halton', 'lhs'] = 'random',
        apply_constraints: bool = True,
        max_constraint_attempts: int = None
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
        apply_constraints : bool, optional
            Whether to apply constraints using rejection sampling.
            If True, only samples satisfying all constraints are returned.
            By default True
        max_constraint_attempts : int, optional
            Maximum attempts when applying constraints via rejection sampling.
            Default is n * 1000 (assumes ~0.1% acceptance rate minimum).
            Only used if apply_constraints=True and constraints exist.
        
        Returns
        -------
        Dict[str, np.ndarray] or pd.DataFrame or np.ndarray
            Generated samples in requested format
        
        Raises
        ------
        ValueError
            If no parameters defined, invalid return_type, or invalid method
        RuntimeError
            If unable to generate n samples satisfying constraints within max_constraint_attempts
        
        Notes
        -----
        Quasi-Monte Carlo (QMC) methods (Sobol, Halton) provide better space-filling
        properties than random sampling, which can be beneficial for:
        - Experimental design with limited samples
        - Sensitivity analysis
        - Parameter space exploration
        
        Latin Hypercube Sampling ensures that each parameter is sampled evenly
        across its range, providing good coverage with fewer samples.
        
        If constraints are defined and apply_constraints=True, rejection sampling
        is used to ensure all returned samples satisfy the constraints. This can
        be slow if the constrained region is small. Consider using generate_conditional()
        for one-time conditional generation, or add_constraint() for consistent
        constraint enforcement across multiple generate() calls.
        
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
        >>> 
        >>> # With constraints
        >>> sampler.add_constraint('x + y <= 1')
        >>> samples = sampler.generate(n=100)  # Only returns valid samples
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
            # Random sampling requires different handling for correlations
            if self.has_correlations():
                # Generate correlated uniform samples first
                uniform_samples = np.random.uniform(0, 1, size=(n, len(self._parameter_order)))
                if self.random_seed is not None:
                    np.random.seed(self.random_seed)
                    uniform_samples = np.random.uniform(0, 1, size=(n, len(self._parameter_order)))
                
                # Apply correlation structure
                correlated_uniforms = self._correlation_manager.transform_samples(uniform_samples)
                
                # Transform through each distribution's inverse CDF
                samples_dict = {}
                for i, param_name in enumerate(self._parameter_order):
                    distribution = self._parameters[param_name]
                    samples_dict[param_name] = self._transform_uniform_to_distribution(
                        correlated_uniforms[:, i], distribution
                    )
            else:
                # Standard independent sampling
                samples_dict = {}
                for param_name in self._parameter_order:
                    distribution = self._parameters[param_name]
                    samples_dict[param_name] = distribution.sample(size=n)
        else:
            # QMC methods - generate uniform samples, then transform
            samples_dict = self._generate_qmc(n, method)
        
        # Compute derived parameters
        if self._derived_parameters:
            samples_dict = self._compute_derived_parameters(samples_dict)
        
        # Apply constraints if requested
        if apply_constraints and self._constraints:
            if max_constraint_attempts is None:
                max_constraint_attempts = n * 1000
            
            # Use rejection sampling to satisfy constraints
            valid_samples = []
            total_generated = n
            batch_size = min(n * 10, 10000)
            
            # Check current batch
            current_df = pd.DataFrame(samples_dict)
            namespace = {'np': np, 'numpy': np}
            namespace.update({col: current_df[col].values for col in current_df.columns})
            
            # Evaluate all constraints
            mask = np.ones(len(current_df), dtype=bool)
            for constraint in self._constraints:
                try:
                    constraint_mask = eval(constraint, {"__builtins__": {}}, namespace)
                    mask &= constraint_mask
                except Exception as e:
                    raise ValueError(f"Error evaluating constraint '{constraint}': {str(e)}")
            
            valid_samples.append(current_df[mask])
            
            # Generate more if needed
            while sum(len(s) for s in valid_samples) < n and total_generated < max_constraint_attempts:
                # Generate another batch
                if method == 'random':
                    if self.has_correlations():
                        uniform_samples = np.random.uniform(0, 1, size=(batch_size, len(self._parameter_order)))
                        correlated_uniforms = self._correlation_manager.transform_samples(uniform_samples)
                        batch_dict = {}
                        for i, param_name in enumerate(self._parameter_order):
                            distribution = self._parameters[param_name]
                            batch_dict[param_name] = self._transform_uniform_to_distribution(
                                correlated_uniforms[:, i], distribution
                            )
                    else:
                        batch_dict = {}
                        for param_name in self._parameter_order:
                            distribution = self._parameters[param_name]
                            batch_dict[param_name] = distribution.sample(size=batch_size)
                else:
                    batch_dict = self._generate_qmc(batch_size, method)
                
                # Compute derived parameters for batch
                if self._derived_parameters:
                    batch_dict = self._compute_derived_parameters(batch_dict)
                
                # Filter by constraints
                batch_df = pd.DataFrame(batch_dict)
                namespace = {'np': np, 'numpy': np}
                namespace.update({col: batch_df[col].values for col in batch_df.columns})
                
                mask = np.ones(len(batch_df), dtype=bool)
                for constraint in self._constraints:
                    constraint_mask = eval(constraint, {"__builtins__": {}}, namespace)
                    mask &= constraint_mask
                
                if mask.any():
                    valid_samples.append(batch_df[mask])
                
                total_generated += batch_size
            
            # Check if we got enough
            total_valid = sum(len(s) for s in valid_samples)
            if total_valid < n:
                constraints_str = " AND ".join(f"({c})" for c in self._constraints)
                raise RuntimeError(
                    f"Unable to generate {n} samples satisfying constraints. "
                    f"Only got {total_valid} valid samples after {total_generated} attempts. "
                    f"Constraints: {constraints_str}. "
                    f"Consider: (1) relaxing constraints, (2) adjusting parameter bounds, "
                    f"or (3) increasing max_constraint_attempts."
                )
            
            # Combine and trim to exactly n
            result_df = pd.concat(valid_samples, ignore_index=True).iloc[:n]
            samples_dict = {col: result_df[col].values for col in result_df.columns}
        
        # Return in requested format
        if return_type == 'dict':
            return samples_dict
        
        elif return_type == 'dataframe':
            # Ensure proper column order: base params, then derived params
            ordered_columns = self._parameter_order + self._derived_order
            df = pd.DataFrame(samples_dict)
            return df[ordered_columns]
        
        elif return_type == 'array':
            # Stack arrays as columns in order: base params, then derived params
            ordered_names = self._parameter_order + self._derived_order
            arrays = [samples_dict[name] for name in ordered_names]
            return np.column_stack(arrays)
    
    def _generate_qmc(
        self,
        n: int,
        method: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate samples using Quasi-Monte Carlo methods.
        
        QMC methods generate uniform samples in [0, 1]^d space, which are then
        transformed through the inverse CDF of each distribution. If correlations
        are specified, they are applied before the transformation.
        
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
        
        # Apply correlation structure if specified
        if self.has_correlations():
            uniform_samples = self._correlation_manager.transform_samples(uniform_samples)
        
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
    
    def _compute_derived_parameters(
        self,
        samples_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute derived parameters based on formulas.
        
        Parameters
        ----------
        samples_dict : Dict[str, np.ndarray]
            Dictionary of base parameter samples
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary including both base and derived parameter samples
        """
        # Sort derived parameters by dependencies
        sorted_derived = self._sort_derived_parameters()
        
        # Compute each derived parameter in order
        for param_name in sorted_derived:
            param_info = self._derived_parameters[param_name]
            formula = param_info['formula']
            formula_type = param_info['type']
            
            if formula_type == 'string':
                # Evaluate string formula
                # Create a safe namespace with numpy and current samples
                namespace = {'np': np, 'numpy': np}
                namespace.update(samples_dict)
                
                try:
                    result = eval(formula, {"__builtins__": {}}, namespace)
                    samples_dict[param_name] = np.asarray(result)
                except Exception as e:
                    raise ValueError(
                        f"Error evaluating formula for '{param_name}': {formula}\n"
                        f"Error: {str(e)}"
                    )
            
            elif formula_type == 'callable':
                # Call the function with samples
                try:
                    result = formula(samples_dict)
                    samples_dict[param_name] = np.asarray(result)
                except Exception as e:
                    raise ValueError(
                        f"Error computing derived parameter '{param_name}'\n"
                        f"Error: {str(e)}"
                    )
        
        return samples_dict
    
    def __repr__(self) -> str:
        param_info = []
        for name in self._parameter_order:
            dist = self._parameters[name]
            param_info.append(f"  {name}: {dist}")
        
        params_str = "\n".join(param_info) if param_info else "  (none)"
        
        # Add derived parameters info
        if self._derived_parameters:
            derived_info = []
            for name in self._derived_order:
                info = self._derived_parameters[name]
                if info['type'] == 'string':
                    derived_info.append(f"  {name} = {info['formula']}")
                else:
                    derived_info.append(f"  {name} = <callable>")
            derived_str = "\nDerived Parameters:\n" + "\n".join(derived_info)
        else:
            derived_str = ""
        
        # Add correlation info if present
        if self.has_correlations():
            n_params = len(self._parameter_order)
            corr_matrix = self.get_correlation_matrix()
            n_corr = np.sum(np.abs(corr_matrix - np.eye(n_params)) > 1e-10) // 2
            corr_str = f"\nCorrelations: {n_corr} pairwise correlation(s) set"
        else:
            corr_str = ""
        
        return f"Sampler(\nParameters:\n{params_str}{derived_str}{corr_str}\n)"
