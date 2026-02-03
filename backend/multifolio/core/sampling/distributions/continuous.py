"""Continuous probability distributions."""

from typing import Optional, Callable, Union, Literal
import numpy as np
from scipy import stats
from pathlib import Path
import pandas as pd

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


class ExponentialDistribution(Distribution):
    """
    Exponential distribution for modeling time between events.
    
    Parameters
    ----------
    rate : float
        Rate parameter (λ > 0). Mean = 1/rate.
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> dist = ExponentialDistribution(rate=0.5)  # Mean = 2.0
    >>> samples = dist.sample(size=1000)
    >>> print(f"Mean: {samples.mean():.2f}")  # Should be ~2.0
    """
    
    def __init__(self, rate: float, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        
        if rate <= 0:
            raise ValueError(f"rate ({rate}) must be positive")
        
        self.rate = rate
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from exponential distribution."""
        return self._rng.exponential(scale=1/self.rate, size=size)
    
    def __repr__(self) -> str:
        return f"ExponentialDistribution(rate={self.rate})"


class GammaDistribution(Distribution):
    """
    Gamma distribution for modeling wait times and other positive values.
    
    Parameters
    ----------
    shape : float
        Shape parameter (k > 0)
    scale : float
        Scale parameter (θ > 0). Mean = shape * scale.
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> dist = GammaDistribution(shape=2, scale=2)  # Mean = 4.0
    >>> samples = dist.sample(size=1000)
    """
    
    def __init__(self, shape: float, scale: float, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        
        if shape <= 0:
            raise ValueError(f"shape ({shape}) must be positive")
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")
        
        self.shape = shape
        self.scale = scale
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from gamma distribution."""
        return self._rng.gamma(shape=self.shape, scale=self.scale, size=size)
    
    def __repr__(self) -> str:
        return f"GammaDistribution(shape={self.shape}, scale={self.scale})"


class LogNormalDistribution(Distribution):
    """
    Log-normal distribution for modeling multiplicative processes.
    
    Parameters
    ----------
    mu : float
        Mean of underlying normal distribution (not the mean of lognormal!)
    sigma : float
        Standard deviation of underlying normal distribution (σ > 0)
    random_seed : int, optional
        Random seed for reproducibility
    
    Notes
    -----
    Mean of lognormal: exp(mu + sigma²/2)
    Variance of lognormal: (exp(sigma²) - 1) * exp(2*mu + sigma²)
    
    Examples
    --------
    >>> dist = LogNormalDistribution(mu=0, sigma=1)
    >>> samples = dist.sample(size=1000)
    >>> # Mean is exp(0 + 1/2) = exp(0.5) ≈ 1.65
    """
    
    def __init__(self, mu: float, sigma: float, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        
        if sigma <= 0:
            raise ValueError(f"sigma ({sigma}) must be positive")
        
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from log-normal distribution."""
        return self._rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)
    
    def __repr__(self) -> str:
        return f"LogNormalDistribution(mu={self.mu}, sigma={self.sigma})"


class WeibullDistribution(Distribution):
    """
    Weibull distribution for modeling failure times and reliability.
    
    Parameters
    ----------
    shape : float
        Shape parameter (k > 0). k < 1: failure rate decreases,
        k = 1: constant (exponential), k > 1: failure rate increases.
    scale : float
        Scale parameter (λ > 0)
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> dist = WeibullDistribution(shape=1.5, scale=2.0)
    >>> samples = dist.sample(size=1000)
    """
    
    def __init__(self, shape: float, scale: float, random_seed: Optional[int] = None):
        super().__init__(random_seed)
        
        if shape <= 0:
            raise ValueError(f"shape ({shape}) must be positive")
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")
        
        self.shape = shape
        self.scale = scale
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from Weibull distribution."""
        return self.scale * self._rng.weibull(a=self.shape, size=size)
    
    def __repr__(self) -> str:
        return f"WeibullDistribution(shape={self.shape}, scale={self.scale})"


class TriangularDistribution(Distribution):
    """
    Triangular distribution for modeling with min/mode/max estimates.
    
    Parameters
    ----------
    low : float
        Minimum value
    mode : float
        Most likely value (peak of triangle)
    high : float
        Maximum value
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> dist = TriangularDistribution(low=1, mode=3, high=6)
    >>> samples = dist.sample(size=1000)
    """
    
    def __init__(self, low: float, mode: float, high: float, 
                 random_seed: Optional[int] = None):
        super().__init__(random_seed)
        
        if not (low <= mode <= high):
            raise ValueError(
                f"Must have low ({low}) <= mode ({mode}) <= high ({high})"
            )
        
        self.low = low
        self.mode = mode
        self.high = high
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from triangular distribution."""
        return self._rng.triangular(left=self.low, mode=self.mode, 
                                    right=self.high, size=size)
    
    def __repr__(self) -> str:
        return f"TriangularDistribution(low={self.low}, mode={self.mode}, high={self.high})"


class CustomContinuousDistribution(Distribution):
    """
    Custom continuous distribution from data or function.
    
    Can be created from:
    - Array/list of sample data (various fitting methods available)
    - CSV/text file with sample data
    - Custom function that generates samples
    
    Parameters
    ----------
    data : np.ndarray, list, str, Path, or callable, optional
        Source of distribution:
        - Array/list: Sample data to fit distribution
        - str/Path: Path to CSV file with data
        - callable: Function that returns samples when called with size parameter
    method : {'kde', 'empirical_cdf', 'spline', 'histogram'}, optional
        Method for fitting distribution from data (ignored for callable):
        - 'kde': Kernel Density Estimation (smooth, can extend beyond data)
        - 'empirical_cdf': Empirical CDF interpolation (stays within data range)
        - 'spline': Cubic spline interpolation of PDF (smooth, bounded)
        - 'histogram': Histogram-based sampling (fast, discretized)
        Default is 'kde'.
    negative_handling : {'truncate', 'shift', 'allow'}, optional
        How to handle negative values:
        - 'truncate': Replace negative values with 0
        - 'shift': Shift entire distribution so minimum is 0
        - 'allow': Allow negative values (default)
    bandwidth : float or str, optional
        KDE bandwidth. Use 'scott' or 'silverman' for automatic selection.
        Only used when method='kde'. Default is 'scott'.
    bins : int or str, optional
        Number of bins for histogram method. Can be int or 'auto', 'sqrt', etc.
        Only used when method='histogram'. Default is 'auto'.
    random_seed : int, optional
        Random seed for reproducibility
    
    Examples
    --------
    >>> # KDE (default) - smooth but can extend beyond data
    >>> data = np.random.gamma(2, 2, size=1000)
    >>> dist = CustomContinuousDistribution(data=data, method='kde')
    >>> 
    >>> # Empirical CDF - stays within data range (good for bounded data)
    >>> dist = CustomContinuousDistribution(data=data, method='empirical_cdf')
    >>> samples = dist.sample(size=100)
    >>> assert samples.min() >= data.min()  # Always within bounds
    >>> 
    >>> # Spline - smooth and bounded
    >>> dist = CustomContinuousDistribution(data=data, method='spline')
    >>> 
    >>> # Histogram - fast, discretized
    >>> dist = CustomContinuousDistribution(data=data, method='histogram', bins=50)
    >>> 
    >>> # From custom function
    >>> def my_sampler(size):
    ...     return np.random.exponential(2, size) + 5
    >>> dist = CustomContinuousDistribution(data=my_sampler)
    
    Notes
    -----
    Method selection guide:
    - **kde**: Best for unknown shapes, smooth results, but can produce values
      outside data range (e.g., negatives from positive data)
    - **empirical_cdf**: Best for bounded quantities (times, concentrations),
      guaranteed to stay within observed range
    - **spline**: Good compromise - smooth and bounded, but requires enough data
    - **histogram**: Fastest, good for quick approximation, creates discrete bins
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list, str, Path, Callable] = None,
        method: Literal['kde', 'empirical_cdf', 'spline', 'histogram'] = 'kde',
        negative_handling: Literal['truncate', 'shift', 'allow'] = 'allow',
        bandwidth: Union[float, str] = 'scott',
        bins: Union[int, str] = 'auto',
        random_seed: Optional[int] = None
    ):
        super().__init__(random_seed)
        
        if data is None:
            raise ValueError("Must provide data source")
        
        if method not in ['kde', 'empirical_cdf', 'spline', 'histogram']:
            raise ValueError(
                f"method must be 'kde', 'empirical_cdf', 'spline', or 'histogram', "
                f"got '{method}'"
            )
        
        if negative_handling not in ['truncate', 'shift', 'allow']:
            raise ValueError(
                f"negative_handling must be 'truncate', 'shift', or 'allow', "
                f"got '{negative_handling}'"
            )
        
        self.method = method
        self.negative_handling = negative_handling
        self.bandwidth = bandwidth
        self.bins = bins
        self._shift_amount = 0
        
        # Determine type of data source
        if callable(data):
            self._sample_function = data
            self._source_type = 'function'
            # Other attributes not needed for function-based
        else:
            self._sample_function = None
            self._source_type = 'data'
            
            # Load data
            if isinstance(data, (str, Path)):
                data = self._load_from_file(data)
            else:
                data = np.asarray(data)
            
            # Handle negative values
            data = self._handle_negatives(data)
            
            # Fit using selected method
            self._fit_distribution(data)
    
    def _load_from_file(self, filepath: Union[str, Path]) -> np.ndarray:
        """Load data from CSV or text file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Try reading as CSV
            df = pd.read_csv(filepath)
            # Use first column if multiple columns
            data = df.iloc[:, 0].values
        except Exception:
            # Try reading as plain text (one value per line)
            data = np.loadtxt(filepath)
        
        return data
    
    def _handle_negatives(self, data: np.ndarray) -> np.ndarray:
        """Handle negative values according to strategy."""
        if self.negative_handling == 'allow':
            return data
        
        min_val = data.min()
        
        if self.negative_handling == 'truncate':
            if min_val < 0:
                data = np.maximum(data, 0)
        elif self.negative_handling == 'shift':
            if min_val < 0:
                self._shift_amount = -min_val
                data = data + self._shift_amount
        
        return data
    
    def _fit_distribution(self, data: np.ndarray):
        """Fit distribution using selected method."""
        if self.method == 'kde':
            self._fit_kde(data)
        elif self.method == 'empirical_cdf':
            self._fit_empirical_cdf(data)
        elif self.method == 'spline':
            self._fit_spline(data)
        elif self.method == 'histogram':
            self._fit_histogram(data)
    
    def _fit_kde(self, data: np.ndarray):
        """Fit kernel density estimate to data."""
        from scipy.stats import gaussian_kde
        
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to fit KDE")
        
        self._kde = gaussian_kde(data, bw_method=self.bandwidth)
        self._data_min = data.min()
        self._data_max = data.max()
    
    def _fit_empirical_cdf(self, data: np.ndarray):
        """Fit empirical CDF for inverse transform sampling."""
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to fit empirical CDF")
        
        # Sort data for CDF
        self._sorted_data = np.sort(data)
        self._data_min = self._sorted_data[0]
        self._data_max = self._sorted_data[-1]
        
        # Create empirical CDF values
        n = len(self._sorted_data)
        self._cdf_values = np.arange(1, n + 1) / n
    
    def _fit_spline(self, data: np.ndarray):
        """Fit cubic spline to PDF estimated from histogram."""
        from scipy.interpolate import CubicSpline
        from scipy.integrate import cumulative_trapezoid
        
        if len(data) < 10:
            raise ValueError("Need at least 10 data points to fit spline")
        
        # Create histogram for PDF estimation
        n_bins = min(50, len(data) // 10)
        hist, edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        # Ensure positive PDF
        hist = np.maximum(hist, 1e-10)
        
        # Fit spline to PDF
        self._pdf_spline = CubicSpline(bin_centers, hist, extrapolate=False)
        
        # Create CDF by integrating PDF
        x_dense = np.linspace(bin_centers[0], bin_centers[-1], 1000)
        pdf_dense = np.maximum(self._pdf_spline(x_dense), 0)
        cdf_dense = np.concatenate([[0], cumulative_trapezoid(pdf_dense, x_dense)])
        cdf_dense = cdf_dense / cdf_dense[-1]  # Normalize to [0, 1]
        
        # Store for inverse CDF
        self._spline_x = x_dense
        self._spline_cdf = cdf_dense
        self._data_min = bin_centers[0]
        self._data_max = bin_centers[-1]
    
    def _fit_histogram(self, data: np.ndarray):
        """Fit histogram for bin-based sampling."""
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to fit histogram")
        
        # Create histogram
        hist, edges = np.histogram(data, bins=self.bins)
        
        # Store bin edges and probabilities
        self._hist_edges = edges
        self._hist_probs = hist / hist.sum()
        self._bin_centers = (edges[:-1] + edges[1:]) / 2
        self._data_min = edges[0]
        self._data_max = edges[-1]
    
    def sample(self, size: Union[int, tuple] = 1) -> np.ndarray:
        """Generate samples from custom distribution."""
        if self._sample_function is not None:
            # Use custom function
            samples = self._sample_function(size)
            samples = np.asarray(samples)
        elif self.method == 'kde':
            samples = self._sample_kde(size)
        elif self.method == 'empirical_cdf':
            samples = self._sample_empirical_cdf(size)
        elif self.method == 'spline':
            samples = self._sample_spline(size)
        elif self.method == 'histogram':
            samples = self._sample_histogram(size)
        
        # Handle negatives in output
        if self.negative_handling == 'truncate':
            samples = np.maximum(samples, 0)
        elif self.negative_handling == 'shift' and hasattr(self, '_shift_amount'):
            # Unshift samples back (data was shifted up during fitting)
            samples = samples - self._shift_amount
            # Also truncate any remaining negatives from KDE extrapolation
            samples = np.maximum(samples, 0)
        
        return samples
    
    def _sample_kde(self, size: Union[int, tuple]) -> np.ndarray:
        """Sample using KDE."""
        return self._kde.resample(size, seed=self._get_seed())[0]
    
    def _sample_empirical_cdf(self, size: Union[int, tuple]) -> np.ndarray:
        """Sample using inverse empirical CDF (quantile function)."""
        # Generate uniform random numbersmethod='{self.method}', negative_handling='{self.negative_handling}'
        u = self._rng.uniform(0, 1, size=size)
        
        # Use linear interpolation of empirical quantile function
        # This is equivalent to inverse CDF sampling
        samples = np.interp(u, self._cdf_values, self._sorted_data)
        
        return samples
    
    def _sample_spline(self, size: Union[int, tuple]) -> np.ndarray:
        """Sample using spline-fitted distribution."""
        # Generate uniform random numbers
        u = self._rng.uniform(0, 1, size=size)
        
        # Inverse CDF using spline
        samples = np.interp(u, self._spline_cdf, self._spline_x)
        
        return samples
    
    def _sample_histogram(self, size: Union[int, tuple]) -> np.ndarray:
        """Sample using histogram bins."""
        # Choose bins according to probabilities
        bin_indices = self._rng.choice(len(self._hist_probs), size=size, p=self._hist_probs)
        
        # Sample uniformly within each chosen bin
        samples = self._rng.uniform(
            self._hist_edges[bin_indices],
            self._hist_edges[bin_indices + 1]
        )
        
        return samples
    
    def _get_seed(self) -> Optional[int]:
        """Get a seed for KDE resampling."""
        if self.random_seed is not None:
            # Generate a seed from current RNG state
            return self._rng.integers(0, 2**31)
        return None
    
    def __repr__(self) -> str:
        if self._source_type == 'function':
            return f"CustomContinuousDistribution(source=function, negative_handling='{self.negative_handling}')"
        else:
            return f"CustomContinuousDistribution(source=data, method='{self.method}', negative_handling='{self.negative_handling}')"
