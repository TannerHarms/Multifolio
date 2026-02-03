# Sampling API Reference

Complete API documentation for Multifolio's `Sampler` class.

## Quick Links

- [Constructor](#constructor)
- [Adding Parameters](#adding-parameters)
- [Correlations](#correlations)
- [Derived Parameters](#derived-parameters)
- [Constraints](#constraints)
- [Generation Methods](#generation-methods)
- [Quality Metrics](#quality-metrics)
- [Visualizations](#visualizations)
- [Save/Load](#saveload)
- [Distributions](#distributions)

---

## Constructor

### `Sampler(random_seed=None)`

Create a new sampler instance.

**Parameters:**
- `random_seed` (int, optional): Seed for reproducibility. If None, uses random seed.

**Returns:** `Sampler` instance

**Example:**
```python
sampler = Sampler(random_seed=42)  # Reproducible
sampler = Sampler()                # Random each time
```

---

## Adding Parameters

### `add_parameter(name, distribution)`

Add a parameter with specified distribution.

**Parameters:**
- `name` (str): Parameter name (must be unique, valid Python identifier)
- `distribution` (Distribution): Distribution object (e.g., `UniformDistribution(0, 10)`)

**Returns:** None

**Raises:**
- `ValueError`: If parameter name already exists or is invalid

**Example:**
```python
from multifolio.core.sampling.distributions import UniformDistribution, NormalDistribution

sampler.add_parameter('temperature', UniformDistribution(20, 100))
sampler.add_parameter('pressure', NormalDistribution(1.0, 0.2))
```

---

## Correlations

### `set_correlation(param1, param2, correlation)`

Set pairwise correlation between two parameters.

**Parameters:**
- `param1` (str): First parameter name
- `param2` (str): Second parameter name
- `correlation` (float): Pearson correlation coefficient, range [-1, 1]

**Returns:** None

**Raises:**
- `ValueError`: If parameters don't exist or correlation is out of range

**Example:**
```python
sampler.set_correlation('temperature', 'pressure', 0.7)
```

### `set_correlation_matrix(parameters, matrix)`

Set full correlation matrix for multiple parameters.

**Parameters:**
- `parameters` (list of str): Parameter names in order
- `matrix` (array-like): Correlation matrix (n×n, symmetric, diagonal=1.0)

**Returns:** None

**Raises:**
- `ValueError`: If matrix is invalid (not symmetric, not positive definite, wrong size)

**Example:**
```python
import numpy as np

corr_matrix = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])
sampler.set_correlation_matrix(['A', 'B', 'C'], corr_matrix)
```

### `get_correlation_matrix()`

Get current correlation matrix.

**Parameters:** None

**Returns:** `np.ndarray` - Correlation matrix for all parameters

**Example:**
```python
matrix = sampler.get_correlation_matrix()
print(matrix)
```

---

## Derived Parameters

### `add_derived_parameter(name, formula)`

Add a parameter computed from other parameters.

**Parameters:**
- `name` (str): Derived parameter name (must be unique)
- `formula` (str or callable): 
  - String: Python expression with parameter names (e.g., `'X + Y'`)
  - Callable: Function taking parameter names as kwargs, returning array

**Returns:** None

**Raises:**
- `ValueError`: If name already exists or formula is invalid

**Example:**
```python
# String formula (preferred, can be saved)
sampler.add_derived_parameter('area', 'length * width')
sampler.add_derived_parameter('distance', 'np.sqrt(x**2 + y**2)')

# Callable (cannot be saved to JSON)
def custom_calc(length, width, height):
    return length * width * height

sampler.add_derived_parameter('volume', custom_calc)
```

**Available in formulas:**
- All parameter names
- `np`: NumPy module
- Standard Python operators and functions

---

## Constraints

### `add_constraint(expression)`

Add constraint that samples must satisfy.

**Parameters:**
- `expression` (str): Python expression evaluating to boolean (e.g., `'X + Y < 10'`)

**Returns:** None

**Raises:**
- `ValueError`: If expression is invalid

**Example:**
```python
sampler.add_constraint('X > 0')
sampler.add_constraint('X + Y < 10')
sampler.add_constraint('(X > 5) and (Y < 8)')
sampler.add_constraint('np.sqrt(X**2 + Y**2) < 5')
```

**Notes:**
- All constraints must be satisfied (AND logic)
- Constraints filter generated samples (may reduce performance)
- Can reference derived parameters

---

## Generation Methods

### `generate(n, method='random', return_type='dict')`

Generate samples using specified method.

**Parameters:**
- `n` (int): Number of samples to generate
- `method` (str): Sampling method
  - `'random'`: Pseudo-random (default)
  - `'sobol'`: Sobol sequence (QMC)
  - `'halton'`: Halton sequence (QMC)
  - `'lhs'`: Latin Hypercube Sampling
- `return_type` (str): Output format
  - `'dict'`: Dictionary of arrays (default)
  - `'dataframe'`: pandas DataFrame

**Returns:**
- If `return_type='dict'`: `dict` with parameter names as keys, numpy arrays as values
- If `return_type='dataframe'`: `pandas.DataFrame` with columns for each parameter

**Raises:**
- `ValueError`: If method is unknown or n is invalid

**Example:**
```python
# Dictionary output
samples = sampler.generate(n=1000, method='random')
# {'X': array([...]), 'Y': array([...])}

# DataFrame output
df = sampler.generate(n=1000, method='sobol', return_type='dataframe')
# DataFrame with columns: X, Y, ...

# QMC methods
samples = sampler.generate(n=1024, method='sobol')  # Best: n = 2^k
samples = sampler.generate(n=1000, method='halton')
samples = sampler.generate(n=1000, method='lhs')
```

### `generate_stratified(n, strata_per_param, method='random', return_type='dict', random_seed=None)`

Generate stratified samples with guaranteed coverage.

**Parameters:**
- `n` (int): Number of samples (should match total strata if possible)
- `strata_per_param` (int or dict): 
  - int: Same number of strata for all parameters
  - dict: {param_name: n_strata} for different strata per parameter
- `method` (str): Sampling within strata
  - `'random'`: Uniform random within each stratum (default)
  - `'center'`: Deterministic center points
  - `'jittered'`: Center ± 10% random noise
- `return_type` (str): `'dict'` or `'dataframe'`
- `random_seed` (int, optional): Override sampler's random seed

**Returns:** Samples in specified format

**Raises:**
- `ValueError`: If parameters invalid or strata count mismatch

**Example:**
```python
# Uniform strata (5×5×5 = 125 strata)
samples = sampler.generate_stratified(n=125, strata_per_param=5)

# Different strata per parameter
samples = sampler.generate_stratified(
    n=60,
    strata_per_param={'X': 3, 'Y': 4, 'Z': 5}  # 3×4×5 = 60
)

# Deterministic center points (reproducible)
experiments = sampler.generate_stratified(
    n=27,
    strata_per_param=3,
    method='center',
    return_type='dataframe'
)
```

### `bootstrap_resample(data, n=None, return_type='dict', random_seed=None)`

Resample with replacement (bootstrap).

**Parameters:**
- `data` (dict or DataFrame): Original samples to resample from
- `n` (int, optional): Number of bootstrap samples. If None, same as original size
- `return_type` (str): `'dict'` or `'dataframe'`
- `random_seed` (int, optional): Override sampler's random seed

**Returns:** Bootstrap samples in specified format

**Example:**
```python
# Generate original samples
samples = sampler.generate(n=500, return_type='dataframe')

# Bootstrap resample (same size)
boot = sampler.bootstrap_resample(samples, return_type='dataframe')

# Bootstrap with different size
boot = sampler.bootstrap_resample(samples, n=1000, return_type='dataframe')

# Bootstrap confidence intervals
bootstrap_means = []
for i in range(1000):
    boot = sampler.bootstrap_resample(samples, random_seed=42+i, return_type='dataframe')
    bootstrap_means.append(boot['X'].mean())

ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
```

---

## Quality Metrics

### `compute_quality_metrics(samples, metrics=None)`

Compute quality metrics for generated samples.

**Parameters:**
- `samples` (dict or DataFrame): Generated samples to evaluate
- `metrics` (list of str, optional): Metrics to compute. If None, computes all.
  - `'coverage'`: Fraction of parameter space occupied
  - `'discrepancy'`: Star discrepancy (uniformity)
  - `'correlation_error'`: Deviation from target correlations
  - `'distribution_ks'`: Kolmogorov-Smirnov test per parameter
  - `'uniformity'`: Chi-square test per parameter

**Returns:** `dict` with requested metrics

**Example:**
```python
samples = sampler.generate(n=1000, return_type='dataframe')

# All metrics
metrics = sampler.compute_quality_metrics(samples)

# Specific metrics
metrics = sampler.compute_quality_metrics(
    samples,
    metrics=['coverage', 'discrepancy']
)

# Access results
print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Discrepancy: {metrics['discrepancy']:.4f}")

if 'correlation_error' in metrics:
    print(f"Correlation RMSE: {metrics['correlation_error']['rmse']:.4f}")

for param, result in metrics.get('distribution_ks', {}).items():
    print(f"{param}: {'PASS' if result['passes'] else 'FAIL'}")
```

**Metric Details:**

**Coverage:**
```python
{
    'coverage': 0.85,  # Fraction of bins occupied (0-1)
    'coverage_details': {
        'occupied_bins': 850,
        'total_bins': 1000,
        'bins_per_dimension': 10
    }
}
```

**Discrepancy:**
```python
{
    'discrepancy': 0.0923  # Star discrepancy (lower is better)
}
```

**Correlation Error:**
```python
{
    'correlation_error': {
        'rmse': 0.0234,           # Root mean square error
        'max_abs_error': 0.0456,  # Maximum absolute error
        'mean_abs_error': 0.0189  # Mean absolute error
    }
}
```

**Distribution KS:**
```python
{
    'distribution_ks': {
        'param1': {
            'statistic': 0.023,  # KS statistic
            'pvalue': 0.456,     # p-value
            'passes': True       # True if p > 0.05
        },
        # ... for each parameter
    }
}
```

**Uniformity:**
```python
{
    'uniformity': {
        'param1': {
            'statistic': 12.34,  # Chi-square statistic
            'pvalue': 0.234,     # p-value
            'passes': True       # True if p > 0.05
        },
        # ... for each parameter
    }
}
```

---

## Visualizations

**Note:** Requires matplotlib: `pip install matplotlib`

### `plot_distributions(samples, parameters=None, figsize=None, bins=30)`

Plot histograms of parameter distributions.

**Parameters:**
- `samples` (dict or DataFrame): Samples to plot
- `parameters` (list of str, optional): Parameters to plot. If None, plots all
- `figsize` (tuple, optional): Figure size (width, height). Auto-sized if None
- `bins` (int): Number of histogram bins (default: 30)

**Returns:** `matplotlib.figure.Figure`

**Example:**
```python
import matplotlib.pyplot as plt

samples = sampler.generate(n=1000, return_type='dataframe')

# Plot all parameters
fig = sampler.plot_distributions(samples)
plt.show()

# Plot specific parameters
fig = sampler.plot_distributions(
    samples,
    parameters=['X', 'Y'],
    figsize=(12, 5),
    bins=50
)
fig.savefig('distributions.png', dpi=150)
plt.close()
```

### `plot_correlation_matrix(samples, parameters=None, figsize=None, annot=True)`

Plot correlation matrix heatmap.

**Parameters:**
- `samples` (dict or DataFrame): Samples to plot
- `parameters` (list of str, optional): Parameters to include. If None, uses all
- `figsize` (tuple, optional): Figure size. Auto-sized if None
- `annot` (bool): If True, annotate cells with correlation values (default: True)

**Returns:** `matplotlib.figure.Figure`

**Example:**
```python
fig = sampler.plot_correlation_matrix(
    samples,
    figsize=(10, 8),
    annot=True
)
plt.show()
```

### `plot_pairwise(samples, parameters=None, figsize=None, alpha=0.6, point_size=20)`

Plot pairwise scatter plot matrix.

**Parameters:**
- `samples` (dict or DataFrame): Samples to plot
- `parameters` (list of str, optional): Parameters to plot. If None, uses all (max 6 recommended)
- `figsize` (tuple, optional): Figure size. Auto-sized if None
- `alpha` (float): Point transparency, 0-1 (default: 0.6)
- `point_size` (float): Marker size (default: 20)

**Returns:** `matplotlib.figure.Figure`

**Example:**
```python
# Limit to specific parameters (avoid clutter)
fig = sampler.plot_pairwise(
    samples,
    parameters=['X', 'Y', 'Z'],
    figsize=(12, 12),
    alpha=0.5,
    point_size=15
)
plt.show()
```

**Note:** Shows scatter plots off-diagonal, histograms on diagonal.

---

## Save/Load

### `save_config(filepath)`

Save sampler configuration to JSON.

**Parameters:**
- `filepath` (str): Path to save JSON file

**Returns:** None

**Raises:**
- `IOError`: If file cannot be written

**What gets saved:**
- Parameters and distributions
- Correlations
- Derived parameters (string formulas only, not callables)
- Constraints
- Random seed

**Example:**
```python
sampler.save_config('my_config.json')
```

### `load_config(filepath)` [classmethod]

Load sampler configuration from JSON.

**Parameters:**
- `filepath` (str): Path to JSON config file

**Returns:** `Sampler` instance with loaded configuration

**Raises:**
- `IOError`: If file cannot be read
- `ValueError`: If config is invalid

**Example:**
```python
sampler = Sampler.load_config('my_config.json')
samples = sampler.generate(n=1000)
```

### `save_samples(filepath, samples, format='auto', **kwargs)`

Save samples to file.

**Parameters:**
- `filepath` (str): Path to save file
- `samples` (dict or DataFrame): Samples to save
- `format` (str): File format
  - `'auto'`: Detect from file extension (default)
  - `'csv'`: CSV format
  - `'hdf5'`: HDF5 format
- `**kwargs`: Format-specific options
  - For HDF5: `compression='gzip'`, `compression_level=6`

**Returns:** None

**Raises:**
- `ValueError`: If format is unknown
- `IOError`: If file cannot be written

**Example:**
```python
samples = sampler.generate(n=10000, return_type='dataframe')

# CSV (auto-detect)
sampler.save_samples('data.csv', samples)

# HDF5 with compression
sampler.save_samples(
    'data.h5',
    samples,
    format='hdf5',
    compression='gzip',
    compression_level=6
)
```

**HDF5 Requirements:** `pip install tables` or `pip install h5py`

### `load_samples(filepath, format='auto')`

Load samples from file.

**Parameters:**
- `filepath` (str): Path to file
- `format` (str): File format (`'auto'`, `'csv'`, `'hdf5'`)

**Returns:** `pandas.DataFrame`

**Raises:**
- `ValueError`: If format is unknown
- `IOError`: If file cannot be read

**Example:**
```python
samples = sampler.load_samples('data.csv')
samples = sampler.load_samples('data.h5', format='hdf5')
```

---

## Distributions

All distributions are in `multifolio.core.sampling.distributions` module.

### `UniformDistribution(min_val, max_val)`

Uniform (flat) distribution.

**Parameters:**
- `min_val` (float): Minimum value
- `max_val` (float): Maximum value

**Example:**
```python
from multifolio.core.sampling.distributions import UniformDistribution
dist = UniformDistribution(0, 10)
```

### `NormalDistribution(mean, std)`

Normal (Gaussian) distribution.

**Parameters:**
- `mean` (float): Mean (center)
- `std` (float): Standard deviation (spread)

**Example:**
```python
from multifolio.core.sampling.distributions import NormalDistribution
dist = NormalDistribution(mean=50, std=10)
```

### `LogNormalDistribution(mean, std)`

Log-normal distribution (positive, right-skewed).

**Parameters:**
- `mean` (float): Mean of underlying normal
- `std` (float): Std dev of underlying normal

**Example:**
```python
from multifolio.core.sampling.distributions import LogNormalDistribution
dist = LogNormalDistribution(mean=1, std=0.5)
```

### `ExponentialDistribution(rate)`

Exponential distribution (time between events).

**Parameters:**
- `rate` (float): Rate parameter (λ)

**Example:**
```python
from multifolio.core.sampling.distributions import ExponentialDistribution
dist = ExponentialDistribution(rate=0.5)
```

### `BetaDistribution(alpha, beta)`

Beta distribution (bounded [0, 1]).

**Parameters:**
- `alpha` (float): Shape parameter α
- `beta` (float): Shape parameter β

**Example:**
```python
from multifolio.core.sampling.distributions import BetaDistribution
dist = BetaDistribution(alpha=2, beta=5)
```

### `GammaDistribution(shape, scale)`

Gamma distribution (positive, flexible shape).

**Parameters:**
- `shape` (float): Shape parameter (k)
- `scale` (float): Scale parameter (θ)

**Example:**
```python
from multifolio.core.sampling.distributions import GammaDistribution
dist = GammaDistribution(shape=2, scale=2)
```

### `WeibullDistribution(shape, scale)`

Weibull distribution (reliability, lifetime).

**Parameters:**
- `shape` (float): Shape parameter
- `scale` (float): Scale parameter

**Example:**
```python
from multifolio.core.sampling.distributions import WeibullDistribution
dist = WeibullDistribution(shape=1.5, scale=1.0)
```

### `TriangularDistribution(low, mode, high)`

Triangular distribution (min/most likely/max).

**Parameters:**
- `low` (float): Minimum value
- `mode` (float): Most likely value (peak)
- `high` (float): Maximum value

**Example:**
```python
from multifolio.core.sampling.distributions import TriangularDistribution
dist = TriangularDistribution(low=10, mode=15, high=25)
```

### `CustomDistribution(sample_function)`

Custom user-defined distribution.

**Parameters:**
- `sample_function` (callable): Function taking n (int) and returning array of n samples

**Example:**
```python
import numpy as np
from multifolio.core.sampling.distributions import CustomDistribution

def my_sampler(n):
    """Custom sampling logic."""
    return np.random.exponential(2, n) + 5

dist = CustomDistribution(my_sampler)
```

---

## Notes

### Thread Safety

`Sampler` is **not thread-safe**. Create separate instances for concurrent use:

```python
# ✗ Don't share sampler across threads
# ✓ Create separate instances
sampler1 = Sampler(random_seed=42)
sampler2 = Sampler(random_seed=43)
```

### Performance Tips

1. **Large samples**: Use HDF5 instead of CSV
2. **Many parameters**: Limit visualizations to subset
3. **Tight constraints**: Check acceptance rate first
4. **QMC methods**: Use n = 2^k for Sobol (512, 1024, 2048)
5. **Correlations**: Need larger n for accuracy (>1000)

### Reproducibility

Always set `random_seed` for reproducible results:

```python
sampler = Sampler(random_seed=42)
```

Same seed + same operations = same results.

---

## See Also

- [Sampling Guide](SAMPLING_GUIDE.md) - Comprehensive usage guide with examples
- [Examples](../examples/) - Working code examples
- [Tests](../backend/tests/unit/sampling/) - Test suite with usage patterns
