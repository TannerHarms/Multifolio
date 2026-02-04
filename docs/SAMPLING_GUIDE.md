# Multifolio Sampling Guide

Complete guide to parameter sampling in Multifolio, from basic usage to advanced features.

## Quick Navigation

- [Getting Started](#getting-started) - Basic sampling in 5 minutes
- [Core Concepts](#core-concepts) - Understanding the sampler
- [Distribution Types](#distribution-types) - All available distributions
- [Sampling Methods](#sampling-methods) - Random, QMC, stratified, LHS3
- [Correlations](#correlations) - Model parameter dependencies
- [Derived Parameters](#derived-parameters) - Computed parameters
- [Constraints](#constraints) - Filter samples
- [Quality &amp; Validation](#quality--validation) - Assess sample quality
- [Visualizations](#visualizations) - Plot and analyze samples
- [Save/Load](#saveload) - Persistence and reproducibility
- [Best Practices](#best-practices) - Tips and guidelines
- [Examples](#examples) - Complete working examples

---

## Getting Started

### Installation

```bash
pip install multifolio

# Optional: for HDF5 and visualizations
pip install tables matplotlib  # or: pip install h5py matplotlib
```

### 5-Minute Quickstart

```python
from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import UniformDistribution, NormalDistribution

# 1. Create sampler
sampler = Sampler(random_seed=42)

# 2. Add parameters
sampler.add_parameter('temperature', UniformDistribution(20, 100))
sampler.add_parameter('pressure', NormalDistribution(1.0, 0.2))

# 3. Generate samples
samples = sampler.generate(n=1000, return_type='dataframe')

# 4. Use the data
print(samples.describe())
samples.to_csv('samples.csv', index=False)
```

**Output:** A DataFrame with 1000 rows and 2 columns (temperature, pressure).

---

## Core Concepts

### The Sampler Object

The `Sampler` is your main interface for generating parameter samples:

```python
sampler = Sampler(random_seed=42)  # Reproducibility
```

**Key Properties:**

- **Parameters**: Independent variables with distributions
- **Correlations**: Dependencies between parameters
- **Derived Parameters**: Computed from other parameters
- **Constraints**: Conditions samples must satisfy

### Workflow

```
1. Configure → 2. Generate → 3. Validate → 4. Use
    ↓             ↓              ↓           ↓
 Add params    Sample data   Check quality  Analysis
 Set corr.     Choose method  Plot results  Simulation
 Add derived   Set size                     Export
 Add constr.
```

### Return Types

All generation methods support two return types:

```python
# Dictionary (default) - more flexible
samples_dict = sampler.generate(n=100, return_type='dict')
# {'param1': array([...]), 'param2': array([...])}

# DataFrame - more convenient for analysis
samples_df = sampler.generate(n=100, return_type='dataframe')
# pandas DataFrame with columns for each parameter
```

---

## Distribution Types

### Available Distributions

| Distribution                | Use Case                      | Parameters      |
| --------------------------- | ----------------------------- | --------------- |
| `UniformDistribution`     | Equal probability             | min, max        |
| `NormalDistribution`      | Bell curve, natural variation | mean, std       |
| `LogNormalDistribution`   | Positive, right-skewed        | mean, std       |
| `ExponentialDistribution` | Time between events           | rate            |
| `BetaDistribution`        | Bounded [0,1]                 | alpha, beta     |
| `GammaDistribution`       | Wait times, positive values   | shape, scale    |
| `WeibullDistribution`     | Reliability, lifetime         | shape, scale    |
| `TriangularDistribution`  | Min/mode/max known            | low, mode, high |
| `CustomDistribution`      | User-defined sampling         | callable        |

### Usage Examples

```python
from multifolio.core.sampling.distributions import *

# Uniform - equal probability across range
sampler.add_parameter('X', UniformDistribution(0, 10))

# Normal - most common distribution
sampler.add_parameter('Y', NormalDistribution(mean=50, std=10))

# LogNormal - for positive, skewed data (e.g., income)
sampler.add_parameter('Z', LogNormalDistribution(mean=1, std=0.5))

# Exponential - time between events
sampler.add_parameter('wait_time', ExponentialDistribution(rate=0.5))

# Beta - bounded probability
sampler.add_parameter('prob', BetaDistribution(alpha=2, beta=5))

# Triangular - when you know min/most likely/max
sampler.add_parameter('estimate', TriangularDistribution(low=10, mode=15, high=25))

# Custom - your own sampling function
def custom_sampler(n):
    return np.random.exponential(2, n) + 5

sampler.add_parameter('custom', CustomDistribution(custom_sampler))
```

### Choosing a Distribution

**When you have:**

- **Min and max bounds** → Uniform or Triangular
- **Mean and std dev** → Normal
- **Positive values only** → LogNormal, Exponential, or Gamma
- **Probability/proportion** → Beta
- **Most likely value** → Triangular
- **Custom needs** → CustomDistribution

---

## Sampling Methods

The sampler supports multiple sampling algorithms, each with different properties.

### Random (Monte Carlo)

Standard pseudo-random sampling using NumPy's random number generator.

```python
samples = sampler.generate(n=1000, method='random')
```

**Properties:**

- ✓ Fast
- ✓ Statistically independent
- ✗ Uneven space coverage
- ✗ Clumping possible

**Use when:** Large sample sizes, true randomness needed

### Sobol (Quasi-Monte Carlo)

Low-discrepancy sequence with excellent space-filling properties.

```python
samples = sampler.generate(n=1024, method='sobol')  # Use n = 2^k for best results
```

**Properties:**

- ✓ Excellent space coverage
- ✓ Faster convergence
- ✓ Low discrepancy
- ⚠ Best when n = 2^k (512, 1024, 2048, etc.)

**Use when:** Integration, optimization, small-medium samples

### Halton (Quasi-Monte Carlo)

Another low-discrepancy sequence, works well for any n.

```python
samples = sampler.generate(n=1000, method='halton')
```

**Properties:**

- ✓ Good space coverage
- ✓ Works for any n
- ⚠ Can have correlation artifacts in higher dimensions

**Use when:** Flexible sample size needed

### Latin Hypercube Sampling (LHS)

Stratified sampling ensuring coverage in each dimension.

```python
samples = sampler.generate(n=1000, method='lhs')
```

**Properties:**

- ✓ Good coverage per dimension
- ✓ Stratification guarantees
- ✓ Works for any n
- ⚠ No correlation between dimensions

**Use when:** Need guaranteed coverage per parameter

### Stratified Sampling

Divides parameter space into strata (bins) and samples from each.

```python
# Uniform strata
samples = sampler.generate_stratified(n=125, strata_per_param=5)  # 5^3 = 125 strata

# Different strata per parameter
samples = sampler.generate_stratified(
    n=60,
    strata_per_param={'X': 3, 'Y': 4, 'Z': 5}  # 3×4×5 = 60 strata
)

# Sampling methods within strata
samples = sampler.generate_stratified(n=125, strata_per_param=5, method='center')
```

**Methods:**

- `'random'` (default): Uniform random within each stratum
- `'center'`: Deterministic center points
- `'jittered'`: Center ± 10% random noise

**Properties:**

- ✓ Excellent coverage
- ✓ Works with correlations
- ✓ Reproducible with 'center'
- ⚠ Best when n = strata product

**Use when:** Experimental design, guaranteed coverage, small budgets

### Method Comparison

```python
# Compare different methods
methods = ['random', 'sobol', 'halton', 'lhs']
results = {}

for method in methods:
    samples = sampler.generate(n=1000, method=method)
    metrics = sampler.compute_quality_metrics(samples, metrics=['coverage', 'discrepancy'])
    results[method] = metrics

# Stratified
samples = sampler.generate_stratified(n=1000, strata_per_param=10)
metrics = sampler.compute_quality_metrics(samples, metrics=['coverage', 'discrepancy'])
results['stratified'] = metrics

# Print comparison
for method, m in results.items():
    print(f"{method:12s} Coverage: {m['coverage']:.3f}  Discrepancy: {m['discrepancy']:.4f}")
```

**Typical Results:**

```
Method       Coverage  Discrepancy
random       0.350     0.180
sobol        0.380     0.085
halton       0.370     0.095
lhs          0.360     0.120
stratified   0.950     0.140
```

---

## Correlations

Model dependencies between parameters while preserving their individual distributions.

### Why Correlations Matter

In real systems, parameters are often dependent:

- Temperature and pressure in physical systems
- Asset returns in financial portfolios
- Customer behaviors in markets
- Component failures in reliability analysis

**Ignoring correlations leads to:**

- ❌ Unrealistic scenarios
- ❌ Poor risk estimates
- ❌ Overestimated diversification
- ❌ Incorrect extreme event probabilities

### Basic Usage

```python
# Add parameters
sampler.add_parameter('X', NormalDistribution(0, 1))
sampler.add_parameter('Y', NormalDistribution(0, 1))

# Set correlation (Pearson correlation coefficient)
sampler.set_correlation('X', 'Y', 0.8)  # Strong positive correlation

# Generate correlated samples
samples = sampler.generate(n=1000, return_type='dataframe')

# Verify
print(samples.corr())  # Will show ~0.8 correlation
```

### Correlation Coefficients

```python
# Positive correlation (0 to 1)
sampler.set_correlation('temp', 'pressure', 0.7)  # Move together

# Negative correlation (-1 to 0)
sampler.set_correlation('price', 'demand', -0.6)  # Move opposite

# No correlation (0)
sampler.set_correlation('X', 'Y', 0)  # Independent

# Perfect correlation (±1)
sampler.set_correlation('X', 'Y', 1.0)   # Perfectly aligned
sampler.set_correlation('X', 'Y', -1.0)  # Perfectly opposite
```

**Interpretation:**

- **+1.0**: Perfect positive (Y increases when X increases)
- **+0.7**: Strong positive
- **+0.3**: Weak positive
- **0.0**: No linear relationship
- **-0.3**: Weak negative
- **-0.7**: Strong negative
- **-1.0**: Perfect negative (Y decreases when X increases)

### Multiple Correlations

```python
# Set up multiple correlated parameters
sampler.add_parameter('A', UniformDistribution(0, 10))
sampler.add_parameter('B', UniformDistribution(0, 10))
sampler.add_parameter('C', UniformDistribution(0, 10))

# Create correlation structure
sampler.set_correlation('A', 'B', 0.8)   # A and B strongly correlated
sampler.set_correlation('B', 'C', 0.5)   # B and C moderately correlated
sampler.set_correlation('A', 'C', 0.3)   # A and C weakly correlated

# View full correlation matrix
print(sampler.get_correlation_matrix())
```

### Correlation Matrix

Set all correlations at once:

```python
import numpy as np

# Define correlation matrix
corr_matrix = np.array([
    [1.0, 0.8, 0.3],  # A
    [0.8, 1.0, 0.5],  # B
    [0.3, 0.5, 1.0]   # C
])

sampler.set_correlation_matrix(['A', 'B', 'C'], corr_matrix)
```

**Rules:**

- Must be symmetric (corr[i,j] = corr[j,i])
- Diagonal must be 1.0
- Must be positive definite (valid correlation matrix)

### Works with All Methods

Correlations are preserved across all sampling methods:

```python
sampler.set_correlation('X', 'Y', 0.7)

# All preserve the correlation structure
samples1 = sampler.generate(n=1000, method='random')
samples2 = sampler.generate(n=1024, method='sobol')
samples3 = sampler.generate(n=1000, method='halton')
samples4 = sampler.generate(n=1000, method='lhs')
samples5 = sampler.generate_stratified(n=125, strata_per_param=5)
```

### Technical Details

**Method**: Gaussian copula

- Transforms uniform samples to correlated uniform
- Applies inverse CDF to get target distributions
- Preserves marginal distributions exactly
- Works with any distribution type

**Limitations:**

- Models linear (Pearson) correlation only
- Assumes Gaussian dependence structure
- Cannot model extreme tail dependencies perfectly
- Requires valid (positive definite) correlation matrix

---

## Derived Parameters

Create new parameters computed from existing ones.

### Basic Usage

```python
# Add base parameters
sampler.add_parameter('length', UniformDistribution(1, 10))
sampler.add_parameter('width', UniformDistribution(1, 10))

# Add derived parameter using formula
sampler.add_derived_parameter('area', 'length * width')

# Generate samples (includes derived parameters)
samples = sampler.generate(n=1000, return_type='dataframe')
# Columns: length, width, area
```

### Formula Syntax

Formulas are Python expressions with access to all parameters:

```python
# Arithmetic
sampler.add_derived_parameter('sum', 'X + Y')
sampler.add_derived_parameter('product', 'X * Y')
sampler.add_derived_parameter('ratio', 'X / Y')

# Math functions (numpy available as 'np')
sampler.add_derived_parameter('hypotenuse', 'np.sqrt(X**2 + Y**2)')
sampler.add_derived_parameter('log_ratio', 'np.log(X / Y)')
sampler.add_derived_parameter('angle', 'np.arctan2(Y, X)')

# Conditionals
sampler.add_derived_parameter('category', 'np.where(X > 5, 1, 0)')
sampler.add_derived_parameter('max_value', 'np.maximum(X, Y)')

# Complex formulas
sampler.add_derived_parameter(
    'resistance', 
    '(resistivity * length) / (np.pi * radius**2)'
)
```

### Callable Functions

For more complex logic, use Python functions:

```python
def complex_calculation(length, width, height):
    """Custom calculation with multiple parameters."""
    volume = length * width * height
    surface_area = 2 * (length*width + length*height + width*height)
    return volume / surface_area  # Volume-to-surface ratio

sampler.add_derived_parameter('ratio', complex_calculation)
```

**Note**: Callables work but cannot be saved to JSON config.

### Chained Derivations

Derived parameters can depend on other derived parameters:

```python
sampler.add_parameter('price', UniformDistribution(100, 200))
sampler.add_parameter('quantity', UniformDistribution(10, 100))

# First level
sampler.add_derived_parameter('revenue', 'price * quantity')

# Second level (depends on derived parameter)
sampler.add_derived_parameter('profit', 'revenue * 0.2')  # 20% margin
```

**Important**: Parameters are computed in order of addition.

### Use Cases

**Physics/Engineering:**

```python
sampler.add_parameter('voltage', UniformDistribution(100, 240))
sampler.add_parameter('current', UniformDistribution(1, 10))
sampler.add_derived_parameter('power', 'voltage * current')
sampler.add_derived_parameter('energy', 'power * time')  # If time is a parameter
```

**Finance:**

```python
sampler.add_parameter('stock_price', LogNormalDistribution(100, 20))
sampler.add_parameter('shares', UniformDistribution(100, 1000))
sampler.add_derived_parameter('portfolio_value', 'stock_price * shares')
sampler.add_derived_parameter('gain_loss', 'portfolio_value - initial_investment')
```

**Geometry:**

```python
sampler.add_parameter('radius', UniformDistribution(1, 10))
sampler.add_derived_parameter('circumference', '2 * np.pi * radius')
sampler.add_derived_parameter('area', 'np.pi * radius**2')
sampler.add_derived_parameter('volume', '(4/3) * np.pi * radius**3')
```

---

## Constraints

Filter generated samples based on conditions.

### Basic Usage

```python
# Add parameters
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))

# Add constraint
sampler.add_constraint('X + Y < 15')

# Generate samples (only returns samples satisfying constraint)
samples = sampler.generate(n=1000, return_type='dataframe')
# All samples will have X + Y < 15
```

### Constraint Syntax

Constraints are Python expressions that evaluate to True/False:

```python
# Comparison operators
sampler.add_constraint('X > 5')
sampler.add_constraint('Y <= 100')
sampler.add_constraint('Z != 0')

# Arithmetic relationships
sampler.add_constraint('X + Y < 20')
sampler.add_constraint('X * Y > 50')
sampler.add_constraint('X / Y < 2')

# Multiple conditions with 'and', 'or'
sampler.add_constraint('(X > 5) and (Y < 10)')
sampler.add_constraint('(X > 8) or (Y > 8)')

# Math functions
sampler.add_constraint('np.sqrt(X**2 + Y**2) < 10')  # Inside circle
sampler.add_constraint('np.abs(X - Y) < 2')  # X and Y within 2 units
```

### Multiple Constraints

All constraints must be satisfied:

```python
sampler.add_constraint('X > 0')       # Must be positive
sampler.add_constraint('Y > 0')       # Must be positive
sampler.add_constraint('X + Y < 10')  # Sum must be < 10
sampler.add_constraint('X > Y')       # X must exceed Y

# Generates samples satisfying ALL four constraints
samples = sampler.generate(n=1000)
```

### With Derived Parameters

Constraints can reference derived parameters:

```python
sampler.add_parameter('length', UniformDistribution(1, 10))
sampler.add_parameter('width', UniformDistribution(1, 10))
sampler.add_derived_parameter('area', 'length * width')

# Constraint on derived parameter
sampler.add_constraint('area > 20')
sampler.add_constraint('area < 80')

samples = sampler.generate(n=1000)
# All samples have area between 20 and 80
```

### Performance Considerations

**Acceptance Rate**: Constraints filter samples, so tight constraints require more generation attempts.

```python
# Tight constraint = low acceptance rate
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))
sampler.add_constraint('X + Y < 2')  # Only ~2% of samples pass

# This may take time to generate 1000 samples
samples = sampler.generate(n=1000)  
```

**Tips:**

- Avoid overly restrictive constraints (acceptance rate < 1%)
- Consider if constraint should be a distribution instead
- Use stratified sampling to improve efficiency
- Check feasibility before large sample generation

### Constraint vs Distribution

Sometimes what looks like a constraint is better as a distribution:

```python
# ❌ Inefficient: constraint on unbounded distribution
sampler.add_parameter('X', NormalDistribution(0, 10))
sampler.add_constraint('(X > -5) and (X < 5)')

# ✓ Better: use bounded distribution
from scipy.stats import truncnorm
sampler.add_parameter('X', CustomDistribution(
    lambda n: truncnorm.rvs(-5, 5, loc=0, scale=10, size=n)
))
```

---

## Quality & Validation

Assess and validate your sample quality using built-in metrics.

### Quick Quality Check

```python
samples = sampler.generate(n=1000, return_type='dataframe')

# Compute all metrics
metrics = sampler.compute_quality_metrics(samples)

print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Discrepancy: {metrics['discrepancy']:.4f}")

# Check correlation accuracy
if 'correlation_error' in metrics:
    print(f"Correlation RMSE: {metrics['correlation_error']['rmse']:.4f}")
```

### Available Metrics

#### 1. Coverage

Fraction of parameter space bins that contain samples.

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['coverage'])

print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Bins occupied: {metrics['coverage_details']['occupied_bins']}")
print(f"Total bins: {metrics['coverage_details']['total_bins']}")
```

**Interpretation:**

- **1.0** = Perfect (all bins occupied)
- **0.8+** = Excellent
- **0.5-0.8** = Good
- **<0.5** = Poor (need more samples or stratified sampling)

#### 2. Discrepancy

Star discrepancy measuring uniformity (lower is better).

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['discrepancy'])
print(f"Discrepancy: {metrics['discrepancy']:.4f}")
```

**Interpretation:**

- **<0.1** = Excellent (QMC methods)
- **0.1-0.2** = Good
- **>0.2** = Poor (typical random sampling)

#### 3. Correlation Error

Deviation between achieved and target correlations.

```python
sampler.set_correlation('X', 'Y', 0.8)
samples = sampler.generate(n=1000, return_type='dataframe')

metrics = sampler.compute_quality_metrics(samples, metrics=['correlation_error'])

print(f"RMSE: {metrics['correlation_error']['rmse']:.4f}")
print(f"Max error: {metrics['correlation_error']['max_abs_error']:.4f}")
print(f"Mean error: {metrics['correlation_error']['mean_abs_error']:.4f}")
```

**Interpretation:**

- **<0.05** = Excellent
- **0.05-0.1** = Good
- **0.1-0.2** = Acceptable
- **>0.2** = Poor (need more samples)

#### 4. Distribution Tests

Kolmogorov-Smirnov test for each parameter distribution.

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['distribution_ks'])

for param, result in metrics['distribution_ks'].items():
    status = 'PASS' if result['passes'] else 'FAIL'
    print(f"{param}: {status} (p-value: {result['pvalue']:.4f})")
```

**Interpretation:**

- **p > 0.05**: Distribution matches (PASS)
- **p < 0.05**: Distribution differs (FAIL)

#### 5. Uniformity

Chi-square test for uniform distribution in normalized space.

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['uniformity'])

for param, result in metrics['uniformity'].items():
    status = 'PASS' if result['passes'] else 'FAIL'
    print(f"{param}: {status} (p-value: {result['pvalue']:.4f})")
```

### Validation Workflow

```python
def validate_samples(sampler, samples, min_coverage=0.5, max_discrepancy=0.3):
    """Comprehensive sample validation."""
    metrics = sampler.compute_quality_metrics(
        samples,
        metrics=['coverage', 'discrepancy', 'distribution_ks']
    )
  
    issues = []
  
    # Check coverage
    if metrics['coverage'] < min_coverage:
        issues.append(f"Low coverage: {metrics['coverage']:.2%}")
  
    # Check discrepancy
    if metrics['discrepancy'] > max_discrepancy:
        issues.append(f"High discrepancy: {metrics['discrepancy']:.4f}")
  
    # Check distributions
    for param, result in metrics['distribution_ks'].items():
        if not result['passes']:
            issues.append(f"{param} distribution test failed (p={result['pvalue']:.4f})")
  
    # Report
    if issues:
        print("⚠ Quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All quality checks passed")
        return True

# Use it
samples = sampler.generate(n=1000, return_type='dataframe')
if validate_samples(sampler, samples):
    # Proceed with analysis
    pass
```

### Bootstrap Uncertainty

Estimate uncertainty using bootstrap resampling:

```python
# Original samples
samples = sampler.generate(n=500, return_type='dataframe')

# Bootstrap to estimate uncertainty
n_bootstrap = 1000
bootstrap_means = []

for i in range(n_bootstrap):
    boot_samples = sampler.bootstrap_resample(
        samples, 
        random_seed=42+i,
        return_type='dataframe'
    )
    bootstrap_means.append(boot_samples['X'].mean())

# Compute confidence interval
bootstrap_means = np.array(bootstrap_means)
ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])

print(f"Mean: {samples['X'].mean():.3f}")
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
```

---

## Visualizations

Visualize your samples using built-in plotting functions.

**Note**: Requires matplotlib: `pip install matplotlib`

### Distribution Histograms

```python
import matplotlib.pyplot as plt

samples = sampler.generate(n=1000, return_type='dataframe')

# Plot all parameters
fig = sampler.plot_distributions(samples)
plt.show()

# Plot specific parameters with customization
fig = sampler.plot_distributions(
    samples,
    parameters=['X', 'Y', 'Z'],
    figsize=(15, 5),
    bins=30
)
fig.savefig('distributions.png', dpi=150, bbox_inches='tight')
plt.close()
```

**Shows:** Histogram with mean, std dev, and range statistics.

### Correlation Matrix Heatmap

```python
fig = sampler.plot_correlation_matrix(
    samples,
    figsize=(10, 8),
    annot=True  # Show correlation values on heatmap
)
plt.show()
```

**Shows:** Heatmap of actual correlations between all parameters.

### Pairwise Scatter Plots

```python
fig = sampler.plot_pairwise(
    samples,
    parameters=['X', 'Y', 'Z'],  # Limit to avoid clutter
    figsize=(12, 12),
    alpha=0.6,      # Point transparency
    point_size=20
)
plt.show()
```

**Shows:** Matrix of scatter plots (off-diagonal) and histograms (diagonal).

**Tip**: Limit to 6 parameters max for readability.

### Customization

All plot functions return matplotlib Figure objects:

```python
fig = sampler.plot_distributions(samples)

# Customize
fig.suptitle('Parameter Distributions - Experiment 42', fontsize=16)
fig.tight_layout()

# Save with high quality
fig.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close(fig)
```

### Compare Methods Visually

```python
# Generate samples with different methods
methods = {
    'Random': sampler.generate(n=1000, method='random'),
    'Sobol': sampler.generate(n=1024, method='sobol'),
    'Stratified': sampler.generate_stratified(n=1000, strata_per_param=10)
}

# Plot each
for name, samples in methods.items():
    fig = sampler.plot_distributions(samples)
    fig.suptitle(f'{name} Sampling', fontsize=16)
    fig.savefig(f'{name.lower()}_distributions.png', dpi=150)
    plt.close()
```

---

## Save/Load

Persist configurations and data for reproducibility.

### Save/Load Configuration

Save entire sampler configuration to JSON:

```python
# Configure sampler
sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', NormalDistribution(5, 2))
sampler.set_correlation('X', 'Y', 0.7)
sampler.add_derived_parameter('Z', 'X + Y')
sampler.add_constraint('Z < 12')

# Save configuration
sampler.save_config('my_sampler.json')

# Later, load exact same configuration
sampler2 = Sampler.load_config('my_sampler.json')
samples = sampler2.generate(n=1000)
```

**What gets saved:**

- ✓ Parameters and distributions
- ✓ Correlations
- ✓ Derived parameters (string formulas only)
- ✓ Constraints
- ✓ Random seed

**Limitations:**

- ✗ Callable derived parameters (use string formulas instead)
- ✗ Custom distributions (implement as named class)

### Save/Load Data (CSV)

```python
# Generate and save to CSV
samples = sampler.generate(n=1000, return_type='dataframe')
samples.to_csv('samples.csv', index=False)

# Load later
import pandas as pd
samples = pd.read_csv('samples.csv')
```

### Save/Load Data (HDF5)

For large datasets, HDF5 is more efficient:

```python
# Save to HDF5 (with compression)
sampler.save_samples(
    'samples.h5',
    samples,
    format='hdf5',
    compression='gzip',
    compression_level=6
)

# Load from HDF5
samples = sampler.load_samples('samples.h5', format='hdf5')
```

**Requirements:** `pip install tables` (or `pip install h5py`)

**Benefits:**

- Faster for large datasets
- Compression support
- Preserves data types
- Industry standard format

**Formats:**

```python
# CSV - human readable, universal
sampler.save_samples('data.csv', samples, format='csv')

# HDF5 - efficient, compressed
sampler.save_samples('data.h5', samples, format='hdf5')

# Auto-detect from extension
sampler.save_samples('data.csv', samples)  # Detects CSV
sampler.save_samples('data.h5', samples)   # Detects HDF5
```

---

## Best Practices

### Sample Size Selection

**General Guidelines:**

- **Exploration**: 100-1,000 samples
- **Analysis**: 1,000-10,000 samples
- **Monte Carlo**: 10,000-1,000,000 samples
- **Uncertainty quantification**: 1,000-10,000 per parameter

**Rule of thumb**: At least 100 samples per parameter for basic statistics.

### Method Selection

| Method     | Best For                          | Sample Size             |
| ---------- | --------------------------------- | ----------------------- |
| Random     | Large samples, true randomness    | 10,000+                 |
| Sobol      | Integration, small-medium         | 512-8,192 (powers of 2) |
| Halton     | Flexible size QMC                 | Any, 1,000-10,000       |
| LHS        | Guaranteed per-dimension coverage | Any, 100-5,000          |
| Stratified | Experimental design, guarantees   | Match strata count      |

### Correlation Modeling

**Do:**

- ✓ Base correlations on data or domain knowledge
- ✓ Verify correlation matrix is valid (positive definite)
- ✓ Check achieved correlations match targets
- ✓ Use adequate sample size (>1,000 for accurate correlations)

**Don't:**

- ✗ Use perfect correlation (±1.0) unless truly deterministic
- ✗ Assume correlation implies causation
- ✗ Ignore feasibility (some correlation matrices are impossible)

### Derived Parameters

**Do:**

- ✓ Use string formulas (can be saved)
- ✓ Document formula meanings
- ✓ Validate derived parameter ranges
- ✓ Consider adding constraints on derived parameters

**Don't:**

- ✗ Create circular dependencies
- ✗ Use callables if you need to save config
- ✗ Create overly complex formulas (hard to debug)

### Constraints

**Do:**

- ✓ Check acceptance rate before large generations
- ✓ Consider if constraint should be a distribution instead
- ✓ Use constraints for physical impossibilities
- ✓ Test constraint logic with small samples first

**Don't:**

- ✗ Over-constrain (acceptance rate < 1%)
- ✗ Use constraints for preferences (use stratified instead)
- ✗ Create impossible constraint combinations

### Performance

**For Large Samples:**

- Use HDF5 instead of CSV (faster I/O)
- Generate in batches if memory limited
- Use QMC methods (Sobol/Halton) for faster convergence
- Avoid tight constraints (low acceptance rate)

**For Many Parameters:**

- Limit pairwise plots to 6 parameters
- Use stratified sampling judiciously (strata^n_params grows fast)
- Consider reducing correlations if matrix becomes ill-conditioned

### Reproducibility

**Always:**

```python
# Set random seed for reproducibility
sampler = Sampler(random_seed=42)

# Save configuration
sampler.save_config('config.json')

# Document your workflow
"""
Generated samples for Experiment #42
Method: Sobol QMC
N: 2048
Date: 2026-02-03
Purpose: Parameter sensitivity analysis
"""
```

---

## Examples

### Example 1: Simple Monte Carlo

```python
from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import NormalDistribution

# Setup
sampler = Sampler(random_seed=42)
sampler.add_parameter('demand', NormalDistribution(mean=1000, std=200))
sampler.add_parameter('price', NormalDistribution(mean=50, std=10))
sampler.add_derived_parameter('revenue', 'demand * price')

# Generate
samples = sampler.generate(n=10000, return_type='dataframe')

# Analyze
print(samples['revenue'].describe())
print(f"Revenue 95% CI: [{samples['revenue'].quantile(0.025):.0f}, "
      f"{samples['revenue'].quantile(0.975):.0f}]")
```

### Example 2: Correlated Financial Assets

```python
# Three correlated stock returns
sampler = Sampler(random_seed=42)
sampler.add_parameter('Stock_A', NormalDistribution(0.10, 0.20))  # 10% mean, 20% vol
sampler.add_parameter('Stock_B', NormalDistribution(0.08, 0.15))
sampler.add_parameter('Stock_C', NormalDistribution(0.12, 0.25))

# Set correlations based on historical data
sampler.set_correlation('Stock_A', 'Stock_B', 0.6)
sampler.set_correlation('Stock_B', 'Stock_C', 0.4)
sampler.set_correlation('Stock_A', 'Stock_C', 0.3)

# Portfolio return (equal weights)
sampler.add_derived_parameter('Portfolio', '(Stock_A + Stock_B + Stock_C) / 3')

# Generate scenarios
scenarios = sampler.generate(n=10000, return_type='dataframe')

# Risk metrics
print(f"Portfolio Mean Return: {scenarios['Portfolio'].mean():.2%}")
print(f"Portfolio Volatility: {scenarios['Portfolio'].std():.2%}")
print(f"Value at Risk (95%): {scenarios['Portfolio'].quantile(0.05):.2%}")
```

### Example 3: Experimental Design

```python
# Design experiments for chemical process
sampler = Sampler(random_seed=42)
sampler.add_parameter('temperature', UniformDistribution(20, 100))  # °C
sampler.add_parameter('pressure', UniformDistribution(1, 5))  # bar
sampler.add_parameter('catalyst', UniformDistribution(0, 2))  # g/L

# Use stratified sampling with center points for reproducibility
experiments = sampler.generate_stratified(
    n=27,  # 3^3 factorial-like design
    strata_per_param=3,
    method='center',
    return_type='dataframe'
)

# Add experiment IDs
experiments.insert(0, 'exp_id', range(1, len(experiments)+1))

# Save for lab use
experiments.to_csv('experiment_plan.csv', index=False)
print(f"Created {len(experiments)} experiments")

# Check coverage
metrics = sampler.compute_quality_metrics(experiments, metrics=['coverage'])
print(f"Parameter space coverage: {metrics['coverage']:.1%}")
```

### Example 4: Quality Comparison

```python
# Compare sampling methods
sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 1))
sampler.add_parameter('Y', UniformDistribution(0, 1))
sampler.set_correlation('X', 'Y', 0.5)

methods = {
    'Random': {'method': 'random', 'n': 1000},
    'Sobol': {'method': 'sobol', 'n': 1024},
    'Halton': {'method': 'halton', 'n': 1000},
    'LHS': {'method': 'lhs', 'n': 1000},
}

print(f"{'Method':<12} {'Coverage':<12} {'Discrepancy':<12} {'Corr Error':<12}")
print("-" * 50)

for name, params in methods.items():
    samples = sampler.generate(**params)
    metrics = sampler.compute_quality_metrics(
        samples,
        metrics=['coverage', 'discrepancy', 'correlation_error']
    )
  
    print(f"{name:<12} {metrics['coverage']:<12.3f} "
          f"{metrics['discrepancy']:<12.4f} "
          f"{metrics['correlation_error']['rmse']:<12.4f}")

# Stratified
samples = sampler.generate_stratified(n=1000, strata_per_param=10)
metrics = sampler.compute_quality_metrics(
    samples,
    metrics=['coverage', 'discrepancy', 'correlation_error']
)
print(f"{'Stratified':<12} {metrics['coverage']:<12.3f} "
      f"{metrics['discrepancy']:<12.4f} "
      f"{metrics['correlation_error']['rmse']:<12.4f}")
```

### More Examples

See the `examples/` directory for complete working examples:

- `sampling_example.py` - Basic usage
- `correlated_sampling_examples.py` - Correlation features
- `advanced_features_example.py` - Constraints, derived parameters, save/load
- `quality_and_visualization_example.py` - Quality metrics and plots
- `qmc_sampling_example.py` - Quasi-Monte Carlo methods
- `custom_distributions_example.py` - Custom distributions

---

## Next Steps

- **Try it**: Run `examples/quality_and_visualization_example.py`
- **API Reference**: See [SAMPLING_REFERENCE.md](SAMPLING_REFERENCE.md) for complete method documentation
- **Advanced Topics**: Explore custom distributions, complex workflows
- **Get Help**: Open an issue on GitHub

---

## Summary

| Feature                       | Use Case                   | Key Method                            |
| ----------------------------- | -------------------------- | ------------------------------------- |
| **Basic Sampling**      | Quick generation           | `generate()`                        |
| **QMC Sampling**        | Efficient space-filling    | `generate(method='sobol')`          |
| **Stratified Sampling** | Experimental design        | `generate_stratified()`             |
| **Correlations**        | Parameter dependencies     | `set_correlation()`                 |
| **Derived Parameters**  | Computed values            | `add_derived_parameter()`           |
| **Constraints**         | Filter samples             | `add_constraint()`                  |
| **Quality Metrics**     | Validate samples           | `compute_quality_metrics()`         |
| **Visualizations**      | Understand data            | `plot_distributions()`              |
| **Bootstrap**           | Uncertainty quantification | `bootstrap_resample()`              |
| **Save/Load**           | Reproducibility            | `save_config()`, `save_samples()` |

**All features work together seamlessly** - use correlations with stratified sampling, add constraints to derived parameters, visualize correlated samples, etc.
