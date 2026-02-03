# Quality Metrics, Stratified Sampling & Visualizations

> **⚠️ DEPRECATED**: This documentation has been superseded by the new unified documentation.
> 
> **Please use:** [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) for complete feature coverage.
> 
> This file remains for historical reference only.

---

This guide covers advanced sampling quality features including stratified sampling, bootstrap resampling, quality metrics, and basic visualizations.

## Table of Contents
- [Stratified Sampling](#stratified-sampling)
- [Bootstrap Resampling](#bootstrap-resampling)
- [Sample Quality Metrics](#sample-quality-metrics)
- [Visualizations](#visualizations)
- [Use Cases](#use-cases)

---

## Stratified Sampling

Stratified sampling divides the parameter space into bins (strata) and samples from each stratum, ensuring better coverage of the entire parameter space. This is particularly useful for:
- Experimental design
- Rare event sampling
- Ensuring representative coverage
- Reducing variance in estimates

### Basic Usage

```python
from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import UniformDistribution

sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))

# Divide each parameter into 5 strata
samples = sampler.generate_stratified(
    n=100,
    strata_per_param=5,
    return_type='dataframe'
)
```

### Stratification Options

**Uniform Strata:**
```python
# Same number of strata for all parameters
samples = sampler.generate_stratified(n=100, strata_per_param=5)
```

**Parameter-Specific Strata:**
```python
# Different strata for different parameters
samples = sampler.generate_stratified(
    n=100,
    strata_per_param={'X': 3, 'Y': 5, 'Z': 4}
)
```

### Sampling Methods

Stratified sampling supports three methods for sampling within each stratum:

**1. Random (default):**
Uniform random sampling within each stratum.
```python
samples = sampler.generate_stratified(n=100, method='random')
```

**2. Center:**
Sample at the center of each stratum (deterministic, reproducible).
```python
# Excellent for experimental design
samples = sampler.generate_stratified(n=100, method='center')
```

**3. Jittered:**
Center with small random perturbation (±10% of stratum width).
```python
samples = sampler.generate_stratified(n=100, method='jittered')
```

### With Correlations

Stratified sampling fully supports correlation structures:

```python
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))
sampler.set_correlation('X', 'Y', 0.7)

# Correlations are preserved in stratified samples
samples = sampler.generate_stratified(n=500, strata_per_param=5)
```

### With Derived Parameters

Derived parameters are automatically computed:

```python
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))
sampler.add_derived_parameter('Z', 'X + Y')

samples = sampler.generate_stratified(n=100, strata_per_param=5)
# samples contains X, Y, and Z
```

---

## Bootstrap Resampling

Bootstrap resampling generates new samples by randomly selecting from existing data with replacement. Useful for:
- Uncertainty quantification
- Estimating sampling distributions
- Computing confidence intervals
- Assessing variability

### Basic Usage

```python
# Generate original samples
samples = sampler.generate(n=1000, return_type='dataframe')

# Bootstrap resample (same size)
boot_samples = sampler.bootstrap_resample(samples)

# Bootstrap with different size
boot_samples = sampler.bootstrap_resample(samples, n=5000)
```

### Confidence Intervals

```python
# Compute 95% bootstrap confidence intervals
n_bootstrap = 1000
bootstrap_means = []

for i in range(n_bootstrap):
    boot = sampler.bootstrap_resample(samples, random_seed=42+i)
    bootstrap_means.append(boot['X'].mean())

bootstrap_means = np.array(bootstrap_means)

# Confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### With Return Types

```python
# Dict format (default)
boot_dict = sampler.bootstrap_resample(samples, return_type='dict')

# DataFrame format
boot_df = sampler.bootstrap_resample(samples, return_type='dataframe')
```

---

## Sample Quality Metrics

Quality metrics evaluate how well your samples cover the parameter space and match target distributions.

### Available Metrics

1. **Coverage**: Fraction of parameter space strata occupied
2. **Discrepancy**: Star discrepancy (uniformity measure, lower is better)
3. **Correlation Error**: Deviation from target correlation structure
4. **Distribution KS**: Kolmogorov-Smirnov test for each parameter
5. **Uniformity**: Chi-square test for uniform distribution

### Basic Usage

```python
samples = sampler.generate(n=1000, return_type='dataframe')

# Compute all metrics
metrics = sampler.compute_quality_metrics(samples)

print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Discrepancy: {metrics['discrepancy']:.4f}")
```

### Specific Metrics

```python
# Compute only specific metrics
metrics = sampler.compute_quality_metrics(
    samples,
    metrics=['coverage', 'discrepancy']
)
```

### Coverage Metric

Measures what fraction of the parameter space contains samples:

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['coverage'])

print(f"Coverage: {metrics['coverage']:.2%}")
print(f"Occupied bins: {metrics['coverage_details']['occupied_bins']}")
print(f"Total bins: {metrics['coverage_details']['total_bins']}")
print(f"Bins per dimension: {metrics['coverage_details']['bins_per_dimension']}")
```

**Interpretation:**
- 1.0 = Perfect coverage (all bins occupied)
- 0.8+ = Excellent coverage
- 0.5-0.8 = Good coverage
- <0.5 = Poor coverage (consider more samples or stratified sampling)

### Discrepancy Metric

Measures uniformity of samples (star discrepancy):

```python
metrics = sampler.compute_quality_metrics(samples, metrics=['discrepancy'])

print(f"Discrepancy: {metrics['discrepancy']:.4f}")
```

**Interpretation:**
- 0.0 = Perfect uniformity (theoretical minimum)
- <0.1 = Excellent (QMC methods)
- 0.1-0.2 = Good
- >0.2 = Poor (random sampling typically produces this)

### Correlation Error

Measures deviation from target correlations:

```python
sampler.set_correlation('X', 'Y', 0.8)
samples = sampler.generate(n=1000, return_type='dataframe')

metrics = sampler.compute_quality_metrics(samples, metrics=['correlation_error'])

print(f"RMSE: {metrics['correlation_error']['rmse']:.4f}")
print(f"Max error: {metrics['correlation_error']['max_abs_error']:.4f}")
print(f"Mean error: {metrics['correlation_error']['mean_abs_error']:.4f}")
```

**Interpretation:**
- <0.05 = Excellent
- 0.05-0.1 = Good  
- 0.1-0.2 = Acceptable
- >0.2 = Poor (need more samples)

### Distribution Tests

KS test and uniformity test for each parameter:

```python
metrics = sampler.compute_quality_metrics(
    samples,
    metrics=['distribution_ks', 'uniformity']
)

# KS test results
for param, result in metrics['distribution_ks'].items():
    print(f"{param}: {'PASS' if result['passes'] else 'FAIL'} "
          f"(p={result['pvalue']:.4f})")

# Uniformity test results
for param, result in metrics['uniformity'].items():
    print(f"{param}: {'PASS' if result['passes'] else 'FAIL'} "
          f"(p={result['pvalue']:.4f})")
```

### Comparing Sampling Methods

```python
# Compare quality of different methods
random_samples = sampler.generate(n=1000, method='random')
sobol_samples = sampler.generate(n=1000, method='sobol')
stratified_samples = sampler.generate_stratified(n=1000, strata_per_param=10)

for name, samples in [('Random', random_samples), 
                       ('Sobol', sobol_samples),
                       ('Stratified', stratified_samples)]:
    metrics = sampler.compute_quality_metrics(
        samples, metrics=['coverage', 'discrepancy']
    )
    print(f"{name}:")
    print(f"  Coverage: {metrics['coverage']:.3f}")
    print(f"  Discrepancy: {metrics['discrepancy']:.4f}")
```

---

## Visualizations

Basic plotting functions for understanding sample distributions and relationships.

**Note**: Requires matplotlib: `pip install matplotlib`

### Distribution Plots

Plot histograms of parameter distributions:

```python
samples = sampler.generate(n=1000, return_type='dataframe')

# Plot all parameters
fig = sampler.plot_distributions(samples)

# Plot specific parameters
fig = sampler.plot_distributions(
    samples,
    parameters=['X', 'Y'],
    figsize=(12, 5),
    bins=50
)

fig.savefig('distributions.png', dpi=150)
```

### Correlation Matrix Heatmap

Visualize correlation structure:

```python
fig = sampler.plot_correlation_matrix(
    samples,
    figsize=(10, 8),
    annot=True  # Show correlation values
)

fig.savefig('correlation_matrix.png')
```

### Pairwise Scatter Plots

Matrix of scatter plots showing relationships:

```python
fig = sampler.plot_pairwise(
    samples,
    parameters=['X', 'Y', 'Z'],  # Limit to specific params (max 6 recommended)
    figsize=(12, 12),
    alpha=0.5,  # Point transparency
    point_size=10
)

fig.savefig('pairwise_plots.png')
```

### Customization

All plotting functions return matplotlib Figure objects for further customization:

```python
import matplotlib.pyplot as plt

fig = sampler.plot_distributions(samples)

# Customize
fig.suptitle('My Custom Title', fontsize=16)
plt.tight_layout()

# Save with options
fig.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Use Cases

### 1. Experimental Design

Use stratified 'center' method for reproducible, well-spaced experiments:

```python
sampler = Sampler()
sampler.add_parameter('temperature', UniformDistribution(20, 100))
sampler.add_parameter('pressure', UniformDistribution(0.5, 2.0))
sampler.add_parameter('catalyst', UniformDistribution(0, 1))

# Generate 27 experiments (3^3 factorial-like design)
experiments = sampler.generate_stratified(
    n=27,
    strata_per_param=3,  # Low/Medium/High
    method='center',
    return_type='dataframe'
)

# Add IDs and save
experiments.insert(0, 'exp_id', range(1, len(experiments)+1))
experiments.to_csv('experiments.csv', index=False)

# Check quality
metrics = sampler.compute_quality_metrics(experiments, metrics=['coverage'])
print(f"Design covers {metrics['coverage']:.1%} of parameter space")
```

### 2. Uncertainty Quantification

Use bootstrap to estimate parameter uncertainty:

```python
# Original analysis
samples = sampler.generate(n=500, return_type='dataframe')
original_result = some_analysis_function(samples)

# Bootstrap uncertainty
n_bootstrap = 1000
bootstrap_results = []

for i in range(n_bootstrap):
    boot_samples = sampler.bootstrap_resample(
        samples, random_seed=42+i, return_type='dataframe'
    )
    bootstrap_results.append(some_analysis_function(boot_samples))

# Compute confidence interval
bootstrap_results = np.array(bootstrap_results)
ci_95 = np.percentile(bootstrap_results, [2.5, 97.5])

print(f"Result: {original_result:.3f}")
print(f"95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
```

### 3. Sampling Method Selection

Compare methods to choose the best for your application:

```python
methods = {
    'Random': lambda: sampler.generate(n=1000, method='random'),
    'Sobol QMC': lambda: sampler.generate(n=1000, method='sobol'),
    'Stratified (5)': lambda: sampler.generate_stratified(n=1000, strata_per_param=5),
    'Stratified (10)': lambda: sampler.generate_stratified(n=1000, strata_per_param=10),
}

results = {}
for name, generator in methods.items():
    samples = generator()
    metrics = sampler.compute_quality_metrics(
        samples, metrics=['coverage', 'discrepancy']
    )
    results[name] = metrics

# Print comparison
print("Method Comparison:")
print(f"{'Method':<20} {'Coverage':<12} {'Discrepancy':<12}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['coverage']:<12.3f} "
          f"{metrics['discrepancy']:<12.4f}")
```

### 4. Quality Assurance

Validate sample quality before using in downstream analysis:

```python
def validate_samples(samples, min_coverage=0.5, max_discrepancy=0.3):
    """Validate sample quality."""
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
            issues.append(f"{param} distribution KS test failed")
    
    if issues:
        print("⚠ Quality issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Sample quality validated")
        return True

# Use validation
samples = sampler.generate_stratified(n=1000, strata_per_param=10)
if validate_samples(samples):
    # Proceed with analysis
    pass
```

### 5. Sensitivity Analysis

Use stratified sampling to ensure adequate coverage for sensitivity analysis:

```python
# Ensure coverage across all combinations
n_strata = 5
samples = sampler.generate_stratified(
    n=n_strata**3,  # One sample per stratum combination
    strata_per_param=n_strata,
    method='center',
    return_type='dataframe'
)

# Compute outputs
samples['output'] = your_model_function(samples)

# Analyze sensitivity across strata
for param in ['X', 'Y', 'Z']:
    # Bin parameter into strata
    samples[f'{param}_stratum'] = pd.cut(samples[param], bins=n_strata, labels=range(n_strata))
    
    # Compute mean output per stratum
    sensitivity = samples.groupby(f'{param}_stratum')['output'].mean()
    print(f"\n{param} sensitivity:")
    print(sensitivity)
```

---

## Best Practices

### When to Use Stratified Sampling

**Use stratified sampling when:**
- ✓ You need guaranteed coverage of parameter space
- ✓ Performing experimental design
- ✓ Working with limited sample budget
- ✓ Analyzing rare events or tails of distributions
- ✓ Need reproducible, evenly-spaced samples (use method='center')

**Use regular sampling when:**
- Large sample sizes (>10,000)
- Simple random sampling sufficient
- Want fully independent samples

### When to Use Bootstrap

**Use bootstrap when:**
- ✓ Estimating uncertainty of derived statistics
- ✓ Computing confidence intervals
- ✓ Assessing variability without analytical formulas
- ✓ Working with empirical data

**Don't use bootstrap for:**
- Generating new independent samples (use generate() instead)
- When analytical uncertainty is available
- Very small original sample sizes (<30)

### Quality Metrics Best Practices

1. **Always check quality** for small samples (<1000)
2. **Compare methods** before committing to one
3. **Stratified sampling** typically shows best coverage
4. **QMC methods** (Sobol, Halton) show best discrepancy
5. **More strata** improves coverage but requires more samples
6. **Correlation preservation** may slightly reduce coverage

### Visualization Tips

1. **Limit parameters** in pairwise plots (max 6)
2. **Adjust alpha** for overlapping points
3. **Use bins appropriately** (fewer bins for smaller samples)
4. **Save at high DPI** (150-300) for presentations
5. **Close figures** to free memory: `plt.close('all')`

---

## Performance Considerations

### Stratified Sampling

- **Time**: Similar to regular sampling
- **Memory**: Same as regular sampling
- **Best with**: 2-10 strata per parameter
- **Total strata**: strata_per_param^n_parameters

Example: 3 parameters with 5 strata each = 5³ = 125 total strata

### Bootstrap Resampling

- **Time**: Very fast (just indexing)
- **Memory**: Creates new copy of data
- **Typical iterations**: 100-1000 for confidence intervals

### Quality Metrics

- **Time**: Fast for <10,000 samples
- **Memory**: Moderate (creates temporary arrays)
- **Discrepancy**: Uses Monte Carlo approximation (1000 test points)

### Visualizations

- **Time**: Moderate (matplotlib rendering)
- **Memory**: Creates figure in memory
- **Tip**: Close figures after saving: `plt.close(fig)`

---

## Summary

| Feature | Use Case | Key Benefit |
|---------|----------|-------------|
| **Stratified Sampling** | Experimental design, rare events | Better coverage |
| **Bootstrap** | Uncertainty quantification | Confidence intervals |
| **Quality Metrics** | Validation, method comparison | Objective assessment |
| **Visualizations** | Understanding, presentations | Visual insight |

All features work seamlessly with:
- ✓ Correlations
- ✓ Derived parameters
- ✓ Constraints
- ✓ All distribution types
- ✓ Multiple return formats

---

## Next Steps

- Try the example: `examples/quality_and_visualization_example.py`
- Read about [Advanced Features](ADVANCED_FEATURES.md)
- Learn about [Correlation Modeling](CORRELATION_DOCUMENTATION.md)
- Explore [Derived Parameters](../examples/derived_parameters_example.py)
