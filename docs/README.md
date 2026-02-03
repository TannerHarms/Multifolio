# Sampling Documentation

Comprehensive documentation for Multifolio's parameter sampling capabilities.

## Start Here

**New to Multifolio sampling?**
→ Start with [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) - Complete guide from basics to advanced features

**Looking for specific API details?**
→ See [SAMPLING_REFERENCE.md](SAMPLING_REFERENCE.md) - Full API reference

## Documentation Structure

### Main Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [**SAMPLING_GUIDE.md**](SAMPLING_GUIDE.md) | Complete user guide covering all features | All users |
| [**SAMPLING_REFERENCE.md**](SAMPLING_REFERENCE.md) | API reference with method signatures | Developers |

**Contents of SAMPLING_GUIDE.md:**
- Getting Started (5-minute quickstart)
- Core Concepts (sampler workflow)
- Distribution Types (all available distributions)
- Sampling Methods (random, QMC, stratified, LHS)
- Correlations (parameter dependencies)
- Derived Parameters (computed parameters)
- Constraints (filter samples)
- Quality & Validation (assess sample quality)
- Visualizations (plot and analyze)
- Save/Load (persistence)
- Best Practices
- Complete Examples

### Legacy Documentation (Deprecated)

The following files contain older, overlapping documentation and are **deprecated**:

- ~~CORRELATION_README.md~~ → Now in SAMPLING_GUIDE.md (Correlations section)
- ~~CORRELATION_QUICKSTART.md~~ → Now in SAMPLING_GUIDE.md (Correlations section)
- ~~CORRELATION_DOCUMENTATION.md~~ → Now in SAMPLING_GUIDE.md (Correlations section)
- ~~CORRELATION_IMPLEMENTATION_SUMMARY.md~~ → Implementation details
- ~~ADVANCED_FEATURES.md~~ → Now in SAMPLING_GUIDE.md (various sections)
- ~~QUALITY_AND_VISUALIZATION.md~~ → Now in SAMPLING_GUIDE.md (Quality & Visualizations sections)

**These files remain for historical reference but should not be used for new development.**

## Quick Reference

### Common Tasks

**Generate basic samples:**
```python
from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import UniformDistribution

sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 10))
samples = sampler.generate(n=1000, return_type='dataframe')
```

**Add correlations:**
```python
sampler.set_correlation('X', 'Y', 0.7)
```

**Stratified sampling:**
```python
samples = sampler.generate_stratified(n=125, strata_per_param=5)
```

**Quality check:**
```python
metrics = sampler.compute_quality_metrics(samples)
print(f"Coverage: {metrics['coverage']:.2%}")
```

**Visualize:**
```python
fig = sampler.plot_distributions(samples)
```

**Save/load:**
```python
sampler.save_config('config.json')
sampler.save_samples('data.h5', samples, format='hdf5')
```

→ See [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) for complete details

## Feature Overview

### Sampling Methods

- **Random (Monte Carlo)**: Standard pseudo-random
- **Sobol (QMC)**: Low-discrepancy sequence
- **Halton (QMC)**: Another low-discrepancy sequence
- **Latin Hypercube (LHS)**: Stratified per dimension
- **Stratified**: Guaranteed coverage with strata

### Advanced Features

- **Correlations**: Model parameter dependencies (Gaussian copula)
- **Derived Parameters**: Computed from other parameters
- **Constraints**: Filter samples by conditions
- **Quality Metrics**: Coverage, discrepancy, correlation error, KS tests, uniformity
- **Bootstrap**: Resample with replacement for uncertainty quantification
- **Visualizations**: Distributions, correlation matrix, pairwise plots
- **Save/Load**: JSON config, CSV/HDF5 data

### Distribution Types

Uniform, Normal, LogNormal, Exponential, Beta, Gamma, Weibull, Triangular, Custom

## Examples

Working examples in [`../examples/`](../examples/):

| Example File | Features Demonstrated |
|--------------|----------------------|
| `sampling_example.py` | Basic usage, distributions, methods |
| `correlated_sampling_examples.py` | Correlations, multivariate |
| `advanced_features_example.py` | Constraints, derived parameters, save/load |
| `quality_and_visualization_example.py` | Quality metrics, stratified, bootstrap, plots |
| `qmc_sampling_example.py` | Quasi-Monte Carlo methods |
| `custom_distributions_example.py` | Custom distributions |

## Migration Guide

If you're using old documentation:

### From CORRELATION_README.md / CORRELATION_QUICKSTART.md

→ See [SAMPLING_GUIDE.md - Correlations](SAMPLING_GUIDE.md#correlations)

### From CORRELATION_DOCUMENTATION.md

→ See [SAMPLING_GUIDE.md - Correlations](SAMPLING_GUIDE.md#correlations) for usage
→ See CORRELATION_IMPLEMENTATION_SUMMARY.md for technical implementation details (if needed)

### From ADVANCED_FEATURES.md

- Save/load config → [SAMPLING_GUIDE.md - Save/Load](SAMPLING_GUIDE.md#saveload)
- Constraints → [SAMPLING_GUIDE.md - Constraints](SAMPLING_GUIDE.md#constraints)
- Derived parameters → [SAMPLING_GUIDE.md - Derived Parameters](SAMPLING_GUIDE.md#derived-parameters)

### From QUALITY_AND_VISUALIZATION.md

- Stratified sampling → [SAMPLING_GUIDE.md - Sampling Methods](SAMPLING_GUIDE.md#stratified-sampling)
- Bootstrap → [SAMPLING_GUIDE.md - Quality & Validation](SAMPLING_GUIDE.md#bootstrap-uncertainty)
- Quality metrics → [SAMPLING_GUIDE.md - Quality & Validation](SAMPLING_GUIDE.md#quality--validation)
- Visualizations → [SAMPLING_GUIDE.md - Visualizations](SAMPLING_GUIDE.md#visualizations)

## Contributing

When adding new features:

1. Update [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) with usage guide
2. Update [SAMPLING_REFERENCE.md](SAMPLING_REFERENCE.md) with API details
3. Add working example to `../examples/`
4. Add tests to `../backend/tests/unit/sampling/`
5. Update this README if needed

## Support

- **Documentation issues**: Open an issue on GitHub
- **Examples**: See `../examples/` directory
- **Tests**: See `../backend/tests/unit/sampling/` for usage patterns
