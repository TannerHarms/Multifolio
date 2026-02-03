# Advanced Sampling Features

This document covers advanced features for the Multifolio sampling module, including configuration persistence, sample filtering, parameter constraints, conditional generation, and batch processing.

## Table of Contents

- [Save/Load Configuration](#saveload-configuration)
- [Save/Load Data](#saveload-data)
- [Sample Filtering](#sample-filtering)
- [Parameter Constraints](#parameter-constraints)
- [Conditional Generation](#conditional-generation)
- [Batch Generation](#batch-generation)
- [Feature Comparison](#feature-comparison)
- [Best Practices](#best-practices)

---

## Save/Load Configuration

Persist your sampler configuration to JSON for reproducibility and sharing.

### Configuration Persistence

```python
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import UniformDistribution, NormalDistribution

# Create and configure sampler
sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', NormalDistribution(5, 2))
sampler.set_correlation('X', 'Y', 0.8)
sampler.add_derived_parameter('Z', 'X + Y')
sampler.add_constraint('Z < 12')

# Save configuration
sampler.save_config("my_sampler.json")

# Later, reload the exact same configuration
sampler2 = Sampler.load_config("my_sampler.json")
samples = sampler2.generate(n=1000)
```

### What Gets Saved

- **Parameters**: All distribution types and their parameters
- **Correlations**: Full correlation matrix
- **Derived Parameters**: String formulas (callables cannot be serialized)
- **Constraints**: All constraint expressions
- **Random Seed**: For reproducibility

### Limitations

- **Custom Distributions**: Can't be fully serialized, must be recreated manually
- **Callable Formulas**: Not serialized, must be re-added after loading

---

## Save/Load Data

Save generated samples for later analysis or sharing.

### Supported Formats

```python
# Generate samples
samples = sampler.generate(n=10000, return_type='dataframe')

# CSV - human-readable, widely compatible
Sampler.save_data(samples, "samples.csv")
loaded = Sampler.load_data("samples.csv")

# Pickle - fast, preserves dtypes, Python-specific
Sampler.save_data(samples, "samples.pkl", format='pickle')
loaded = Sampler.load_data("samples.pkl")

# Parquet - efficient columnar format, good for large datasets
Sampler.save_data(samples, "samples.parquet", format='parquet')
loaded = Sampler.load_data("samples.parquet")

# HDF5 - very fast, efficient, supports compression, great for scientific data
Sampler.save_data(samples, "samples.h5", format='hdf5')
loaded = Sampler.load_data("samples.h5")

# HDF5 with compression (default is gzip)
Sampler.save_data(samples, "samples.h5", compression='gzip')

# Auto-detect format from extension
Sampler.save_data(samples, "data.csv")  # Automatically uses CSV
Sampler.save_data(samples, "data.h5")   # Automatically uses HDF5
```

### Format Comparison

| Format  | Speed | Size | Compression | Compatibility | Use Case |
|---------|-------|------|-------------|---------------|----------|
| CSV     | Slow  | Large| No          | Universal     | Sharing, human-readable |
| Pickle  | Fast  | Small| No          | Python only   | Internal Python workflows |
| Parquet | Fast  | Small| Yes         | Many tools    | Large datasets, analytics |
| HDF5    | Very Fast | Small | Yes     | Scientific tools | Large scientific datasets |

**Note**: HDF5 support requires either `pytables` or `h5py`. The sampler will try PyTables first (better DataFrame support), then fall back to h5py (more common):
```bash
pip install tables  # Recommended
# or
pip install h5py    # Alternative
```

---

## Sample Filtering

Post-generation filtering of samples based on conditions.

### Basic Filtering

```python
# Generate samples
samples = sampler.generate(n=10000, return_type='dataframe')

# Filter by simple condition
filtered = sampler.filter_samples(samples, 'X > 5')

# Multiple conditions (use & and | for element-wise operations)
filtered = sampler.filter_samples(samples, '(X > 3) & (Y < 7)')

# Complex mathematical conditions
filtered = sampler.filter_samples(samples, 'X**2 + Y**2 < 25')

# Using numpy functions
filtered = sampler.filter_samples(samples, 'np.abs(X) < 2')
```

### Works with Dict Input

```python
# Works with dict format too
samples_dict = sampler.generate(n=1000, return_type='dict')
filtered_dict = sampler.filter_samples(samples_dict, 'X + Y > 10')
```

### When to Use Filtering

**Use filtering when:**
- You've already generated samples and want to post-process them
- Exploring different filtering criteria interactively
- The filtering condition is simple and has high acceptance rate

**Don't use filtering when:**
- You know the constraint ahead of time (use `add_constraint()` instead)
- The condition rejects most samples (use `generate_conditional()` instead)
- You need the filtering applied consistently (use `add_constraint()`)

---

## Parameter Constraints

Pre-generation constraints ensure all samples satisfy specified conditions.

### How Constraints Work

Constraints use **rejection sampling**:
1. Generate a batch of samples
2. Evaluate all constraint expressions
3. Keep only samples satisfying ALL constraints
4. Repeat until N valid samples obtained

```python
sampler = Sampler()
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))

# Add constraints
sampler.add_constraint('X + Y <= 15')  # Budget constraint
sampler.add_constraint('X >= 0.5 * Y')  # Ratio constraint

# All generated samples will satisfy constraints
samples = sampler.generate(n=1000, return_type='dataframe')
assert all(samples['X'] + samples['Y'] <= 15)
assert all(samples['X'] >= 0.5 * samples['Y'])
```

### Constraint Examples

```python
# Budget constraints
sampler.add_constraint('cost1 + cost2 + cost3 <= budget')

# Ratio constraints
sampler.add_constraint('stock_allocation / total_portfolio >= 0.1')
sampler.add_constraint('stock_allocation / total_portfolio <= 0.3')

# Complex regions (e.g., inside a circle)
sampler.add_constraint('X**2 + Y**2 <= radius**2')

# Conditional logic (using numpy operations)
sampler.add_constraint('(X > 0) & (Y > 0)')  # Both positive
```

### Managing Constraints

```python
# View current constraints
constraints = sampler.get_constraints()

# Clear all constraints
sampler.clear_constraints()

# Method chaining
sampler.add_constraint('X > 0').add_constraint('Y > 0')
```

### Constraint Performance

**Efficiency depends on acceptance rate:**

| Acceptance Rate | Performance | Recommendation |
|-----------------|-------------|----------------|
| > 50% | Excellent | Safe to use |
| 10-50% | Good | Consider tighter bounds |
| 1-10% | Slow | Tighten parameter bounds first |
| < 1% | Very slow | Redesign constraints or use conditional generation |

**Tips for efficient constraints:**
1. Set tight parameter bounds first
2. Make constraints as broad as possible
3. Monitor acceptance rate (ratio of valid to total samples)
4. Consider using `max_constraint_attempts` parameter

```python
# Can disable constraints temporarily
samples_all = sampler.generate(n=100, apply_constraints=False)
samples_constrained = sampler.generate(n=100, apply_constraints=True)
```

### Constraints with Derived Parameters

Constraints can reference derived parameters:

```python
sampler.add_parameter('X', UniformDistribution(1, 10))
sampler.add_parameter('Y', UniformDistribution(1, 10))
sampler.add_derived_parameter('Product', 'X * Y')

# Constraint on derived parameter
sampler.add_constraint('Product < 50')

samples = sampler.generate(n=100)
# Product is automatically computed and constrained
```

---

## Conditional Generation

One-time generation of samples satisfying a condition, without persistently adding constraints.

### Basic Usage

```python
# Generate samples inside unit circle
samples = sampler.generate_conditional(
    n=1000,
    condition='X**2 + Y**2 < 1',
    return_type='dataframe'
)
```

### How It Works

1. Generates samples in adaptive batches
2. Filters by condition
3. Collects valid samples until N obtained
4. Adjusts batch size based on observed acceptance rate
5. Raises error if unable to generate N samples within `max_attempts`

### When to Use Conditional Generation

**Use `generate_conditional()` when:**
- You need samples satisfying a condition just once
- The condition is complex or exploratory
- You don't want the condition applied to all future generations

**Use `add_constraint()` when:**
- The constraint should apply to all future generations
- You're generating multiple batches with the same constraint
- You want the constraint documented in saved configurations

### Advanced Options

```python
samples = sampler.generate_conditional(
    n=500,
    condition='X + Y < 10',
    max_attempts=10000,  # Limit total samples generated
    return_type='dict',
    method='sobol'  # Can use QMC methods
)
```

### Error Handling

```python
try:
    samples = sampler.generate_conditional(
        n=1000,
        condition='X > 100',  # Impossible for Uniform(0, 10)
        max_attempts=5000
    )
except RuntimeError as e:
    print(f"Could not satisfy condition: {e}")
    # Consider: relaxing condition, adjusting parameter bounds,
    # or increasing max_attempts
```

---

## Batch Generation

Generate samples in batches for memory-efficient processing.

### Basic Batch Generation

```python
# Generate in batches of 1000
for batch in sampler.generate_batches(n_per_batch=1000, n_batches=10):
    # Process batch (e.g., run simulation, save to database)
    results = expensive_computation(batch)
    save_results(results)
```

### With Batch Tracking

```python
for batch in sampler.generate_batches(
    n_per_batch=1000,
    n_batches=20,
    return_type='dataframe',
    track_batch_id=True  # Adds 'batch_id' column
):
    print(f"Processing batch {batch['batch_id'].iloc[0]}")
    process(batch)
```

### Use Cases

**Memory-Efficient Processing:**
```python
# Process 1 million samples without loading all in memory
total_samples = 0
for batch in sampler.generate_batches(n_per_batch=10000, n_batches=100):
    total_samples += len(batch)
    process_and_discard(batch)
```

**Progressive Computation:**
```python
# Run simulations progressively
results = []
for batch in sampler.generate_batches(n_per_batch=500, n_batches=20):
    batch_results = monte_carlo_simulation(batch)
    results.append(batch_results)
    
    # Can check convergence and stop early
    if has_converged(results):
        break
```

**Parallel Processing:**
```python
from multiprocessing import Pool

# Generate all batches
batches = list(sampler.generate_batches(n_per_batch=1000, n_batches=10))

# Process in parallel
with Pool() as pool:
    results = pool.map(expensive_function, batches)
```

### QMC Methods with Batches

```python
# Consecutive batches continue the low-discrepancy sequence
for batch in sampler.generate_batches(
    n_per_batch=256,  # Power of 2 for Sobol
    n_batches=4,
    method='sobol'
):
    # Batches maintain good space-filling properties
    analyze(batch)
```

---

## Feature Comparison

### When to Use Each Feature

| Feature | Best For | Timing | Persistence |
|---------|----------|--------|-------------|
| **filter_samples()** | Post-processing, exploration | After generation | One-time |
| **add_constraint()** | Consistent enforcement | During generation | Persistent |
| **generate_conditional()** | One-off conditions | During generation | One-time |
| **generate_batches()** | Large datasets, streaming | During generation | Flexible |

### Efficiency Comparison

```python
# Example: 1000 samples where X + Y < 10, acceptance ~25%

# Method 1: Filter (generates 1000, keeps ~250)
samples = sampler.generate(n=1000)
filtered = sampler.filter_samples(samples, 'X + Y < 10')  
# Result: ~250 samples, wasteful

# Method 2: Conditional generation (generates ~4000, keeps 1000)
samples = sampler.generate_conditional(n=1000, condition='X + Y < 10')
# Result: exactly 1000 samples, efficient

# Method 3: Add constraint (generates ~4000, keeps 1000)
sampler.add_constraint('X + Y < 10')
samples = sampler.generate(n=1000)
# Result: exactly 1000 samples, reusable
```

---

## Best Practices

### 1. Choose the Right Tool

```python
# ❌ Don't filter if you know the constraint ahead of time
samples = sampler.generate(n=10000)
filtered = sampler.filter_samples(samples, 'X > 0')  # Wasteful

# ✅ Use constraints for known requirements
sampler.add_constraint('X > 0')
samples = sampler.generate(n=1000)  # Efficient
```

### 2. Set Tight Parameter Bounds

```python
# ❌ Don't use wide bounds with tight constraints
sampler.add_parameter('X', UniformDistribution(0, 100))
sampler.add_constraint('X < 1')  # 99% rejection rate!

# ✅ Set appropriate bounds
sampler.add_parameter('X', UniformDistribution(0, 1))  # No constraint needed
```

### 3. Use Batch Generation for Large Datasets

```python
# ❌ Don't load millions of samples into memory
samples = sampler.generate(n=10_000_000)  # May crash

# ✅ Use batch generation
for batch in sampler.generate_batches(n_per_batch=100_000, n_batches=100):
    process(batch)
```

### 4. Save Configurations for Reproducibility

```python
# ✅ Save configuration before running experiments
sampler.save_config("experiment_config.json")

# Generate samples
samples = sampler.generate(n=10000)
Sampler.save_data(samples, "experiment_data.csv")

# Later, reproduce exact same setup
sampler2 = Sampler.load_config("experiment_config.json")
```

### 5. Monitor Constraint Acceptance Rates

```python
import time

start = time.time()
samples = sampler.generate(n=1000)
elapsed = time.time() - start

if elapsed > 1.0:  # Taking too long
    print("Warning: Constraints may be too restrictive")
    print("Consider: relaxing constraints or tightening parameter bounds")
```

### 6. Use Appropriate Operators

```python
# ❌ Don't use 'and'/'or' with arrays (causes errors)
# filtered = sampler.filter_samples(samples, 'X > 0 and Y < 10')  # ERROR

# ✅ Use '&'/'|' for element-wise operations
filtered = sampler.filter_samples(samples, '(X > 0) & (Y < 10)')
```

### 7. Combine Features Effectively

```python
# Example: Constrained sampling with batch processing
sampler = Sampler(random_seed=42)
sampler.add_parameter('X', UniformDistribution(0, 10))
sampler.add_parameter('Y', UniformDistribution(0, 10))
sampler.add_derived_parameter('Z', 'X + Y')
sampler.add_constraint('Z < 15')  # Persistent constraint

# Save configuration
sampler.save_config("config.json")

# Generate in batches
for i, batch in enumerate(sampler.generate_batches(n_per_batch=1000, n_batches=10)):
    # Additional filtering if needed
    filtered = sampler.filter_samples(batch, 'X > 5')
    
    # Process
    results = process(filtered)
    
    # Save
    Sampler.save_data(filtered, f"batch_{i}.csv")
```

---

## Examples

### Complete Workflow Example

```python
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import UniformDistribution, NormalDistribution

# 1. Create and configure sampler
sampler = Sampler(random_seed=42)
sampler.add_parameter('stock_A', UniformDistribution(0, 1))
sampler.add_parameter('stock_B', UniformDistribution(0, 1))
sampler.add_parameter('stock_C', UniformDistribution(0, 1))

# Stocks are correlated
sampler.set_correlation('stock_A', 'stock_B', 0.6)
sampler.set_correlation('stock_B', 'stock_C', 0.5)

# Derived parameters
sampler.add_derived_parameter('total', 'stock_A + stock_B + stock_C')
sampler.add_derived_parameter('portfolio_value', 'total * 10000')

# Constraints (portfolio must sum to ~1, allow some slack)
sampler.add_constraint('total >= 0.95')
sampler.add_constraint('total <= 1.05')
sampler.add_constraint('stock_A >= 0.1')  # Minimum 10% in each

# 2. Save configuration
sampler.save_config("portfolio_sampler.json")

# 3. Generate samples in batches
all_results = []
for batch in sampler.generate_batches(n_per_batch=1000, n_batches=10):
    # Run Monte Carlo simulation on batch
    results = monte_carlo_simulation(batch)
    all_results.append(results)
    
    # Save batch
    Sampler.save_data(batch, f"batch_{len(all_results)}.csv")

# 4. Optional: generate specific scenarios
high_stock_a = sampler.generate_conditional(
    n=100,
    condition='stock_A > 0.5',
    return_type='dataframe'
)

# 5. Post-process: filter for interesting cases
samples = sampler.generate(n=10000, return_type='dataframe')
high_value = sampler.filter_samples(samples, 'portfolio_value > 12000')

print(f"Generated {len(samples)} total samples")
print(f"Found {len(high_value)} high-value portfolios")
```

---

## Summary

The advanced features provide flexible tools for real-world sampling workflows:

- **Save/Load**: Reproducibility and configuration sharing
- **Filtering**: Interactive exploration and post-processing
- **Constraints**: Consistent enforcement across generations
- **Conditional**: One-time conditional sampling
- **Batches**: Memory-efficient processing of large datasets

Choose the right tool for your use case, and combine features as needed for complex workflows.
