# Correlation Feature - Quick Start Guide

> **⚠️ DEPRECATED**: This documentation has been superseded by the new unified documentation.
> 
> **Please use:** [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) - See the "Correlations" section.
> 
> This file remains for historical reference only.

---

## 5-Minute Introduction

### What Does It Do?

Makes parameters depend on each other while keeping their individual behaviors.

**Example**: Temperature and pressure often rise together.

```python
sampler = Sampler(n_samples=1000)
sampler.add_parameter("temperature", NormalDistribution(25, 5))
sampler.add_parameter("pressure", NormalDistribution(100, 10))

# Make them correlate
sampler.set_correlation("temperature", "pressure", correlation=0.7)

samples = sampler.generate(as_dataframe=True)
```

### Visual Concept

```
BEFORE correlation:                AFTER correlation:
Temperature                        Temperature
    |  ···  ··                        |   ···
    | ·  ·· ·                         |  ·····
    |· ··  ··                         | ····· 
    |··  · ·                          |·····  
    |·· ·  ·                          |····   
    +--------→ Pressure               +--------→ Pressure
    Random scatter                    Clear trend
```

## Common Use Cases

### 1. Physical Systems
```python
# Temperature affects pressure
sampler.set_correlation("temp", "pressure", 0.8)
```

### 2. Financial Assets
```python
# Stocks tend to move together
sampler.set_correlation("stock_A", "stock_B", 0.6)
# Stocks and bonds often move opposite
sampler.set_correlation("stocks", "bonds", -0.3)
```

### 3. Multiple Correlations
```python
# Three variables
correlation_matrix = np.array([
    [1.0,  0.7, -0.3],  # A vs A, B, C
    [0.7,  1.0,  0.5],  # B vs A, B, C
    [-0.3, 0.5,  1.0]   # C vs A, B, C
])
sampler.set_correlation_matrix(correlation_matrix)
```

## Key Facts

**Works with:**
- ✓ Any distribution (normal, custom, etc.)
- ✓ All sampling methods (random, Sobol, Halton, LHS)
- ✓ Any number of parameters

**Preserves:**
- ✓ Original distributions (means, standard deviations)
- ✓ Shape of distributions (skewness, tails)
- ✓ Specified correlation structure

**Correlation values:**
- Range: -1 to +1
- Positive: Parameters increase together
- Negative: One increases, other decreases
- Zero: Independent (no correlation)

**Note**: Use 0.999 instead of 1.0 for perfect correlation (technical limitation)

## How It Works (Simple Version)

1. **Generate independent samples** from each distribution
2. **Transform to uniform** [0,1] space
3. **Apply correlation** in that space
4. **Transform back** to original distributions

Result: Correlated samples with original distributions preserved!

## Code Patterns

### Pattern 1: Method Chaining
```python
samples = (Sampler(n_samples=1000)
    .add_parameter("A", dist_A)
    .add_parameter("B", dist_B)
    .set_correlation("A", "B", 0.7)
    .generate(as_dataframe=True))
```

### Pattern 2: Step by Step
```python
sampler = Sampler(n_samples=1000, method='sobol')
sampler.add_parameter("A", dist_A)
sampler.add_parameter("B", dist_B)
sampler.add_parameter("C", dist_C)

sampler.set_correlation("A", "B", 0.7)
sampler.set_correlation("A", "C", -0.3)
sampler.set_correlation("B", "C", 0.5)

samples = sampler.generate(as_dataframe=True)
```

### Pattern 3: Matrix (for many parameters)
```python
sampler = Sampler(n_samples=1000)
# Add parameters...

# Define all correlations at once
corr_matrix = np.array([
    [1.0, 0.7, 0.5, -0.2],
    [0.7, 1.0, 0.6, -0.3],
    [0.5, 0.6, 1.0, -0.1],
    [-0.2, -0.3, -0.1, 1.0]
])
sampler.set_correlation_matrix(corr_matrix)

samples = sampler.generate(as_dataframe=True)
```

## Verification

Check if correlations were achieved:

```python
from scipy.stats import spearmanr

# Generate samples
samples = sampler.generate(as_dataframe=True)

# Check correlation
corr = spearmanr(samples['A'], samples['B'])[0]
print(f"Achieved correlation: {corr:.3f}")
```

Expected: Close to your target (within ±0.05 typically)

## Common Mistakes

### ❌ Mistake 1: Using correlation=1.0
```python
# This will fail!
sampler.set_correlation("A", "B", 1.0)  # ERROR
```

**Fix**: Use 0.999 instead
```python
sampler.set_correlation("A", "B", 0.999)  # OK
```

### ❌ Mistake 2: Invalid correlation matrix
```python
# Diagonal must be 1.0
corr = np.array([
    [1.0, 0.5],
    [0.5, 0.9]  # Should be 1.0!
])
```

**Fix**: Always 1.0 on diagonal
```python
corr = np.array([
    [1.0, 0.5],
    [0.5, 1.0]  # Correct
])
```

### ❌ Mistake 3: Non-symmetric matrix
```python
# Off-diagonal must match
corr = np.array([
    [1.0, 0.5, 0.7],
    [0.6, 1.0, 0.8],  # 0.6 != 0.5
    [0.7, 0.8, 1.0]
])
```

**Fix**: Make symmetric
```python
corr = np.array([
    [1.0, 0.5, 0.7],
    [0.5, 1.0, 0.8],  # Match row 1
    [0.7, 0.8, 1.0]
])
```

## When to Use

**Use correlations when:**
- Parameters logically depend on each other
- You have data showing correlation
- Realistic scenarios require dependencies
- Risk analysis needs to account for joint behavior

**Skip correlations when:**
- Parameters are truly independent
- No evidence of correlation
- Correlation is negligible (<0.1)
- Computational cost is critical (rare)

## Performance

Correlation adds minimal overhead:
- 2-10 parameters: Negligible (<1ms per 1000 samples)
- 10-50 parameters: Small (<10ms per 1000 samples)
- 50+ parameters: Noticeable but usually acceptable

For most applications, the cost is worth the realism.

## Next Steps

1. **Try examples**: Run `examples/correlated_sampling_examples.py`
2. **Read docs**: See `docs/CORRELATION_DOCUMENTATION.md` for details
3. **Experiment**: Add correlations to your own samplers
4. **Validate**: Check that achieved correlations match targets

## Getting Help

Common questions:

**Q: What correlation value should I use?**
A: Estimate from historical data or literature. If unsure, try 0.3-0.7 for moderate correlation.

**Q: Can I mix correlated and uncorrelated parameters?**
A: Yes! Only parameters with set_correlation() will be correlated.

**Q: Does it work with QMC?**
A: Yes! Works with Sobol, Halton, LHS, and random sampling.

**Q: How accurate are the correlations?**
A: Typically within ±0.05 of target for n_samples ≥ 1000.

**Q: Can I change correlations after generation?**
A: Yes, call set_correlation() again and generate() new samples.

---

**Full documentation**: `docs/CORRELATION_DOCUMENTATION.md`
**Examples**: `examples/correlated_sampling_examples.py`
**Tests**: `backend/tests/unit/sampling/test_correlation.py`
