# Custom Distribution Fitting Methods

## Overview

The `CustomContinuousDistribution` class now supports **4 different fitting methods** that can be user-specified. This addresses the KDE boundary problem discovered in Example 8, where KDE produced negative processing times from positive-only data.

## The Problem

**KDE (Kernel Density Estimation)** uses Gaussian kernels with infinite support, which means:
- It can produce values outside the original data range
- For positive-only data (times, concentrations), it may generate negative values
- In tests: **109 out of 1000 samples** (10.9%) were negative from positive data!

## The Solutions

### Four Fitting Methods

#### 1. **KDE** (`method='kde'`)
**Original method** - Kernel Density Estimation

```python
dist = CustomContinuousDistribution(data=processing_times, method='kde')
```

**When to use:**
- Shape is unknown and complex
- Data can naturally extend beyond observed range
- Negatives are valid (temperatures, errors, returns)
- Need smoothest possible distribution

**Properties:**
- Most flexible, captures complex shapes
- Smoothest results
- Can produce values outside data range
- May generate negatives from positive data

**Example use cases:**
- Measurement errors (can be ±)
- Temperature variations  
- Financial returns
- Complex multimodal distributions

---

#### 2. **Empirical CDF** (`method='empirical_cdf'`)
**Recommended for bounded data**

```python
dist = CustomContinuousDistribution(data=processing_times, method='empirical_cdf')
```

**When to use:**
- Data must stay within observed bounds
- Processing times, concentrations, ages
- Don't want smoothing artifacts
- Have representative sample of full range

**Properties:**
- **Guaranteed within [data.min(), data.max()]**
- **No negative values from positive data**
- Fast sampling
- Less smooth than KDE

**Example use cases:**
- Processing/reaction times
- Concentrations (must be ≥ 0)
- Prices (must be > 0)
- Physical measurements with natural bounds

**How it works:**
Uses inverse transform sampling with linear interpolation:
```python
sorted_data = np.sort(data)
cdf_values = np.linspace(0, 1, len(data))
samples = np.interp(uniform_random, cdf_values, sorted_data)
```

---

#### 3. **Spline** (`method='spline'`)
**Best of both worlds**

```python
dist = CustomContinuousDistribution(data=processing_times, method='spline')
```

**When to use:**
- Want smooth distribution
- Need bounded samples
- Have enough data (> 100 points)
- Balance between KDE and empirical CDF

**Properties:**
- Smooth like KDE
- Bounded like empirical CDF
- Good balance of properties
- Needs more data than other methods

**Example use cases:**
- Most bounded quantities
- When KDE gives too many negatives
- When empirical CDF is too discrete

**How it works:**
1. Create histogram to estimate PDF
2. Fit cubic spline to PDF
3. Integrate to get smooth CDF
4. Sample using inverse CDF

---

#### 4. **Histogram** (`method='histogram'`)
**Fastest approximation**

```python
dist = CustomContinuousDistribution(data=processing_times, method='histogram', bins=20)
```

**When to use:**
- Need fast approximation
- Large number of samples needed
- Shape is relatively simple
- Exact smoothness not critical

**Properties:**
- Fastest sampling
- Bounded within data range
- Simple, robust
- Slightly discretized (bin edges visible)

**Example use cases:**
- Quick prototyping
- Large-scale sampling (millions)
- Rough approximations
- When speed matters most

**Performance:** Can generate **19+ million samples/second**!

---

## Comparison Results

### Test: Gamma-distributed data (n=500)

| Method | Negatives | Below Min | Above Max | Mean | Notes |
|--------|-----------|-----------|-----------|------|-------|
| **KDE** | 20/1000 | 38/1000 | 0/1000 | 29.80 | Produces negatives |
| **Empirical CDF** | 0/1000 | 0/1000 | 0/1000 | 28.68 | Perfect bounds |
| **Spline** | 0/1000 | 0/1000 | 0/1000 | 29.64 | Smooth + bounded |
| **Histogram** | 0/1000 | 0/1000 | 0/1000 | 30.06 | Fast |

### Test: Processing times (exponential, n=200)

| Method | Negative Times | Min Sample | Status |
|--------|----------------|------------|--------|
| **KDE** | 109/1000 (10.9%) | -3.15 min | Invalid |
| **Empirical CDF** | 0/1000 | 0.03 min | Valid |
| **Spline** | 0/1000 | 0.58 min | Valid |
| **Histogram** | 0/1000 | 0.03 min | Valid |

---

## Quick Decision Tree

```
Start here: What kind of data?

├─ Naturally BOUNDED (times, concentrations, prices)?
│  │
│  ├─ Need smoothest possible? → SPLINE
│  ├─ Need guaranteed bounds? → EMPIRICAL_CDF
│  └─ Need speed? → HISTOGRAM
│
└─ Can be UNBOUNDED (errors, temperatures, returns)?
   │
   ├─ Complex/multimodal shape? → KDE
   ├─ Simple unimodal? → SPLINE or KDE
   └─ Need speed? → HISTOGRAM
```

---

## Default Recommendations

### Bounded Positive Data (times, concentrations)
```python
# Safest option
dist = CustomContinuousDistribution(data=times, method='empirical_cdf')

# If want smoothness
dist = CustomContinuousDistribution(data=times, method='spline')
```

### Unbounded Data (errors, temperatures)
```python
# Most flexible
dist = CustomContinuousDistribution(data=errors, method='kde')
```

### Need Speed
```python
# Fastest
dist = CustomContinuousDistribution(data=values, method='histogram')
```

### Complex Shapes
```python
# Best at capturing complexity
dist = CustomContinuousDistribution(data=multimodal, method='kde')
```

### Not Sure?
```python
# Safest, no surprises
dist = CustomContinuousDistribution(data=data, method='empirical_cdf')
```

---

## Code Examples

See `examples/custom_methods_comparison.py` for:
- Side-by-side comparison of all 4 methods
- Boundary problem demonstration
- Performance benchmarks
- Practical use cases
- Comprehensive selection guide

---

## Implementation Details

### Empirical CDF
```python
def _fit_empirical_cdf(self, data):
    self._sorted_data = np.sort(data)
    self._cdf_values = np.linspace(0, 1, len(data))
    
def _sample_empirical_cdf(self, size):
    u = np.random.uniform(0, 1, size)
    return np.interp(u, self._cdf_values, self._sorted_data)
```

### Spline
```python
def _fit_spline(self, data):
    hist, edges = np.histogram(data, bins=self.bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Fit spline to PDF
    pdf_spline = CubicSpline(centers, hist, bc_type='natural')
    
    # Integrate to get CDF
    x_dense = np.linspace(data.min(), data.max(), 1000)
    pdf_dense = pdf_spline(x_dense)
    cdf_dense = cumulative_trapezoid(pdf_dense, x_dense, initial=0)
    
    self._cdf_spline = interp1d(cdf_dense, x_dense)
```

### Histogram
```python
def _fit_histogram(self, data):
    hist, edges = np.histogram(data, bins=self.bins)
    self._hist_edges = edges
    self._hist_probs = hist / hist.sum()
    
def _sample_histogram(self, size):
    # Sample bin indices
    bin_indices = np.random.choice(len(self._hist_probs), 
                                   size=size, 
                                   p=self._hist_probs)
    # Uniform within each bin
    left = self._hist_edges[bin_indices]
    right = self._hist_edges[bin_indices + 1]
    return np.random.uniform(left, right)
```

---

## Testing

All methods are tested in:
- `backend/tests/unit/sampling/test_distributions.py`
- 11 tests for `CustomContinuousDistribution`
- **74 total tests passing** in sampling module

Test coverage includes:
- All 4 methods work correctly
- Empirical CDF stays within bounds
- Negative handling (truncate/shift/allow)
- CSV loading
- Reproducibility
- Invalid method detection
- Method appears in __repr__

---

## Performance

Timing 10,000 samples:

| Method | Setup Time | Sample Time | Total |
|--------|------------|-------------|-------|
| KDE | 0.00 ms | 1.00 ms | 1.00 ms |
| Empirical CDF | 0.00 ms | 0.00 ms | **0.00 ms** (fastest) |
| Spline | 0.00 ms | 1.02 ms | 1.02 ms |
| Histogram | 0.98 ms | 0.00 ms | 0.98 ms |

For 10 million samples:
- Histogram: **0.51 seconds** (~20 million/sec)

---

## Summary

The addition of multiple fitting methods solves the KDE boundary problem while maintaining flexibility:

1. **Empirical CDF**: Safest for bounded data - guaranteed within range
2. **Spline**: Smooth + bounded - best compromise
3. **Histogram**: Fast approximation - great for large-scale sampling
4. **KDE**: Original method - best for complex unbounded distributions

**Key insight**: Choose the method based on your data's natural constraints, not just what's smoothest!
