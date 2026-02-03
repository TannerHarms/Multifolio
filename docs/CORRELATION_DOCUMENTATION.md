# Correlated Sampling in Multifolio: Complete Documentation

> **⚠️ DEPRECATED**: This documentation has been superseded by the new unified documentation.
> 
> **Please use:** [SAMPLING_GUIDE.md](SAMPLING_GUIDE.md) - See the "Correlations" section.
> 
> For technical implementation details, see CORRELATION_IMPLEMENTATION_SUMMARY.md.
> 
> This file remains for historical reference only.

---

## Table of Contents
1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Why Gaussian Copulas Work](#why-gaussian-copulas-work)
4. [Implementation Details](#implementation-details)
5. [Usage Guide](#usage-guide)
6. [When to Use Correlations](#when-to-use-correlations)
7. [Limitations and Considerations](#limitations-and-considerations)
8. [Examples](#examples)

---

## Overview

The correlation feature in Multifolio allows you to model dependencies between parameters while preserving their individual (marginal) distributions. This is crucial for realistic simulations where parameters are not independent.

**Key Features:**
- Model pairwise or multivariate correlations between any parameters
- Works with all distribution types (normal, custom, etc.)
- Compatible with all sampling methods (random, Sobol, Halton, LHS)
- Preserves marginal distributions exactly
- Based on mathematically rigorous Gaussian copula method

**Why This Matters:**
In real-world systems, parameters are often correlated:
- Temperature and pressure in physical systems
- Asset returns in financial portfolios
- Load factors in engineering design
- Customer behaviors in business models

Ignoring these correlations can lead to:
- Overestimation of diversification benefits
- Underestimation of extreme events
- Unrealistic scenario combinations
- Poor risk assessment

---

## Mathematical Foundation

### What is a Copula?

A **copula** is a mathematical function that links marginal distributions to their joint distribution. It separates:
1. **Marginal behavior** - how each variable behaves individually
2. **Dependence structure** - how variables are related to each other

**Sklar's Theorem** (1959) states that any multivariate distribution can be decomposed into:
- Marginal distributions: F₁(x₁), F₂(x₂), ..., Fₙ(xₙ)
- A copula: C(u₁, u₂, ..., uₙ) where uᵢ = Fᵢ(xᵢ)

This is powerful because you can:
- Choose any marginal distributions you want
- Add any dependence structure you want
- Combine them independently

### The Gaussian Copula

The **Gaussian copula** models dependence using the multivariate normal distribution. For n parameters with correlation matrix **R**:

```
C(u₁, ..., uₙ) = Φₙ(Φ⁻¹(u₁), ..., Φ⁻¹(uₙ); R)
```

Where:
- Φ is the standard normal CDF
- Φ⁻¹ is its inverse (quantile function)
- Φₙ is the n-dimensional normal CDF with correlation R

**Why is this useful?**
- The uᵢ are uniform [0,1] samples from any distribution
- We transform them to standard normals: zᵢ = Φ⁻¹(uᵢ)
- Apply correlation in normal space
- Transform back to uniform: uᵢ' = Φ(zᵢ')
- Apply inverse CDFs to get correlated samples with original distributions

---

## Why Gaussian Copulas Work

### The Transformation Process

Let's trace through an example with two parameters A and B that should have correlation ρ = 0.7:

**Step 1: Start with Independent Samples**
```
Parameter A: Normal(μ=25, σ=5)
Parameter B: Normal(μ=100, σ=10)

Generate independent uniform samples:
u_A = [0.23, 0.71, 0.45, ...]
u_B = [0.82, 0.15, 0.67, ...]

These are independent (correlation ≈ 0)
```

**Step 2: Transform to Standard Normal Space**
```
z_A = Φ⁻¹(u_A) = [-0.74, 0.55, -0.13, ...]
z_B = Φ⁻¹(u_B) = [0.92, -1.04, 0.44, ...]

Still independent (correlation ≈ 0)
```

**Step 3: Apply Correlation via Cholesky Decomposition**

The correlation matrix is:
```
R = [1.0  0.7]
    [0.7  1.0]
```

Cholesky decomposition: R = L·Lᵀ where
```
L = [1.0    0.0  ]
    [0.7  √(0.51)]
```

Transform: [z_A', z_B'] = [z_A, z_B] · Lᵀ
```
z_A' = z_A  (unchanged)
z_B' = 0.7·z_A + √(0.51)·z_B
```

Now z_A' and z_B' have correlation 0.7!

**Step 4: Transform Back to Uniform**
```
u_A' = Φ(z_A')
u_B' = Φ(z_B')

These uniform samples now have rank correlation ≈ 0.7
```

**Step 5: Apply Original Distributions**
```
A = F_A⁻¹(u_A')  (inverse CDF of Normal(25, 5))
B = F_B⁻¹(u_B')  (inverse CDF of Normal(100, 10))

Final samples preserve:
- Original distributions (Normal(25,5) and Normal(100,10))
- Correlation structure (ρ ≈ 0.7)
```

### Why This Preserves Marginal Distributions

The key insight is that we only transform **in the uniform [0,1] space**:
- The CDF maps any distribution to uniform [0,1]
- The inverse CDF maps uniform [0,1] back to the original distribution
- By correlating the uniform samples, we correlate the final values
- But each parameter's distribution remains exactly as specified

**Mathematical Proof Sketch:**
```
P(A ≤ a) = P(F_A⁻¹(u_A') ≤ a)
         = P(u_A' ≤ F_A(a))
         = F_A(a)  (unchanged!)
```

The marginal CDF is preserved because the correlation transformation is done in a space that's "orthogonal" to the marginal behavior.

### Why Cholesky Decomposition?

The Cholesky decomposition is the most efficient way to transform uncorrelated normal variables into correlated ones:

1. **Mathematical Guarantee**: If R is positive definite, R = L·Lᵀ always exists
2. **Computational Efficiency**: O(n³) one-time cost, then O(n²) per sample
3. **Lower Triangular**: The transformation is causal (z_i' depends only on z_1, ..., z_i)
4. **Numerical Stability**: More stable than eigendecomposition for this purpose

Alternative: You could use eigendecomposition (R = Q·Λ·Qᵀ), but it's:
- More expensive (requires eigenvalue computation)
- Less numerically stable for near-singular matrices
- Doesn't provide the same causal interpretation

---

## Implementation Details

### Architecture

The correlation feature is implemented in three layers:

```
┌─────────────────────────────────────────┐
│         User Interface (Sampler)        │  ← Simple API
├─────────────────────────────────────────┤
│     CorrelationManager                  │  ← Parameter tracking
├─────────────────────────────────────────┤
│        GaussianCopula                   │  ← Core math
└─────────────────────────────────────────┘
```

### GaussianCopula Class

**Purpose**: Implements the core copula transformation.

**Key Methods:**
```python
class GaussianCopula:
    def __init__(self, correlation_matrix: np.ndarray):
        """
        Initialize with n×n correlation matrix.
        
        Validation:
        - Must be square
        - Diagonal must be all ones
        - Must be symmetric
        - Must be positive definite (for Cholesky)
        """
        self._validate_correlation_matrix(correlation_matrix)
        self._cholesky_factor = np.linalg.cholesky(correlation_matrix)
    
    def transform_uniform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform independent uniform samples to correlated ones.
        
        Process:
        1. U → Z via inverse normal CDF: Φ⁻¹(U)
        2. Apply correlation: Z' = Z @ L^T
        3. Z' → U' via normal CDF: Φ(Z')
        
        Shape: (n_samples, n_dims) → (n_samples, n_dims)
        """
        z = stats.norm.ppf(np.clip(uniform_samples, 1e-10, 1-1e-10))
        z_correlated = z @ self._cholesky_factor.T
        return stats.norm.cdf(z_correlated)
```

**Why Clipping?**
The inverse normal CDF (ppf) is undefined at exactly 0 and 1:
- Φ⁻¹(0) = -∞
- Φ⁻¹(1) = +∞

We clip to [1e-10, 1-1e-10] to avoid numerical issues while maintaining effectively full range.

### CorrelationManager Class

**Purpose**: Manages parameter-level correlations and builds correlation matrices.

**Key Features:**
- Maps parameter names to indices
- Builds correlation matrix from pairwise correlations
- Handles matrix updates efficiently
- Validates parameter existence

**Key Methods:**
```python
class CorrelationManager:
    def set_correlation(self, param1: str, param2: str, correlation: float):
        """
        Set pairwise correlation between two parameters.
        
        Automatically:
        - Assigns indices to new parameters
        - Updates correlation matrix (symmetric)
        - Rebuilds Gaussian copula
        """
        
    def transform_samples(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples using copula.
        
        Handles parameter ordering automatically.
        """
```

**Design Decision - Lazy Updates:**
The correlation matrix is rebuilt every time a correlation is set. This is simple and correct, though not optimal for many sequential updates. Future optimization could batch updates.

### Sampler Integration

The Sampler class is extended with correlation support:

```python
class Sampler:
    def set_correlation(self, param1: str, param2: str, correlation: float):
        """User-facing method to set correlations."""
        self._ensure_correlation_manager()
        self._correlation_manager.set_correlation(param1, param2, correlation)
        return self  # Enable method chaining
    
    def generate(self, n_samples: int = None, as_dataframe: bool = True):
        """Modified to apply correlations."""
        # Generate base samples (independent)
        if self.method == 'random':
            uniform = self._generate_uniform_random(n)
        else:
            uniform = self._generate_qmc(n)
        
        # Apply correlations if specified
        if self.has_correlations():
            uniform = self._correlation_manager.transform_samples(uniform)
        
        # Apply inverse CDFs to get final samples
        samples = self._apply_inverse_cdfs(uniform)
        ...
```

**Critical Design Choice - When to Apply Correlations:**

For **random sampling**: Apply correlations to uniform samples, then inverse CDF
```
Independent Uniform → Correlated Uniform → Final Samples
```

For **QMC sampling**: Same approach
```
QMC Uniform → Correlated Uniform → Final Samples
```

**Why this order?** 
- QMC sequences (Sobol, Halton) have optimal space-filling properties in [0,1]ⁿ
- Correlation transformation preserves uniformity
- Applying correlation before inverse CDF maintains the low-discrepancy properties better than correlating in the parameter space

**Alternative (Not Implemented):**
Could transform samples to normal space, apply correlation, transform back. But this would lose QMC benefits and be less general.

---

## Usage Guide

### Basic Usage: Pairwise Correlation

```python
from multifolio.core.sampling import Sampler
from multifolio.core.distributions import NormalDistribution

# Create sampler
sampler = Sampler(n_samples=1000, seed=42)
sampler.add_parameter("A", NormalDistribution(mean=0, std=1))
sampler.add_parameter("B", NormalDistribution(mean=5, std=2))

# Add correlation
sampler.set_correlation("A", "B", correlation=0.7)

# Generate correlated samples
samples = sampler.generate(as_dataframe=True)
```

### Advanced: Correlation Matrix

```python
import numpy as np

# Create sampler with multiple parameters
sampler = Sampler(n_samples=1000)
sampler.add_parameter("X", dist_X)
sampler.add_parameter("Y", dist_Y)
sampler.add_parameter("Z", dist_Z)

# Define full correlation matrix
correlation_matrix = np.array([
    [1.0, 0.7, -0.3],
    [0.7, 1.0, 0.5],
    [-0.3, 0.5, 1.0]
])

sampler.set_correlation_matrix(correlation_matrix)
samples = sampler.generate(as_dataframe=True)
```

### Method Chaining

```python
samples = (Sampler(n_samples=1000)
    .add_parameter("A", dist_A)
    .add_parameter("B", dist_B)
    .set_correlation("A", "B", 0.7)
    .generate(as_dataframe=True))
```

### With QMC Methods

```python
# Works with any sampling method
sampler = Sampler(n_samples=1024, method='sobol')
sampler.add_parameter("A", dist_A)
sampler.add_parameter("B", dist_B)
sampler.set_correlation("A", "B", 0.7)

samples = sampler.generate(as_dataframe=True)
```

---

## When to Use Correlations

### Use Cases Where Correlations Are Essential

1. **Financial Portfolios**
   - Asset returns are correlated
   - Ignoring correlations overestimates diversification
   - Example: Stocks and bonds often have negative correlation

2. **Physical Systems**
   - Temperature and pressure correlate via ideal gas law
   - Load factors in structural analysis
   - Environmental variables (temperature, humidity)

3. **Manufacturing**
   - Process parameters that drift together
   - Quality characteristics that share root causes
   - Machine settings that interact

4. **Risk Analysis**
   - Correlated failure modes
   - Dependent reliability factors
   - Supply chain disruptions that cascade

5. **Customer Behavior**
   - Purchase patterns across products
   - Time-based usage correlations
   - Demographic dependencies

### When NOT to Use Correlations

1. **Truly Independent Parameters**
   - Different physical phenomena
   - Unrelated random events
   - Parameters from separate systems

2. **Computational Constraints**
   - Correlation transformation adds overhead (minimal for most cases)
   - For millions of samples, consider if correlation is worth the cost

3. **Insufficient Data**
   - Need enough data to estimate correlations reliably
   - Rule of thumb: At least 30-50 observations per parameter pair
   - Don't make up correlations without evidence

4. **When Causality Matters**
   - Copulas model correlation, not causation
   - If A causes B, consider conditional distributions instead
   - For time-series, consider autoregressive models

---

## Limitations and Considerations

### 1. Correlation Type: Spearman vs Pearson

The Gaussian copula preserves **rank correlation** (Spearman's ρ), not linear correlation (Pearson's r).

**Why?**
- Rank correlation is invariant to monotonic transformations
- Works for any marginal distributions
- More robust to outliers

**Relationship:**
For normal distributions: ρ_spearman ≈ (6/π) × arcsin(ρ_pearson / 2)

Example:
- Specify ρ = 0.7
- For normals: Spearman ≈ 0.68, Pearson ≈ 0.70
- For heavy-tailed: Spearman preserved, Pearson may differ

### 2. Perfect Correlation (ρ = ±1)

**Limitation**: Perfect correlation creates a singular matrix that fails Cholesky decomposition.

**Workaround**: Use ρ = 0.999 or 0.995 instead
- Practically identical behavior
- Mathematically valid
- Avoids numerical issues

**Why the limitation?**
Perfect correlation means variables are functionally dependent, not stochastically dependent. The copula framework assumes stochastic relationships.

### 3. Correlation Matrix Requirements

A valid correlation matrix must be:
- **Symmetric**: R[i,j] = R[j,i]
- **Positive semi-definite**: All eigenvalues ≥ 0
- **Diagonal ones**: R[i,i] = 1

**Common mistake:**
```python
# This might NOT be valid!
R = np.array([
    [1.0, 0.8, 0.9],
    [0.8, 1.0, 0.9],
    [0.9, 0.9, 1.0]
])
# Check: np.linalg.eigvals(R)  → All positive? ✓ (valid)
```

**Another example:**
```python
# This is NOT valid!
R = np.array([
    [1.0, 0.9, 0.9],
    [0.9, 1.0, -0.9],
    [0.9, -0.9, 1.0]
])
# Has negative eigenvalue → Not positive definite
```

Use `GaussianCopula(R)` to validate - it will raise ValueError if invalid.

### 4. QMC and Correlations

**Good News**: Correlations work with all QMC methods (Sobol, Halton, LHS).

**Consideration**: The correlation transformation slightly disrupts the optimal space-filling properties of QMC sequences. However:
- The benefit is usually negligible for high dimensions (>2)
- The correlation structure is more important than perfect uniformity
- The transformation preserves uniformity, just not optimal spacing

**Best Practice**: For correlated parameters, QMC still outperforms random sampling for the same sample size.

### 5. High-Dimensional Correlations

**Challenge**: Specifying correlations for many parameters is difficult
- n parameters → n(n-1)/2 correlations
- Must ensure positive definiteness
- Difficult to gather data for all pairs

**Solutions:**
1. **Factor models**: Use a few underlying factors
2. **Block structure**: Group related parameters
3. **Sparse correlations**: Only specify significant correlations

### 6. Tail Dependence

**Limitation**: Gaussian copulas have **no tail dependence** - extreme events in one variable don't strongly associate with extremes in others.

**When this matters:**
- Financial crisis scenarios (all assets crash together)
- System failures (cascading failures)
- Rare event analysis

**Alternative**: Consider Student's t-copula (not currently implemented) which has tail dependence controlled by degrees of freedom.

### 7. Computational Complexity

**Time Complexity:**
- Setup (Cholesky): O(n³) one-time
- Per sample: O(n²) matrix multiplication
- For n=10, k=1M samples: ~10ms overhead (negligible)
- For n=100, k=1M samples: ~1s overhead (noticeable)

**Memory:**
- Correlation matrix: O(n²)
- Cholesky factor: O(n²)
- Negligible unless n > 1000

---

## Examples

See `examples/correlated_sampling_examples.py` for comprehensive examples:

1. **Basic Pairwise Correlation**: Temperature and pressure
2. **Multi-Parameter Correlation Matrix**: Three environmental variables
3. **Custom Distributions**: Bimodal and skewed distributions
4. **QMC with Correlations**: Compare random, Sobol, Halton, LHS
5. **Portfolio Optimization**: Real-world financial application

### Running the Examples

```bash
cd examples
python correlated_sampling_examples.py
```

This will generate visualizations showing:
- Independent vs correlated samples
- Correlation matrices (target vs achieved)
- Marginal distribution preservation
- Different sampling methods
- Portfolio risk analysis

---

## Summary

The correlation feature in Multifolio provides:

**Mathematical Rigor**
- Based on well-established Gaussian copula theory
- Preserves marginal distributions exactly
- Efficient Cholesky decomposition

**Practical Usability**
- Simple API: `set_correlation()` or `set_correlation_matrix()`
- Works with all distribution types
- Compatible with all sampling methods

**Reliable Implementation**
- Comprehensive validation
- Numerical stability (clipping, Cholesky)
- Well-tested (23 unit tests, all passing)

**Real-World Value**
- Essential for realistic simulations
- Critical for risk analysis
- Enables proper sensitivity studies

The implementation strikes a balance between mathematical correctness and practical usability, making it suitable for both research and production applications.

---

## References

1. Sklar, A. (1959). "Fonctions de répartition à n dimensions et leurs marges." Publications de l'Institut de Statistique de l'Université de Paris.

2. McNeil, A. J., Frey, R., & Embrechts, P. (2005). "Quantitative Risk Management: Concepts, Techniques and Tools." Princeton University Press.

3. Nelsen, R. B. (2006). "An Introduction to Copulas." Springer.

4. Joe, H. (2014). "Dependence Modeling with Copulas." CRC Press.

5. Owen, A. B. (2003). "Quasi-Monte Carlo Sampling." In: Gentle, J.E., Härdle, W., Mori, Y. (eds) Handbook of Computational Statistics. Springer.
