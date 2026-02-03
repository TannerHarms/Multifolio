"""
Example demonstrating new common distributions and custom distributions.

This includes:
- Common distributions: Exponential, Gamma, LogNormal, Weibull, Triangular
- Custom continuous distributions from data, files, and functions
- Custom discrete distributions from probabilities, data, and functions
- Negative value handling options
"""

import sys
from pathlib import Path
import numpy as np
import tempfile

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    # New common distributions
    ExponentialDistribution,
    GammaDistribution,
    LogNormalDistribution,
    WeibullDistribution,
    TriangularDistribution,
    # Custom distributions
    CustomContinuousDistribution,
    CustomDiscreteDistribution,
)


def example_1_common_distributions():
    """Demonstrate new common distributions."""
    print("=" * 70)
    print("Example 1: Common Probability Distributions")
    print("=" * 70)
    
    # Exponential: time between events
    exp_dist = ExponentialDistribution(rate=0.5)  # Mean = 2.0
    exp_samples = exp_dist.sample(size=5)
    print("\nExponentialDistribution(rate=0.5) - Mean=2.0")
    print(f"  Samples: {exp_samples}")
    print(f"  Sample mean: {exp_samples.mean():.2f}")
    
    # Gamma: wait times, shape parameter
    gamma_dist = GammaDistribution(shape=2, scale=2)  # Mean = 4.0
    gamma_samples = gamma_dist.sample(size=5)
    print("\nGammaDistribution(shape=2, scale=2) - Mean=4.0")
    print(f"  Samples: {gamma_samples}")
    print(f"  Sample mean: {gamma_samples.mean():.2f}")
    
    # LogNormal: multiplicative processes
    lognorm_dist = LogNormalDistribution(mu=0, sigma=0.5)
    lognorm_samples = lognorm_dist.sample(size=5)
    print("\nLogNormalDistribution(mu=0, sigma=0.5)")
    print(f"  Samples: {lognorm_samples}")
    print(f"  Sample mean: {lognorm_samples.mean():.2f}")
    
    # Weibull: failure times, reliability
    weibull_dist = WeibullDistribution(shape=1.5, scale=2.0)
    weibull_samples = weibull_dist.sample(size=5)
    print("\nWeibullDistribution(shape=1.5, scale=2.0)")
    print(f"  Samples: {weibull_samples}")
    print(f"  Sample mean: {weibull_samples.mean():.2f}")
    
    # Triangular: min/mode/max estimates
    tri_dist = TriangularDistribution(low=1, mode=3, high=6)
    tri_samples = tri_dist.sample(size=5)
    print("\nTriangularDistribution(low=1, mode=3, high=6)")
    print(f"  Samples: {tri_samples}")
    print(f"  Sample mean: {tri_samples.mean():.2f} (expected ~3.3)")
    print()


def example_2_custom_continuous_from_data():
    """Create custom continuous distribution from data array."""
    print("=" * 70)
    print("Example 2: Custom Continuous Distribution from Data")
    print("=" * 70)
    
    # Generate some sample data (e.g., from experiments)
    np.random.seed(42)
    measured_data = np.random.gamma(shape=3, scale=2, size=500)
    
    print(f"\nOriginal data: {len(measured_data)} measurements")
    print(f"  Range: [{measured_data.min():.2f}, {measured_data.max():.2f}]")
    print(f"  Mean: {measured_data.mean():.2f}")
    print(f"  Std: {measured_data.std():.2f}")
    
    # Create distribution using KDE
    custom_dist = CustomContinuousDistribution(data=measured_data)
    
    # Generate new samples
    new_samples = custom_dist.sample(size=10)
    print(f"\nNew samples from fitted distribution:")
    print(f"  {new_samples}")
    print(f"  Mean: {new_samples.mean():.2f}")
    print()


def example_3_custom_continuous_from_file():
    """Load custom continuous distribution from CSV file."""
    print("=" * 70)
    print("Example 3: Custom Continuous Distribution from CSV File")
    print("=" * 70)
    
    # Create temporary CSV file with data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("measurement\n")
        for val in np.random.exponential(scale=5, size=100):
            f.write(f"{val:.4f}\n")
        temp_file = f.name
    
    try:
        print(f"\nLoading data from: {temp_file}")
        
        # Load distribution from file
        dist = CustomContinuousDistribution(data=temp_file)
        
        samples = dist.sample(size=10)
        print(f"Samples: {samples}")
        print(f"Mean: {samples.mean():.2f}")
        
    finally:
        Path(temp_file).unlink()
    
    print()


def example_4_custom_continuous_from_function():
    """Create custom distribution from a sampling function."""
    print("=" * 70)
    print("Example 4: Custom Distribution from Function")
    print("=" * 70)
    
    # Define a custom mixture distribution
    def mixture_sampler(size):
        """Mix of two normals: 70% N(0,1) + 30% N(5,1)."""
        n1 = int(size * 0.7)
        n2 = size - n1
        component1 = np.random.normal(0, 1, size=n1)
        component2 = np.random.normal(5, 1, size=n2)
        mixture = np.concatenate([component1, component2])
        np.random.shuffle(mixture)
        return mixture
    
    # Create distribution
    dist = CustomContinuousDistribution(data=mixture_sampler)
    
    samples = dist.sample(size=100)
    print(f"\nBimodal mixture distribution:")
    print(f"  100 samples - Mean: {samples.mean():.2f}")
    print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")
    print(f"  (Expected mean ≈ 1.5 from 0.7*0 + 0.3*5)")
    print()


def example_5_negative_handling():
    """Demonstrate negative value handling options."""
    print("=" * 70)
    print("Example 5: Handling Negative Values")
    print("=" * 70)
    
    # Data with some negative values
    data_with_negatives = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    print(f"\nOriginal data: {data_with_negatives}")
    print(f"  Min: {data_with_negatives.min()}, Max: {data_with_negatives.max()}")
    
    # Option 1: Allow negatives (default)
    dist_allow = CustomContinuousDistribution(
        data=data_with_negatives,
        negative_handling='allow'
    )
    samples_allow = dist_allow.sample(size=10)
    print(f"\nAllow negatives:")
    print(f"  Samples: {samples_allow}")
    print(f"  Min: {samples_allow.min():.2f}")
    
    # Option 2: Truncate to zero
    dist_truncate = CustomContinuousDistribution(
        data=data_with_negatives,
        negative_handling='truncate'
    )
    samples_truncate = dist_truncate.sample(size=10)
    print(f"\nTruncate to zero:")
    print(f"  Samples: {samples_truncate}")
    print(f"  Min: {samples_truncate.min():.2f} (should be >= 0)")
    
    # Option 3: Shift distribution
    dist_shift = CustomContinuousDistribution(
        data=data_with_negatives,
        negative_handling='shift'
    )
    samples_shift = dist_shift.sample(size=10)
    print(f"\nShift distribution (min becomes 0):")
    print(f"  Samples: {samples_shift}")
    print(f"  Min: {samples_shift.min():.2f} (shifted by +2)")
    print()


def example_6_custom_discrete_from_probabilities():
    """Create custom discrete distribution from probability dict."""
    print("=" * 70)
    print("Example 6: Custom Discrete Distribution from Probabilities")
    print("=" * 70)
    
    # Loaded dice example
    dice_probs = {
        1: 0.10,
        2: 0.15,
        3: 0.15,
        4: 0.15,
        5: 0.15,
        6: 0.30,  # Weighted toward 6
    }
    
    print("\nLoaded dice probabilities:")
    for value, prob in dice_probs.items():
        print(f"  {value}: {prob:.0%}")
    
    dist = CustomDiscreteDistribution(data=dice_probs)
    
    # Roll the dice
    rolls = dist.sample(size=20)
    print(f"\n20 rolls: {rolls}")
    
    # Check frequency
    unique, counts = np.unique(rolls, return_counts=True)
    print("\nObserved frequencies:")
    for val, count in zip(unique, counts):
        print(f"  {val}: {count}/20 ({count/20:.0%})")
    print()


def example_7_custom_discrete_from_data():
    """Create discrete distribution from observed data."""
    print("=" * 70)
    print("Example 7: Custom Discrete Distribution from Observed Data")
    print("=" * 70)
    
    # Survey responses (1-5 scale)
    survey_data = [3, 4, 4, 5, 3, 4, 3, 5, 4, 4, 5, 5, 3, 4, 5, 2, 4, 4, 3, 5]
    
    print(f"\nSurvey responses (n={len(survey_data)}): {survey_data}")
    
    # Create distribution from observed frequencies
    dist = CustomDiscreteDistribution(data=survey_data)
    
    # Generate synthetic responses
    synthetic = dist.sample(size=100)
    
    print(f"\nGenerated 100 synthetic responses:")
    unique, counts = np.unique(synthetic, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Rating {val}: {count}/100 ({count}%)")
    print()


def example_8_integration_with_sampler():
    """Use custom distributions in multi-parameter Sampler."""
    print("=" * 70)
    print("Example 8: Custom Distributions with Sampler")
    print("=" * 70)
    
    print("\nExperimental design mixing standard and custom distributions:")
    
    # Historical processing time data
    # Note: For naturally positive distributions like processing times,
    # it's better to fit a parametric distribution (like Gamma) rather than
    # use KDE, which can produce negative values due to kernel tails.
    historical_times = np.random.gamma(shape=2, scale=15, size=200)
    
    sampler = Sampler(random_seed=42)
    
    # Standard distributions
    sampler.add_parameter('temperature', TriangularDistribution(20, 25, 35))
    sampler.add_parameter('batch_size', GammaDistribution(shape=3, scale=10))
    
    # For processing times, use parametric Gamma distribution instead of KDE
    # This avoids negative values that can occur with KDE kernel tails
    sampler.add_parameter('processing_time', 
                         GammaDistribution(shape=2, scale=15))  # Same parameters as data
    
    # Custom discrete distribution for priority (this is fine - discrete data)
    priority_probs = {1: 0.1, 2: 0.3, 3: 0.4, 4: 0.2}
    sampler.add_parameter('priority',
                         CustomDiscreteDistribution(data=priority_probs))
    
    # Generate experimental conditions
    experiments = sampler.generate(n=10, return_type='dataframe')
    
    print("\nGenerated 10 experimental conditions:")
    print(experiments.to_string(index=True, float_format=lambda x: f'{x:.2f}'))
    
    print("\n\nNote: We used GammaDistribution for processing_time instead of")
    print("CustomContinuousDistribution because:")
    print("  1. Processing times are naturally positive (gamma-like)")
    print("  2. KDE can produce negative values due to Gaussian kernel tails")
    print("  3. Parametric distributions are better when you know the shape")
    print("\nCustom distributions are best for:")
    print("  - Unknown/complex shapes that don't fit standard distributions")
    print("  - Data that includes the full valid range (e.g., temperatures can be negative)")
    print("  - Discrete probability mappings")
    print()


def example_9_when_to_use_custom_vs_parametric():
    """Demonstrate when to use custom distributions vs parametric ones."""
    print("=" * 70)
    print("Example 9: When to Use Custom vs Parametric Distributions")
    print("=" * 70)
    
    # Case 1: Known parametric distribution - use parametric
    print("\nCase 1: Processing times (known to be gamma-like)")
    print("  ✓ Use GammaDistribution - avoids KDE boundary issues")
    processing_times = GammaDistribution(shape=2, scale=15)
    samples = processing_times.sample(size=5)
    print(f"  Samples: {samples}")
    print(f"  All positive: {(samples >= 0).all()}")
    
    # Case 2: Unknown complex shape - use custom
    print("\nCase 2: Measurement errors (complex, unknown shape)")
    print("  ✓ Use CustomContinuousDistribution - captures actual behavior")
    # Simulated complex measurement errors (could be from real instruments)
    measurement_errors = np.concatenate([
        np.random.normal(-0.5, 0.2, 300),  # Systematic bias
        np.random.normal(0.3, 0.5, 200),   # Different mode
        np.random.uniform(-2, 2, 100)      # Some outliers
    ])
    error_dist = CustomContinuousDistribution(data=measurement_errors, 
                                              negative_handling='allow')
    samples = error_dist.sample(size=5)
    print(f"  Samples: {samples}")
    print(f"  Can be negative: True (errors can be positive or negative)")
    
    # Case 3: Bounded quantity that can go to zero - be careful with KDE
    print("\nCase 3: Concentration measurements (must be >= 0)")
    print("  ⚠ KDE can produce negative values near boundaries")
    concentrations = np.random.exponential(scale=2, size=100)
    
    # Show the problem
    kde_dist = CustomContinuousDistribution(data=concentrations, 
                                           negative_handling='allow')
    kde_samples = kde_dist.sample(size=1000)
    n_negative = (kde_samples < 0).sum()
    print(f"  KDE with 'allow': {n_negative}/1000 samples are negative!")
    
    # Solutions:
    print("\n  Better solutions:")
    print("    1. Use ExponentialDistribution if shape fits")
    exp_dist = ExponentialDistribution(rate=0.5)
    exp_samples = exp_dist.sample(size=5)
    print(f"       Exponential: {exp_samples} (all positive)")
    
    print("    2. Use 'truncate' if you must use KDE (but understand the bias)")
    truncated_dist = CustomContinuousDistribution(data=concentrations,
                                                  negative_handling='truncate')
    trunc_samples = truncated_dist.sample(size=5)
    print(f"       Truncated KDE: {trunc_samples}")
    print(f"       (Creates artificial spike at zero)")
    
    print("\n  Recommendation: Use parametric distributions (Exponential, Gamma,")
    print("  LogNormal, Weibull) for naturally positive quantities!")
    print()


def example_10_qmc_with_new_distributions():
    """Test QMC methods with new distributions."""
    print("=" * 70)
    print("Example 9: QMC Sampling with New Distributions")
    print("=" * 70)
    
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('exponential', ExponentialDistribution(rate=0.5))
    sampler.add_parameter('gamma', GammaDistribution(shape=2, scale=2))
    sampler.add_parameter('lognormal', LogNormalDistribution(mu=0, sigma=0.5))
    sampler.add_parameter('weibull', WeibullDistribution(shape=1.5, scale=2))
    sampler.add_parameter('triangular', TriangularDistribution(1, 3, 6))
    
    print("\nComparing random vs Sobol sampling with new distributions:")
    
    for method in ['random', 'sobol']:
        samples = sampler.generate(n=100, return_type='dataframe', method=method)
        
        print(f"\n{method.upper()} sampling:")
        print(f"  Exponential mean: {samples['exponential'].mean():.2f} (expected ~2.0)")
        print(f"  Gamma mean: {samples['gamma'].mean():.2f} (expected ~4.0)")
        print(f"  LogNormal mean: {samples['lognormal'].mean():.2f} (expected ~1.13)")
        print(f"  Weibull mean: {samples['weibull'].mean():.2f}")
        print(f"  Triangular mean: {samples['triangular'].mean():.2f} (expected ~3.3)")
    print()


if __name__ == '__main__':
    example_1_common_distributions()
    example_2_custom_continuous_from_data()
    example_3_custom_continuous_from_file()
    example_4_custom_continuous_from_function()
    example_5_negative_handling()
    example_6_custom_discrete_from_probabilities()
    example_7_custom_discrete_from_data()
    example_8_integration_with_sampler()
    example_9_when_to_use_custom_vs_parametric()
    example_10_qmc_with_new_distributions()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nSummary:")
    print("  - 5 new common distributions added")
    print("  - Custom continuous distributions from data/files/functions")
    print("  - Custom discrete distributions from probabilities/data/functions")
    print("  - Negative value handling (truncate/shift/allow)")
    print("  - Full integration with QMC sampling")
