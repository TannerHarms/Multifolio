"""
Comparison of different methods for CustomContinuousDistribution.

This demonstrates:
- KDE: Smooth but can extend beyond data
- Empirical CDF: Stays within data range
- Spline: Smooth and bounded
- Histogram: Fast, discretized

Shows when to use each method and their trade-offs.
"""

import sys
from pathlib import Path
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from multifolio.core.sampling.distributions import CustomContinuousDistribution


def example_1_method_comparison():
    """Compare all four methods on the same data."""
    print("=" * 70)
    print("Example 1: Method Comparison on Gamma-Distributed Data")
    print("=" * 70)
    
    # Generate positive-only data (like processing times)
    np.random.seed(42)
    data = np.random.gamma(shape=2, scale=15, size=500)
    
    print(f"\nOriginal data (n={len(data)}):")
    print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    methods = ['kde', 'empirical_cdf', 'spline', 'histogram']
    
    print("\nGenerating 1000 samples with each method:\n")
    
    for method in methods:
        dist = CustomContinuousDistribution(data=data, method=method)
        samples = dist.sample(size=1000)
        
        n_negative = (samples < 0).sum()
        n_below_min = (samples < data.min()).sum()
        n_above_max = (samples > data.max()).sum()
        
        print(f"{method.upper():15s}:")
        print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")
        print(f"  Mean: {samples.mean():.2f}, Std: {samples.std():.2f}")
        print(f"  Negative samples: {n_negative}/1000")
        print(f"  Below original min: {n_below_min}/1000")
        print(f"  Above original max: {n_above_max}/1000")
        print()


def example_2_bounded_data_problem():
    """Show the boundary problem with KDE vs solutions."""
    print("=" * 70)
    print("Example 2: Boundary Problem - Processing Times")
    print("=" * 70)
    
    # Processing times from a factory floor (always positive)
    np.random.seed(42)
    processing_times = np.random.exponential(scale=5, size=200)
    
    print(f"\nProcessing times from 200 jobs:")
    print(f"  Min: {processing_times.min():.2f} minutes")
    print(f"  Max: {processing_times.max():.2f} minutes")
    print(f"  Mean: {processing_times.mean():.2f} minutes")
    
    print("\n" + "-" * 70)
    print("METHOD 1: KDE (default) - Can produce negative times!")
    print("-" * 70)
    
    kde_dist = CustomContinuousDistribution(data=processing_times, method='kde')
    kde_samples = kde_dist.sample(size=1000)
    n_negative = (kde_samples < 0).sum()
    
    print(f"  Negative times: {n_negative}/1000 ({n_negative/10:.1f}%)")
    print(f"  Min sample: {kde_samples.min():.2f} minutes")
    print(f"  Problem: Can't have negative processing times!")
    
    print("\n" + "-" * 70)
    print("METHOD 2: Empirical CDF - Stays within observed range")
    print("-" * 70)
    
    ecdf_dist = CustomContinuousDistribution(data=processing_times, method='empirical_cdf')
    ecdf_samples = ecdf_dist.sample(size=1000)
    
    print(f"  Negative times: {(ecdf_samples < 0).sum()}/1000")
    print(f"  Min sample: {ecdf_samples.min():.2f} minutes")
    print(f"  Max sample: {ecdf_samples.max():.2f} minutes")
    print(f"  Guaranteed: {ecdf_samples.min():.2f} >= {processing_times.min():.2f}")
    
    print("\n" + "-" * 70)
    print("METHOD 3: Spline - Smooth and bounded")
    print("-" * 70)
    
    spline_dist = CustomContinuousDistribution(data=processing_times, method='spline')
    spline_samples = spline_dist.sample(size=1000)
    
    print(f"  Negative times: {(spline_samples < 0).sum()}/1000")
    print(f"  Min sample: {spline_samples.min():.2f} minutes")
    print(f"  Max sample: {spline_samples.max():.2f} minutes")
    print(f"  Smooth and stays within approximate bounds")
    
    print("\n" + "-" * 70)
    print("METHOD 4: Histogram - Fast approximation")
    print("-" * 70)
    
    hist_dist = CustomContinuousDistribution(data=processing_times, method='histogram')
    hist_samples = hist_dist.sample(size=1000)
    
    print(f"  Negative times: {(hist_samples < 0).sum()}/1000")
    print(f"  Min sample: {hist_samples.min():.2f} minutes")
    print(f"  Max sample: {hist_samples.max():.2f} minutes")
    print(f"  Fast and bounded, but slightly discretized")
    print()


def example_3_complex_distribution():
    """Show when KDE excels - complex, multimodal distribution."""
    print("=" * 70)
    print("Example 3: Complex Distribution - When KDE Shines")
    print("=" * 70)
    
    # Bimodal distribution representing two different process conditions
    np.random.seed(42)
    mode1 = np.random.normal(10, 2, 300)   # Standard process
    mode2 = np.random.normal(25, 3, 200)   # Alternative process
    complex_data = np.concatenate([mode1, mode2])
    
    print("\nBimodal process data (two operating modes):")
    print(f"  Range: [{complex_data.min():.2f}, {complex_data.max():.2f}]")
    print(f"  Overall mean: {complex_data.mean():.2f}")
    
    print("\nComparing methods on complex shape:\n")
    
    methods = ['kde', 'empirical_cdf', 'spline', 'histogram']
    
    for method in methods:
        dist = CustomContinuousDistribution(data=complex_data, method=method)
        samples = dist.sample(size=1000)
        
        # Check if bimodal structure is preserved
        # Count samples in each mode
        in_mode1 = ((samples >= 5) & (samples <= 15)).sum()
        in_mode2 = ((samples >= 20) & (samples <= 30)).sum()
        in_gap = ((samples > 15) & (samples < 20)).sum()
        
        print(f"{method.upper():15s}:")
        print(f"  Mode 1 (5-15): {in_mode1}/1000")
        print(f"  Gap (15-20): {in_gap}/1000")
        print(f"  Mode 2 (20-30): {in_mode2}/1000")
        print(f"  Mean: {samples.mean():.2f}")
        print()
    
    print("Note: KDE and Spline best preserve the bimodal structure with smooth gap.")
    print("Empirical CDF and Histogram have more samples in the gap (from resampling).")
    print()


def example_4_performance_comparison():
    """Compare speed of different methods."""
    print("=" * 70)
    print("Example 4: Performance Comparison")
    print("=" * 70)
    
    import time
    
    np.random.seed(42)
    data = np.random.gamma(2, 15, size=1000)
    
    methods = ['kde', 'empirical_cdf', 'spline', 'histogram']
    n_samples = 10000
    
    print(f"\nTiming {n_samples} samples from each method:\n")
    
    for method in methods:
        # Setup
        start_setup = time.time()
        dist = CustomContinuousDistribution(data=data, method=method)
        setup_time = time.time() - start_setup
        
        # Sampling
        start_sample = time.time()
        samples = dist.sample(size=n_samples)
        sample_time = time.time() - start_sample
        
        print(f"{method.upper():15s}:")
        print(f"  Setup time: {setup_time*1000:.2f} ms")
        print(f"  Sample time: {sample_time*1000:.2f} ms")
        print(f"  Total: {(setup_time + sample_time)*1000:.2f} ms")
        print()


def example_5_method_selection_guide():
    """Comprehensive guide for choosing the right method."""
    print("=" * 70)
    print("Example 5: Method Selection Guide")
    print("=" * 70)
    
    guide = """
    
    METHOD SELECTION GUIDE
    =====================
    
    1. KDE (Kernel Density Estimation)
       --------------------------------
       Use when:
         - Shape is unknown and complex
         - Data can naturally extend beyond observed range
         - Need smooth, continuous distribution
         - Negatives are valid (temperatures, errors, deltas)
       
       Avoid when:
         - Data must be bounded (times, concentrations, counts)
         - Small sample size (< 100 points)
       
       Properties:
         - Most flexible, captures complex shapes
         - Can produce values outside data range
         - Smoothest results
         - Moderate speed
       
       Example use cases:
         - Measurement errors (can be ±)
         - Temperature variations
         - Financial returns
         - Unknown/weird distributions
    
    
    2. EMPIRICAL_CDF (Empirical Cumulative Distribution)
       --------------------------------------------------
       Use when:
         - Data must stay within observed bounds
         - Processing times, concentrations, ages
         - Don't want smoothing artifacts
         - Have representative sample of full range
       
       Avoid when:
         - Need to extrapolate beyond data
         - Very small sample size (< 50 points)
       
       Properties:
         - Guaranteed within [data.min(), data.max()]
         - No negative values from positive data
         - Fast sampling
         - Less smooth than KDE
       
       Example use cases:
         - Processing/reaction times
         - Concentrations (must be >= 0)
         - Prices (must be > 0)
         - Physical measurements with natural bounds
    
    
    3. SPLINE (Cubic Spline Interpolation)
       ------------------------------------
       Use when:
         - Want smooth distribution
         - Need bounded samples
         - Have enough data (> 100 points)
         - Balance between KDE and empirical CDF
       
       Avoid when:
         - Very small sample size (< 50 points)
         - Data is highly irregular
       
       Properties:
         - Smooth like KDE
         - Bounded like empirical CDF
         - Good balance of properties
         - Moderate speed
       
       Example use cases:
         - Most bounded quantities
         - When KDE gives too many negatives
         - When empirical CDF is too discrete
    
    
    4. HISTOGRAM (Binned Sampling)
       ----------------------------
       Use when:
         - Need fast approximation
         - Large number of samples needed
         - Shape is relatively simple
         - Exact smoothness not critical
       
       Avoid when:
         - Need very accurate tail behavior
         - Small sample size (< 100 points)
       
       Properties:
         - Fastest sampling
         - Bounded within data range
         - Slightly discretized (bin edges visible)
         - Simple, robust
       
       Example use cases:
         - Quick prototyping
         - Large-scale sampling (millions)
         - Rough approximations
         - When speed matters most
    
    
    QUICK DECISION TREE:
    ===================
    
    Start here: What kind of data?
    
    - Naturally BOUNDED (times, concentrations, prices)?
      - Need smoothest possible? -> SPLINE
      - Need guaranteed bounds? -> EMPIRICAL_CDF
      - Need speed? -> HISTOGRAM
    
    - Can be UNBOUNDED (errors, temperatures, returns)?
      - Complex/multimodal shape? -> KDE
      - Simple unimodal? -> SPLINE or KDE
      - Need speed? -> HISTOGRAM
    
    
    DEFAULT RECOMMENDATIONS:
    =======================
    
    - BOUNDED POSITIVE DATA (times, concentrations):
      -> method='empirical_cdf'  (safest)
      -> method='spline'  (if want smoothness)
    
    - UNBOUNDED DATA (errors, temperatures):
      -> method='kde'  (most flexible)
    
    - NEED SPEED:
      -> method='histogram'
    
    - COMPLEX SHAPES:
      -> method='kde'  (best at capturing complexity)
    
    - NOT SURE?
      -> method='empirical_cdf'  (safest, no surprises)
    """
    
    print(guide)


def example_6_practical_examples():
    """Show practical examples for common use cases."""
    print("=" * 70)
    print("Example 6: Practical Examples")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Use case 1: Chemical reaction times (must be positive)
    print("\nUse Case 1: Chemical Reaction Times")
    print("-" * 40)
    reaction_times = np.random.gamma(3, 5, size=150)  # In seconds
    dist = CustomContinuousDistribution(data=reaction_times, method='empirical_cdf')
    samples = dist.sample(size=5)
    print(f"  Method: empirical_cdf (guaranteed positive)")
    print(f"  Samples: {samples}")
    print(f"  All positive: {(samples >= 0).all()}")
    
    # Use case 2: Temperature measurements (can be negative)
    print("\nUse Case 2: Temperature Measurements (°C)")
    print("-" * 40)
    temperatures = np.random.normal(20, 5, size=200)
    dist = CustomContinuousDistribution(data=temperatures, method='kde')
    samples = dist.sample(size=5)
    print(f"  Method: kde (allows negatives)")
    print(f"  Samples: {samples}")
    print(f"  Can be negative: negatives are valid temperatures")
    
    # Use case 3: Protein concentration (bounded, smooth needed)
    print("\nUse Case 3: Protein Concentrations (mg/mL)")
    print("-" * 40)
    concentrations = np.random.lognormal(1, 0.5, size=300)
    dist = CustomContinuousDistribution(data=concentrations, method='spline')
    samples = dist.sample(size=5)
    print(f"  Method: spline (smooth + bounded)")
    print(f"  Samples: {samples}")
    print(f"  All positive: {(samples >= 0).all()}")
    
    # Use case 4: Quick monte carlo (need speed)
    print("\nUse Case 4: Monte Carlo Simulation (10M samples)")
    print("-" * 40)
    mc_data = np.random.exponential(2, size=1000)
    dist = CustomContinuousDistribution(data=mc_data, method='histogram')
    print(f"  Method: histogram (fastest)")
    import time
    start = time.time()
    samples = dist.sample(size=10000000)
    elapsed = time.time() - start
    print(f"  Generated 10M samples in {elapsed:.2f} seconds")
    print(f"  Rate: {10000000/elapsed/1e6:.2f} million samples/sec")
    print()


if __name__ == '__main__':
    example_1_method_comparison()
    example_2_bounded_data_problem()
    example_3_complex_distribution()
    example_4_performance_comparison()
    example_5_method_selection_guide()
    example_6_practical_examples()
    
    print("=" * 70)
    print("Summary: Choose the right method for your data!")
    print("=" * 70)
    print("\nQuick picks:")
    print("  - Bounded data -> empirical_cdf or spline")
    print("  - Unbounded data -> kde")
    print("  - Need speed -> histogram")
    print("  - Complex shapes -> kde")
    print("  - Not sure -> empirical_cdf (safest)")
