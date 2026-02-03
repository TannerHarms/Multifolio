"""
Example: Using Multifolio Sampling Module

This script demonstrates how to use the sampling module to generate
parameter samples for experimental design.
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path for import
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    BetaDistribution,
    ConstantDistribution,
    PoissonDistribution,
    UniformDiscreteDistribution,
)


def example_basic_usage():
    """Basic usage of distributions."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Distribution Usage")
    print("=" * 60)
    
    # Uniform distribution
    uniform = UniformDistribution(low=0, high=10, random_seed=42)
    print(f"\n{uniform}")
    print(f"Samples: {uniform.sample(5)}")
    
    # Normal distribution
    normal = NormalDistribution(mean=100, std=15, random_seed=42)
    print(f"\n{normal}")
    print(f"Samples: {normal.sample(5)}")
    
    # Truncated normal
    truncated = TruncatedNormalDistribution(
        mean=0, std=1, low=-1.5, high=1.5, random_seed=42
    )
    print(f"\n{truncated}")
    print(f"Samples: {truncated.sample(5)}")
    
    # Beta distribution
    beta = BetaDistribution(alpha=2, beta=5, random_seed=42)
    print(f"\n{beta}")
    print(f"Samples: {beta.sample(5)}")
    
    # Beta distribution with custom range
    beta_scaled = BetaDistribution(alpha=3, beta=3, low=50, high=150, random_seed=42)
    print(f"\n{beta_scaled}")
    print(f"Samples: {beta_scaled.sample(5)}")
    
    # Constant distribution
    constant = ConstantDistribution(value=42)
    print(f"\n{constant}")
    print(f"Samples: {constant.sample(5)}")
    
    # Poisson distribution
    poisson = PoissonDistribution(lam=3.5, random_seed=42)
    print(f"\n{poisson}")
    print(f"Samples: {poisson.sample(5)}")
    
    # Discrete uniform
    discrete = UniformDiscreteDistribution(low=1, high=6, random_seed=42)
    print(f"\n{discrete}")
    print(f"Samples: {discrete.sample(5)}")


def example_multi_parameter_sampling():
    """Using Sampler for multi-parameter experimental design."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Parameter Sampling")
    print("=" * 60)
    
    # Create sampler
    sampler = Sampler(random_seed=42)
    
    # Add parameters for a hypothetical experiment
    sampler.add_parameter('temperature', UniformDistribution(low=20, high=100))
    sampler.add_parameter('pressure', NormalDistribution(mean=1.0, std=0.1))
    sampler.add_parameter('catalyst_amount', TruncatedNormalDistribution(
        mean=5, std=1, low=3, high=7
    ))
    sampler.add_parameter('reaction_time', UniformDiscreteDistribution(low=60, high=180))
    sampler.add_parameter('stirring_speed', ConstantDistribution(value=500))
    
    print(f"\n{sampler}")
    
    # Generate samples as dictionary
    print("\nGenerating 5 samples (as dict):")
    samples_dict = sampler.generate(n=5, return_type='dict')
    for param, values in samples_dict.items():
        print(f"  {param}: {values}")
    
    # Generate samples as DataFrame
    print("\nGenerating 10 samples (as DataFrame):")
    df = sampler.generate(n=10, return_type='dataframe')
    print(df)
    
    # Generate samples as array
    print("\nGenerating samples (as array):")
    arr = sampler.generate(n=3, return_type='array')
    print(arr)
    print(f"Shape: {arr.shape}")


def example_experimental_design():
    """Real-world experimental design example."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 3: Experimental Design for Material Science")
    print("=" * 60)
    
    # Design experiments for testing a new alloy
    sampler = Sampler(random_seed=123)
    
    # Composition parameters
    sampler.add_parameter('iron_percent', UniformDistribution(low=60, high=80))
    sampler.add_parameter('carbon_percent', UniformDistribution(low=0.1, high=2.0))
    sampler.add_parameter('chromium_percent', UniformDistribution(low=10, high=20))
    
    # Processing parameters
    sampler.add_parameter('heating_temp', NormalDistribution(mean=1200, std=50))
    sampler.add_parameter('cooling_rate', UniformDistribution(low=5, high=50))
    sampler.add_parameter('hold_time', UniformDiscreteDistribution(low=30, high=120))
    
    # Environmental parameters
    sampler.add_parameter('atmosphere', UniformDiscreteDistribution(low=1, high=3))
    # 1=nitrogen, 2=argon, 3=vacuum
    
    # Generate experimental matrix
    experiments = sampler.generate(n=20, return_type='dataframe')
    
    print("\nGenerated 20 experimental conditions:")
    print(experiments.head(10))
    
    print(f"\nTotal experiments: {len(experiments)}")
    print("\nSummary statistics:")
    print(experiments.describe())
    
    # Save to CSV (would do this in practice)
    # experiments.to_csv('experimental_design.csv', index=False)
    print("\n(In practice, save with: experiments.to_csv('design.csv', index=False))")


def example_reproducibility():
    """Demonstrate reproducibility with random seeds."""
    print("\n\n" + "=" * 60)
    print("EXAMPLE 4: Reproducibility with Random Seeds")
    print("=" * 60)
    
    # Create two samplers with same seed
    sampler1 = Sampler(random_seed=42)
    sampler1.add_parameter('x', UniformDistribution(0, 1))
    sampler1.add_parameter('y', NormalDistribution(0, 1))
    
    sampler2 = Sampler(random_seed=42)
    sampler2.add_parameter('x', UniformDistribution(0, 1))
    sampler2.add_parameter('y', NormalDistribution(0, 1))
    
    # Generate samples
    samples1 = sampler1.generate(n=5, return_type='dict')
    samples2 = sampler2.generate(n=5, return_type='dict')
    
    print("\nSampler 1 (seed=42):")
    print(f"  x: {samples1['x']}")
    print(f"  y: {samples1['y']}")
    
    print("\nSampler 2 (seed=42):")
    print(f"  x: {samples2['x']}")
    print(f"  y: {samples2['y']}")
    
    print("\nSamples are identical:", np.array_equal(samples1['x'], samples2['x']))
    
    # Different seed gives different results
    sampler3 = Sampler(random_seed=123)
    sampler3.add_parameter('x', UniformDistribution(0, 1))
    sampler3.add_parameter('y', NormalDistribution(0, 1))
    
    samples3 = sampler3.generate(n=5, return_type='dict')
    
    print("\nSampler 3 (seed=123):")
    print(f"  x: {samples3['x']}")
    print(f"  y: {samples3['y']}")
    
    print("\nDifferent from Sampler 1:", not np.array_equal(samples1['x'], samples3['x']))


def example_qmc_methods():
    """Demonstrate Quasi-Monte Carlo sampling methods."""
    print("\n" + "=" * 70)
    print("Example 5: Quasi-Monte Carlo (QMC) Sampling Methods")
    print("=" * 70)
    
    print("\nQMC methods provide better space coverage than random sampling.")
    print("This is valuable for experimental design and parameter exploration.\n")
    
    # Create a 2-parameter sampler
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('temperature', UniformDistribution(20, 80))  # °C
    sampler.add_parameter('pressure', UniformDistribution(1, 5))       # bar
    
    n_samples = 20
    
    # Compare different sampling methods
    print(f"Generating {n_samples} experimental conditions with different methods:\n")
    
    for method in ['random', 'sobol', 'halton', 'lhs']:
        samples = sampler.generate(n=n_samples, return_type='dataframe', method=method)
        
        print(f"{method.upper()} sampling:")
        print(f"  Temperature range: [{samples['temperature'].min():.1f}, {samples['temperature'].max():.1f}]°C")
        print(f"  Pressure range: [{samples['pressure'].min():.2f}, {samples['pressure'].max():.2f}] bar")
        
        # Show first 3 samples
        print("  First 3 samples:")
        for i in range(min(3, len(samples))):
            print(f"    {i+1}. T={samples.iloc[i]['temperature']:.1f}°C, P={samples.iloc[i]['pressure']:.2f} bar")
        print()
    
    print("QMC Methods Guide:")
    print("  - random: Standard Monte Carlo (default)")
    print("  - sobol: Best overall space-filling (recommended for most cases)")
    print("  - halton: Good for low-dimensional problems")
    print("  - lhs: Latin Hypercube - ensures stratified coverage")


if __name__ == '__main__':
    example_basic_usage()
    example_multi_parameter_sampling()
    example_experimental_design()
    example_reproducibility()
    example_qmc_methods()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
