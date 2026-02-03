"""
Example demonstrating Quasi-Monte Carlo (QMC) sampling methods.

QMC methods provide better space-filling properties than random sampling,
which is especially valuable for:
- Experimental design with limited samples
- Parameter space exploration
- Sensitivity analysis
- Optimization with expensive function evaluations
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.insert(0, str(backend_path))

import numpy as np
# import matplotlib.pyplot as plt  # Optional - comment out if not installed
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution,
    NormalDistribution,
    BetaDistribution,
)


def example_1_visual_comparison():
    """Compare random vs QMC sampling visually."""
    print("=" * 70)
    print("Example 1: Visual Comparison of Sampling Methods")
    print("=" * 70)
    
    n_samples = 100
    
    # Create samplers with 2D uniform distribution
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('x', UniformDistribution(0, 1))
    sampler.add_parameter('y', UniformDistribution(0, 1))
    
    # Generate samples with different methods
    methods = ['random', 'sobol', 'halton', 'lhs']
    
    print(f"\nGenerated {n_samples} samples with each method.")
    print("(Install matplotlib to see visual comparison)\n")
    
    for method in methods:
        samples = sampler.generate(n=n_samples, return_type='dataframe', method=method)
        print(f"{method.upper():8s}: x range [{samples['x'].min():.3f}, {samples['x'].max():.3f}], "
              f"y range [{samples['y'].min():.3f}, {samples['y'].max():.3f}]")
    
    # Uncomment if matplotlib is installed:
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # axes = axes.flatten()
    # 
    # for idx, method in enumerate(methods):
    #     samples = sampler.generate(n=n_samples, return_type='dataframe', method=method)
    #     
    #     ax = axes[idx]
    #     ax.scatter(samples['x'], samples['y'], alpha=0.6, s=30)
    #     ax.set_xlim(0, 1)
    #     ax.set_ylim(0, 1)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_title(f'{method.upper()} Sampling (n={n_samples})')
    #     ax.grid(True, alpha=0.3)
    #     ax.set_aspect('equal')
    # 
    # plt.tight_layout()
    # plt.savefig('qmc_comparison.png', dpi=150)
    # print(f"\nSaved visualization to qmc_comparison.png")
    
    print("Notice how QMC methods (Sobol, Halton, LHS) fill space more evenly than random!")
    print()


def example_2_coverage_analysis():
    """Quantify space coverage of different methods."""
    print("=" * 70)
    print("Example 2: Space Coverage Analysis")
    print("=" * 70)
    
    n_samples = 100
    n_bins = 10
    
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('x', UniformDistribution(0, 1))
    sampler.add_parameter('y', UniformDistribution(0, 1))
    
    def count_filled_bins(samples):
        """Count number of grid cells that contain at least one sample."""
        x_bins = np.digitize(samples[:, 0], np.linspace(0, 1, n_bins + 1))
        y_bins = np.digitize(samples[:, 1], np.linspace(0, 1, n_bins + 1))
        filled = set(zip(x_bins, y_bins))
        return len(filled)
    
    print(f"\nDividing [0,1]×[0,1] space into {n_bins}×{n_bins} = {n_bins**2} grid cells")
    print(f"Generating {n_samples} samples with each method:\n")
    
    for method in ['random', 'sobol', 'halton', 'lhs']:
        samples = sampler.generate(n=n_samples, return_type='array', method=method)
        coverage = count_filled_bins(samples)
        coverage_pct = (coverage / (n_bins ** 2)) * 100
        
        print(f"  {method.upper():8s}: {coverage:3d}/{n_bins**2} cells filled ({coverage_pct:.1f}% coverage)")
    
    print("\nQMC methods typically achieve better coverage with fewer samples!")
    print()


def example_3_experimental_design():
    """Use QMC for experimental design with multiple parameter types."""
    print("=" * 70)
    print("Example 3: Multi-Parameter Experimental Design with QMC")
    print("=" * 70)
    
    # Design chemical reactor experiments
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('temperature', UniformDistribution(20, 80))  # °C
    sampler.add_parameter('pressure', UniformDistribution(1, 5))       # bar
    sampler.add_parameter('catalyst_conc', BetaDistribution(2, 5, low=0.1, high=2.0))  # mol/L
    sampler.add_parameter('reaction_time', UniformDistribution(10, 120))  # minutes
    
    # Generate experimental plan with Sobol sequence
    n_experiments = 20
    experiments = sampler.generate(n=n_experiments, return_type='dataframe', method='sobol')
    
    print(f"\nGenerated {n_experiments} experimental conditions using Sobol sequence:")
    print("\nFirst 10 experiments:")
    print(experiments.head(10).to_string(index=True, float_format=lambda x: f'{x:.2f}'))
    
    print("\n\nParameter ranges covered:")
    for col in experiments.columns:
        print(f"  {col:20s}: [{experiments[col].min():6.2f}, {experiments[col].max():6.2f}]")
    
    print("\nSobol sampling ensures good coverage of the parameter space with fewer experiments!")
    print()


def example_4_convergence_comparison():
    """Compare convergence rates of different methods."""
    print("=" * 70)
    print("Example 4: Convergence Rate Comparison")
    print("=" * 70)
    
    # Simple integration problem: estimate integral of sin(x)*cos(y) over [0,π]×[0,π]
    # True value = 0
    
    def test_function(x, y):
        return np.sin(x) * np.cos(y)
    
    sample_sizes = [10, 25, 50, 100, 250, 500]
    
    print("\nEstimating integral of sin(x)*cos(y) over [0,π]×[0,π]")
    print("True value = 0.0\n")
    print(f"{'N':>6s}  {'Random':>10s}  {'Sobol':>10s}  {'Halton':>10s}  {'LHS':>10s}")
    print("-" * 60)
    
    for n in sample_sizes:
        estimates = {}
        
        for method in ['random', 'sobol', 'halton', 'lhs']:
            sampler = Sampler(random_seed=42)
            sampler.add_parameter('x', UniformDistribution(0, np.pi))
            sampler.add_parameter('y', UniformDistribution(0, np.pi))
            
            samples = sampler.generate(n=n, return_type='dataframe', method=method)
            values = test_function(samples['x'].values, samples['y'].values)
            estimate = np.mean(values) * (np.pi ** 2)  # Scale by area
            estimates[method] = estimate
        
        print(f"{n:6d}  {estimates['random']:10.6f}  {estimates['sobol']:10.6f}  "
              f"{estimates['halton']:10.6f}  {estimates['lhs']:10.6f}")
    
    print("\nQMC methods typically converge faster (error ~ O(1/N)) vs random (error ~ O(1/√N))!")
    print()


def example_5_high_dimensional():
    """Demonstrate QMC in higher dimensions."""
    print("=" * 70)
    print("Example 5: High-Dimensional Sampling")
    print("=" * 70)
    
    # 10-dimensional optimization problem
    n_dimensions = 10
    sampler = Sampler(random_seed=42)
    
    for i in range(n_dimensions):
        sampler.add_parameter(f'x{i+1}', UniformDistribution(-5, 5))
    
    n_samples = 100
    
    print(f"\nGenerating {n_samples} samples in {n_dimensions}-dimensional space")
    print("Comparing random vs Sobol sampling:\n")
    
    for method in ['random', 'sobol']:
        samples = sampler.generate(n=n_samples, return_type='array', method=method)
        
        # Check coverage by looking at min/max in each dimension
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        ranges = maxs - mins
        avg_range = ranges.mean()
        
        print(f"{method.upper()} sampling:")
        print(f"  Average range covered: {avg_range:.2f} / 10.0 ({avg_range/10*100:.1f}%)")
        print(f"  Dimensions reaching [-4, 4]: {np.sum((mins < -3.5) & (maxs > 3.5))}/{n_dimensions}")
        print()
    
    print("Sobol maintains better coverage even in high dimensions!")
    print()


def example_6_method_selection_guide():
    """Guide for selecting appropriate sampling method."""
    print("=" * 70)
    print("Example 6: When to Use Each Method")
    print("=" * 70)
    
    guide = """
    RANDOM SAMPLING (method='random')
    - When: Need truly independent samples
    - Use cases: Monte Carlo simulation, bootstrapping, statistical inference
    - Pros: Simple, independent samples, good for statistical testing
    - Cons: Poor space coverage, high variance
    - Convergence: O(1/√N)
    
    SOBOL SEQUENCE (method='sobol')
    - When: Need best overall space-filling properties
    - Use cases: Experimental design, global optimization, sensitivity analysis
    - Pros: Excellent coverage, fast convergence, works well in high dimensions
    - Cons: Samples are deterministic (less suitable for statistical inference)
    - Convergence: O((log N)^d / N)
    - Best for: d ≤ 20 dimensions
    
    HALTON SEQUENCE (method='halton')
    - When: Need good coverage in low-moderate dimensions
    - Use cases: Computer graphics, numerical integration
    - Pros: Simple to generate, good low-discrepancy properties
    - Cons: Quality degrades in high dimensions, correlation between dimensions
    - Convergence: O((log N)^d / N)
    - Best for: d ≤ 10 dimensions
    
    LATIN HYPERCUBE SAMPLING (method='lhs')
    - When: Need stratified sampling with guaranteed coverage
    - Use cases: Risk analysis, robust design, response surface modeling
    - Pros: Ensures samples in all regions, works well for few samples
    - Cons: Not as uniformly distributed as Sobol/Halton
    - Convergence: Better than random, not as good as Sobol
    - Best for: Small to moderate sample sizes (N < 100)
    
    RECOMMENDATIONS:
    - Default choice: Sobol (best all-around performance)
    - Limited samples (N < 50): LHS (ensures coverage)
    - High dimensions (d > 20): LHS or stratified random
    - Need independence: Random
    """
    
    print(guide)


if __name__ == '__main__':
    # Run all examples
    example_1_visual_comparison()
    example_2_coverage_analysis()
    example_3_experimental_design()
    example_4_convergence_comparison()
    example_5_high_dimensional()
    example_6_method_selection_guide()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
