"""
Examples demonstrating correlated sampling with Multifolio.

This module shows how to use the correlation features to model dependencies
between parameters while preserving their marginal distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from multifolio.core.sampling import Sampler
from multifolio.core.distributions import NormalDistribution, CustomContinuousDistribution


def example_1_basic_pairwise_correlation():
    """
    Example 1: Basic Pairwise Correlation
    
    Shows how to add a simple pairwise correlation between two parameters.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Pairwise Correlation")
    print("="*70)
    
    # Create sampler with two independent normal distributions
    sampler = Sampler(n_samples=2000, seed=42)
    sampler.add_parameter("temperature", NormalDistribution(mean=25, std=5))
    sampler.add_parameter("pressure", NormalDistribution(mean=100, std=10))
    
    # Generate independent samples first
    independent = sampler.generate(as_dataframe=True)
    corr_before = spearmanr(independent["temperature"], independent["pressure"])[0]
    print(f"Correlation before: {corr_before:.3f}")
    
    # Add positive correlation (temperature and pressure often correlate)
    sampler.set_correlation("temperature", "pressure", correlation=0.7)
    
    # Generate correlated samples
    correlated = sampler.generate(as_dataframe=True)
    corr_after = spearmanr(correlated["temperature"], correlated["pressure"])[0]
    print(f"Correlation after: {corr_after:.3f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(independent["temperature"], independent["pressure"], 
                alpha=0.5, s=20)
    ax1.set_xlabel("Temperature (C)")
    ax1.set_ylabel("Pressure (kPa)")
    ax1.set_title(f"Independent (ρ = {corr_before:.2f})")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(correlated["temperature"], correlated["pressure"], 
                alpha=0.5, s=20, color='orange')
    ax2.set_xlabel("Temperature (C)")
    ax2.set_ylabel("Pressure (kPa)")
    ax2.set_title(f"Correlated (ρ = {corr_after:.2f})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("example1_pairwise_correlation.png", dpi=150)
    print("Saved visualization to example1_pairwise_correlation.png")
    plt.close()


def example_2_correlation_matrix():
    """
    Example 2: Multi-Parameter Correlation Matrix
    
    Shows how to specify correlations between multiple parameters using
    a correlation matrix.
    """
    print("\n" + "="*70)
    print("Example 2: Multi-Parameter Correlation Matrix")
    print("="*70)
    
    # Create sampler with three parameters
    sampler = Sampler(n_samples=2000, seed=42)
    sampler.add_parameter("temperature", NormalDistribution(mean=25, std=5))
    sampler.add_parameter("pressure", NormalDistribution(mean=100, std=10))
    sampler.add_parameter("humidity", NormalDistribution(mean=60, std=15))
    
    # Define correlation matrix
    # temperature-pressure: 0.7 (positive)
    # temperature-humidity: -0.5 (negative - hotter = drier)
    # pressure-humidity: 0.3 (slight positive)
    correlation_matrix = np.array([
        [1.0,  0.7, -0.5],  # temperature
        [0.7,  1.0,  0.3],  # pressure
        [-0.5, 0.3,  1.0]   # humidity
    ])
    
    sampler.set_correlation_matrix(correlation_matrix)
    
    # Generate samples
    samples = sampler.generate(as_dataframe=True)
    
    # Compute correlation matrix of samples
    actual_corr = np.corrcoef(samples.T)
    
    print("\nTarget correlation matrix:")
    print(correlation_matrix)
    print("\nActual correlation matrix:")
    print(actual_corr)
    
    # Visualize correlation matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Target correlation
    im1 = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['Temp', 'Press', 'Humid'])
    ax1.set_yticklabels(['Temp', 'Press', 'Humid'])
    ax1.set_title("Target Correlations")
    plt.colorbar(im1, ax=ax1)
    
    # Actual correlation
    im2 = ax2.imshow(actual_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Temp', 'Press', 'Humid'])
    ax2.set_yticklabels(['Temp', 'Press', 'Humid'])
    ax2.set_title("Actual Correlations")
    plt.colorbar(im2, ax=ax2)
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                    ha='center', va='center', color='black')
            ax2.text(j, i, f'{actual_corr[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig("example2_correlation_matrix.png", dpi=150)
    print("\nSaved visualization to example2_correlation_matrix.png")
    plt.close()


def example_3_custom_distributions():
    """
    Example 3: Correlations with Custom Distributions
    
    Shows that correlations work with any distribution type, preserving
    the marginal distributions while inducing correlation.
    """
    print("\n" + "="*70)
    print("Example 3: Correlations with Custom Distributions")
    print("="*70)
    
    # Create custom distributions from data
    np.random.seed(42)
    
    # Bimodal distribution for parameter A
    data_a = np.concatenate([
        np.random.normal(10, 2, 500),
        np.random.normal(20, 2, 500)
    ])
    dist_a = CustomContinuousDistribution(data_a, name="A")
    
    # Exponential-like distribution for parameter B
    data_b = np.random.gamma(shape=2, scale=5, size=1000)
    dist_b = CustomContinuousDistribution(data_b, name="B")
    
    # Create sampler
    sampler = Sampler(n_samples=2000, seed=42)
    sampler.add_parameter("A", dist_a)
    sampler.add_parameter("B", dist_b)
    
    # Generate without correlation
    independent = sampler.generate(as_dataframe=True)
    
    # Add strong positive correlation
    sampler.set_correlation("A", "B", correlation=0.8)
    correlated = sampler.generate(as_dataframe=True)
    
    # Compute correlations
    corr_before = spearmanr(independent["A"], independent["B"])[0]
    corr_after = spearmanr(correlated["A"], correlated["B"])[0]
    
    print(f"Correlation before: {corr_before:.3f}")
    print(f"Correlation after: {corr_after:.3f}")
    
    # Visualize
    fig = plt.figure(figsize=(14, 5))
    
    # Scatter plots
    ax1 = plt.subplot(1, 3, 1)
    ax1.scatter(independent["A"], independent["B"], alpha=0.5, s=20)
    ax1.set_xlabel("Parameter A")
    ax1.set_ylabel("Parameter B")
    ax1.set_title(f"Independent (ρ = {corr_before:.2f})")
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.scatter(correlated["A"], correlated["B"], alpha=0.5, s=20, color='orange')
    ax2.set_xlabel("Parameter A")
    ax2.set_ylabel("Parameter B")
    ax2.set_title(f"Correlated (ρ = {corr_after:.2f})")
    ax2.grid(True, alpha=0.3)
    
    # Marginal distributions (preserved)
    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(independent["A"], bins=30, alpha=0.5, label="Independent", density=True)
    ax3.hist(correlated["A"], bins=30, alpha=0.5, label="Correlated", density=True)
    ax3.set_xlabel("Parameter A")
    ax3.set_ylabel("Density")
    ax3.set_title("Marginal Distribution Preserved")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("example3_custom_distributions.png", dpi=150)
    print("Saved visualization to example3_custom_distributions.png")
    plt.close()


def example_4_qmc_with_correlation():
    """
    Example 4: QMC Sampling with Correlations
    
    Shows that correlations work with quasi-Monte Carlo methods (Sobol, Halton, LHS).
    """
    print("\n" + "="*70)
    print("Example 4: QMC Sampling with Correlations")
    print("="*70)
    
    # Create sampler with Sobol sequence
    sampler = Sampler(n_samples=1024, method='sobol', seed=42)  # 1024 = 2^10
    sampler.add_parameter("X", NormalDistribution(mean=0, std=1))
    sampler.add_parameter("Y", NormalDistribution(mean=0, std=1))
    sampler.set_correlation("X", "Y", correlation=0.7)
    
    samples = sampler.generate(as_dataframe=True)
    corr = spearmanr(samples["X"], samples["Y"])[0]
    
    print(f"QMC correlation achieved: {corr:.3f}")
    
    # Compare different methods
    methods = ['random', 'sobol', 'halton', 'lhs']
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        
        # Create sampler with specific method
        n = 1024 if method == 'sobol' else 1000
        s = Sampler(n_samples=n, method=method, seed=42)
        s.add_parameter("X", NormalDistribution(mean=0, std=1))
        s.add_parameter("Y", NormalDistribution(mean=0, std=1))
        s.set_correlation("X", "Y", correlation=0.7)
        
        data = s.generate(as_dataframe=True)
        corr = spearmanr(data["X"], data["Y"])[0]
        
        ax.scatter(data["X"], data["Y"], alpha=0.5, s=20)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{method.upper()} (ρ = {corr:.2f})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig("example4_qmc_correlation.png", dpi=150)
    print("Saved visualization to example4_qmc_correlation.png")
    plt.close()


def example_5_real_world_portfolio():
    """
    Example 5: Real-World Portfolio Optimization
    
    Shows a realistic use case: modeling correlated asset returns for
    portfolio risk analysis.
    """
    print("\n" + "="*70)
    print("Example 5: Real-World Portfolio Optimization")
    print("="*70)
    
    # Asset return distributions (annual %)
    sampler = Sampler(n_samples=10000, seed=42)
    sampler.add_parameter("stocks", NormalDistribution(mean=8.0, std=15.0))
    sampler.add_parameter("bonds", NormalDistribution(mean=3.5, std=5.0))
    sampler.add_parameter("real_estate", NormalDistribution(mean=6.0, std=12.0))
    
    # Correlation matrix based on historical data
    # Stocks and bonds: -0.3 (negative correlation - diversification)
    # Stocks and real estate: 0.6 (positive)
    # Bonds and real estate: 0.2 (slight positive)
    correlation = np.array([
        [1.0, -0.3,  0.6],
        [-0.3, 1.0,  0.2],
        [0.6,  0.2,  1.0]
    ])
    
    sampler.set_correlation_matrix(correlation)
    
    # Generate scenarios
    scenarios = sampler.generate(as_dataframe=True)
    
    # Calculate portfolio returns for different allocations
    allocations = {
        "Conservative": [0.3, 0.6, 0.1],
        "Balanced": [0.5, 0.3, 0.2],
        "Aggressive": [0.7, 0.1, 0.2]
    }
    
    portfolio_returns = {}
    for name, weights in allocations.items():
        returns = (scenarios.values * np.array(weights)).sum(axis=1)
        portfolio_returns[name] = returns
        
        print(f"\n{name} Portfolio:")
        print(f"  Mean Return: {returns.mean():.2f}%")
        print(f"  Std Dev: {returns.std():.2f}%")
        print(f"  VaR (95%): {np.percentile(returns, 5):.2f}%")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Asset correlation
    actual_corr = np.corrcoef(scenarios.T)
    im = ax1.imshow(actual_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['Stocks', 'Bonds', 'RE'])
    ax1.set_yticklabels(['Stocks', 'Bonds', 'RE'])
    ax1.set_title("Asset Correlations")
    plt.colorbar(im, ax=ax1)
    
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{actual_corr[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    # Portfolio return distributions
    for name, returns in portfolio_returns.items():
        ax2.hist(returns, bins=50, alpha=0.5, label=name, density=True)
    
    ax2.set_xlabel("Portfolio Return (%)")
    ax2.set_ylabel("Density")
    ax2.set_title("Portfolio Return Distributions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    
    plt.tight_layout()
    plt.savefig("example5_portfolio.png", dpi=150)
    print("\nSaved visualization to example5_portfolio.png")
    plt.close()


if __name__ == "__main__":
    """Run all examples."""
    print("\n" + "="*70)
    print("MULTIFOLIO CORRELATED SAMPLING EXAMPLES")
    print("="*70)
    
    example_1_basic_pairwise_correlation()
    example_2_correlation_matrix()
    example_3_custom_distributions()
    example_4_qmc_with_correlation()
    example_5_real_world_portfolio()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
