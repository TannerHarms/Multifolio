"""
Example demonstrating stratified sampling, bootstrap, quality metrics, and visualizations.
"""

import numpy as np
import pandas as pd
from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution, NormalDistribution
)

def main():
    print("="*70)
    print("Quality Metrics, Stratified Sampling & Visualization Demo")
    print("="*70)
    
    # ====================================================================
    # 1. SETUP SAMPLER
    # ====================================================================
    print("\n1. Creating sampler with correlations")
    print("-" * 70)
    
    sampler = Sampler(random_seed=42)
    sampler.add_parameter('temperature', UniformDistribution(20, 100))
    sampler.add_parameter('pressure', UniformDistribution(0.5, 2.0))
    sampler.add_parameter('flow_rate', NormalDistribution(mean=10, std=2))
    
    # Add correlations (temperature and pressure often correlated)
    sampler.set_correlation('temperature', 'pressure', 0.6)
    
    # Add derived parameter
    sampler.add_derived_parameter('efficiency', 'temperature * pressure / 100')
    
    print(f"Created sampler with {len(sampler._parameters)} base parameters")
    print(f"1 derived parameter, 1 correlation")
    
    # ====================================================================
    # 2. COMPARE RANDOM VS STRATIFIED SAMPLING
    # ====================================================================
    print("\n2. Comparing random vs stratified sampling")
    print("-" * 70)
    
    # Random sampling
    random_samples = sampler.generate(n=500, return_type='dataframe')
    
    # Stratified sampling
    stratified_samples = sampler.generate_stratified(
        n=500,
        strata_per_param=5,  # 5 bins per parameter
        method='random',
        return_type='dataframe'
    )
    
    print(f"Generated {len(random_samples)} random samples")
    print(f"Generated {len(stratified_samples)} stratified samples")
    
    # ====================================================================
    # 3. COMPUTE QUALITY METRICS
    # ====================================================================
    print("\n3. Computing quality metrics")
    print("-" * 70)
    
    # Metrics for random sampling
    random_metrics = sampler.compute_quality_metrics(
        random_samples,
        metrics=['coverage', 'discrepancy', 'correlation_error']
    )
    
    # Metrics for stratified sampling
    stratified_metrics = sampler.compute_quality_metrics(
        stratified_samples,
        metrics=['coverage', 'discrepancy', 'correlation_error']
    )
    
    print("\nRandom Sampling Metrics:")
    print(f"  Coverage: {random_metrics['coverage']:.3f}")
    print(f"  Discrepancy: {random_metrics['discrepancy']:.4f}")
    print(f"  Correlation RMSE: {random_metrics['correlation_error']['rmse']:.4f}")
    
    print("\nStratified Sampling Metrics:")
    print(f"  Coverage: {stratified_metrics['coverage']:.3f}")
    print(f"  Discrepancy: {stratified_metrics['discrepancy']:.4f}")
    print(f"  Correlation RMSE: {stratified_metrics['correlation_error']['rmse']:.4f}")
    
    print("\nStratified sampling typically shows:")
    print("  ✓ Better coverage (closer to 1.0)")
    print("  ✓ Lower discrepancy (better space-filling)")
    print("  ✓ Similar correlation preservation")
    
    # ====================================================================
    # 4. BOOTSTRAP RESAMPLING
    # ====================================================================
    print("\n4. Bootstrap resampling for uncertainty quantification")
    print("-" * 70)
    
    # Generate bootstrap samples and compute statistics
    n_bootstrap = 100
    bootstrap_means = []
    
    for i in range(n_bootstrap):
        boot_sample = sampler.bootstrap_resample(
            stratified_samples,
            n=500,
            random_seed=42 + i,
            return_type='dataframe'
        )
        bootstrap_means.append(boot_sample['efficiency'].mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    print(f"Performed {n_bootstrap} bootstrap resamples")
    print(f"\nEfficiency statistics:")
    print(f"  Original mean: {stratified_samples['efficiency'].mean():.3f}")
    print(f"  Bootstrap mean: {bootstrap_means.mean():.3f}")
    print(f"  Bootstrap std: {bootstrap_means.std():.3f}")
    print(f"  95% CI: [{np.percentile(bootstrap_means, 2.5):.3f}, "
          f"{np.percentile(bootstrap_means, 97.5):.3f}]")
    
    # ====================================================================
    # 5. STRATIFIED SAMPLING METHODS
    # ====================================================================
    print("\n5. Comparing stratified sampling methods")
    print("-" * 70)
    
    methods = ['random', 'center', 'jittered']
    method_samples = {}
    
    for method in methods:
        method_samples[method] = sampler.generate_stratified(
            n=500,
            strata_per_param=5,
            method=method,
            return_type='dataframe',
            random_seed=42
        )
    
    print("\nMethod characteristics:")
    for method in methods:
        samples = method_samples[method]
        print(f"\n  {method.upper()}:")
        print(f"    Temperature range: [{samples['temperature'].min():.1f}, "
              f"{samples['temperature'].max():.1f}]")
        print(f"    Temperature std: {samples['temperature'].std():.2f}")
        print(f"    Efficiency mean: {samples['efficiency'].mean():.3f}")
    
    # ====================================================================
    # 6. VISUALIZATIONS
    # ====================================================================
    print("\n6. Creating visualizations")
    print("-" * 70)
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot distributions
        print("  Creating distribution plots...")
        fig1 = sampler.plot_distributions(
            stratified_samples,
            parameters=['temperature', 'pressure', 'flow_rate'],
            figsize=(15, 5)
        )
        fig1.suptitle('Stratified Sample Distributions', fontsize=14, y=1.02)
        plt.savefig('stratified_distributions.png', dpi=150, bbox_inches='tight')
        print("    Saved: stratified_distributions.png")
        
        # Plot correlation matrix
        print("  Creating correlation matrix...")
        fig2 = sampler.plot_correlation_matrix(
            stratified_samples,
            figsize=(8, 6)
        )
        fig2.suptitle('Sample Correlation Matrix', fontsize=14, y=0.98)
        plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
        print("    Saved: correlation_matrix.png")
        
        # Plot pairwise relationships
        print("  Creating pairwise plots...")
        fig3 = sampler.plot_pairwise(
            stratified_samples,
            parameters=['temperature', 'pressure', 'efficiency'],
            figsize=(10, 10),
            alpha=0.3
        )
        fig3.suptitle('Pairwise Relationships', fontsize=14, y=0.995)
        plt.savefig('pairwise_plots.png', dpi=150, bbox_inches='tight')
        print("    Saved: pairwise_plots.png")
        
        plt.close('all')
        print("\n  ✓ All visualizations created successfully")
        
    except ImportError:
        print("  Matplotlib not available - skipping visualizations")
        print("  Install with: pip install matplotlib")
    
    # ====================================================================
    # 7. COMPREHENSIVE QUALITY REPORT
    # ====================================================================
    print("\n7. Comprehensive quality report")
    print("-" * 70)
    
    all_metrics = sampler.compute_quality_metrics(stratified_samples)
    
    print("\nAll Quality Metrics:")
    print(f"  Coverage: {all_metrics['coverage']:.3f} "
          f"({all_metrics['coverage_details']['occupied_bins']}/"
          f"{all_metrics['coverage_details']['total_bins']} bins)")
    print(f"  Discrepancy: {all_metrics['discrepancy']:.4f}")
    
    if 'correlation_error' in all_metrics:
        print(f"\n  Correlation Errors:")
        print(f"    RMSE: {all_metrics['correlation_error']['rmse']:.4f}")
        print(f"    Max error: {all_metrics['correlation_error']['max_abs_error']:.4f}")
    
    print(f"\n  Distribution KS Tests:")
    for param, result in all_metrics['distribution_ks'].items():
        status = "✓" if result['passes'] else "✗"
        print(f"    {param}: {status} (p={result['pvalue']:.4f})")
    
    print(f"\n  Uniformity Tests:")
    for param, result in all_metrics['uniformity'].items():
        status = "✓" if result['passes'] else "✗"
        print(f"    {param}: {status} (p={result['pvalue']:.4f})")
    
    # ====================================================================
    # 8. PRACTICAL USE CASE: EXPERIMENT DESIGN
    # ====================================================================
    print("\n8. Practical use case: Experimental design")
    print("-" * 70)
    
    # Generate experiment design with good coverage
    experiment_design = sampler.generate_stratified(
        n=50,  # 50 experiments
        strata_per_param=3,  # Low/Medium/High for each parameter
        method='center',  # Sample at bin centers for reproducibility
        return_type='dataframe'
    )
    
    # Add experiment IDs
    experiment_design.insert(0, 'experiment_id', range(1, len(experiment_design) + 1))
    
    print(f"\nGenerated {len(experiment_design)} experiments")
    print("\nFirst 5 experiments:")
    print(experiment_design.head().to_string(index=False))
    
    # Save to CSV for use
    experiment_design.to_csv('experiment_design.csv', index=False)
    print("\nSaved experiment design to: experiment_design.csv")
    
    # Compute quality to ensure good experimental coverage
    design_metrics = sampler.compute_quality_metrics(
        experiment_design.drop('experiment_id', axis=1),
        metrics=['coverage']
    )
    print(f"Design coverage: {design_metrics['coverage']:.1%} of parameter space")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("  • Stratified sampling provides better parameter space coverage")
    print("  • Quality metrics help validate sampling effectiveness")
    print("  • Bootstrap resampling enables uncertainty quantification")
    print("  • Visualizations aid in understanding sample distributions")
    print("  • Stratified 'center' method excellent for experimental design")


if __name__ == "__main__":
    main()
