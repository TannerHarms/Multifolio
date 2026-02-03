"""
Example demonstrating advanced sampler features:
- Save/load configuration and data
- Parameter constraints
- Conditional generation
- Batch generation
- Sample filtering
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import UniformDistribution, NormalDistribution


def main():
    print("=" * 70)
    print("Advanced Sampler Features Demo")
    print("=" * 70)
    
    # Create temporary directory for saving files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # ====================================================================
        # 1. CREATE SAMPLER WITH CONSTRAINTS
        # ====================================================================
        print("\n1. Creating sampler with parameters, correlations, and constraints")
        print("-" * 70)
        
        sampler = Sampler(random_seed=42)
        
        # Portfolio allocation example
        sampler.add_parameter('stocks', UniformDistribution(0, 1))
        sampler.add_parameter('bonds', UniformDistribution(0, 1))
        sampler.add_parameter('cash', UniformDistribution(0, 1))
        
        # Correlations (stocks and bonds are somewhat correlated)
        sampler.set_correlation('stocks', 'bonds', 0.4)
        
        # Derived parameters
        sampler.add_derived_parameter('total', 'stocks + bonds + cash')
        sampler.add_derived_parameter('portfolio_value', 'total * 100000')
        
        # Constraints (must sum to 1, minimum allocations)
        sampler.add_constraint('total >= 0.98')
        sampler.add_constraint('total <= 1.02')
        sampler.add_constraint('stocks >= 0.1')  # At least 10% in stocks
        sampler.add_constraint('bonds >= 0.1')   # At least 10% in bonds
        sampler.add_constraint('cash >= 0.05')   # At least 5% in cash
        
        print(f"Created sampler with {len(sampler._parameters)} parameters")
        print(f"Correlations: {sampler.has_correlations()}")
        print(f"Derived parameters: {list(sampler._derived_parameters.keys())}")
        print(f"Constraints: {len(sampler.get_constraints())}")
        
        # ====================================================================
        # 2. SAVE CONFIGURATION
        # ====================================================================
        print("\n2. Saving sampler configuration")
        print("-" * 70)
        
        config_file = tmpdir / "portfolio_sampler.json"
        sampler.save_config(config_file)
        print(f"Saved configuration to: {config_file}")
        
        # ====================================================================
        # 3. GENERATE SAMPLES WITH CONSTRAINTS
        # ====================================================================
        print("\n3. Generating samples (constraints automatically applied)")
        print("-" * 70)
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        print(f"Generated {len(samples)} samples")
        print(f"\nSample statistics:")
        print(samples[['stocks', 'bonds', 'cash', 'total']].describe())
        
        # Verify constraints
        print(f"\nConstraint verification:")
        print(f"  All total >= 0.98: {(samples['total'] >= 0.98).all()}")
        print(f"  All total <= 1.02: {(samples['total'] <= 1.02).all()}")
        print(f"  All stocks >= 0.1: {(samples['stocks'] >= 0.1).all()}")
        
        # ====================================================================
        # 4. SAVE DATA
        # ====================================================================
        print("\n4. Saving generated data in multiple formats")
        print("-" * 70)
        
        csv_file = tmpdir / "samples.csv"
        Sampler.save_data(samples, csv_file)
        print(f"Saved as CSV: {csv_file}")
        
        # Try HDF5 if pytables is available
        try:
            h5_file = tmpdir / "samples.h5"
            Sampler.save_data(samples, h5_file, format='hdf5')
            print(f"Saved as HDF5: {h5_file}")
            
            # Compare file sizes
            csv_size = csv_file.stat().st_size
            h5_size = h5_file.stat().st_size
            print(f"\nFile size comparison:")
            print(f"  CSV:  {csv_size:,} bytes")
            print(f"  HDF5: {h5_size:,} bytes ({h5_size/csv_size*100:.1f}% of CSV size)")
        except ImportError:
            print("  (HDF5 format skipped - neither pytables nor h5py installed)")
            print("  To enable HDF5 support, install: pip install tables (or pip install h5py)")
        
        # ====================================================================
        # 5. LOAD CONFIGURATION
        # ====================================================================
        print("\n5. Loading configuration from file")
        print("-" * 70)
        
        sampler2 = Sampler.load_config(config_file)
        print(f"Loaded sampler with {len(sampler2._parameters)} parameters")
        print(f"Constraints loaded: {len(sampler2.get_constraints())}")
        
        # ====================================================================
        # 6. CONDITIONAL GENERATION
        # ====================================================================
        print("\n6. Conditional generation (specific scenarios)")
        print("-" * 70)
        
        # Generate portfolios with high stock allocation
        high_stock_samples = sampler.generate_conditional(
            n=100,
            condition='stocks > 0.5',
            return_type='dataframe'
        )
        print(f"Generated {len(high_stock_samples)} samples with stocks > 50%")
        print(f"Stock allocation range: [{high_stock_samples['stocks'].min():.3f}, "
              f"{high_stock_samples['stocks'].max():.3f}]")
        
        # Generate balanced portfolios
        balanced_samples = sampler.generate_conditional(
            n=100,
            condition='(stocks > 0.25) & (stocks < 0.35) & (bonds > 0.25) & (bonds < 0.35)',
            return_type='dataframe'
        )
        print(f"\nGenerated {len(balanced_samples)} balanced portfolios")
        print(f"Stocks: mean={balanced_samples['stocks'].mean():.3f}, "
              f"std={balanced_samples['stocks'].std():.3f}")
        print(f"Bonds:  mean={balanced_samples['bonds'].mean():.3f}, "
              f"std={balanced_samples['bonds'].std():.3f}")
        
        # ====================================================================
        # 7. SAMPLE FILTERING
        # ====================================================================
        print("\n7. Post-generation filtering")
        print("-" * 70)
        
        # Filter for high-value portfolios
        high_value = sampler.filter_samples(samples, 'portfolio_value > 101000')
        print(f"Samples with portfolio > $101k: {len(high_value)} ({len(high_value)/len(samples)*100:.1f}%)")
        
        # Filter for conservative portfolios
        conservative = sampler.filter_samples(samples, '(bonds + cash) > 0.5')
        print(f"Conservative portfolios: {len(conservative)} ({len(conservative)/len(samples)*100:.1f}%)")
        
        # Complex filter
        aggressive = sampler.filter_samples(
            samples,
            '(stocks > 0.4) & (cash < 0.2)'
        )
        print(f"Aggressive portfolios: {len(aggressive)} ({len(aggressive)/len(samples)*100:.1f}%)")
        
        # ====================================================================
        # 8. BATCH GENERATION
        # ====================================================================
        print("\n8. Batch generation (memory-efficient)")
        print("-" * 70)
        
        print("Generating 10 batches of 500 samples each...")
        total_samples = 0
        batch_stats = []
        
        for i, batch in enumerate(sampler.generate_batches(
            n_per_batch=500,
            n_batches=10,
            track_batch_id=True
        ), 1):
            total_samples += len(batch)
            mean_stocks = batch['stocks'].mean()
            batch_stats.append(mean_stocks)
            
            if i % 3 == 0:  # Print every 3rd batch
                print(f"  Batch {i}: {len(batch)} samples, "
                      f"mean stocks allocation = {mean_stocks:.3f}")
        
        print(f"\nTotal samples generated: {total_samples}")
        print(f"Stock allocation consistency across batches: "
              f"std = {np.std(batch_stats):.4f}")
        
        # ====================================================================
        # 9. CONSTRAINT MANAGEMENT
        # ====================================================================
        print("\n9. Managing constraints")
        print("-" * 70)
        
        print("Current constraints:")
        for i, constraint in enumerate(sampler.get_constraints(), 1):
            print(f"  {i}. {constraint}")
        
        # Generate without constraints
        print("\nGenerating WITHOUT constraints...")
        unconstrained = sampler.generate(n=100, apply_constraints=False, return_type='dataframe')
        print(f"Total allocation range: [{unconstrained['total'].min():.3f}, "
              f"{unconstrained['total'].max():.3f}]")
        
        # Generate with constraints
        print("\nGenerating WITH constraints...")
        constrained = sampler.generate(n=100, apply_constraints=True, return_type='dataframe')
        print(f"Total allocation range: [{constrained['total'].min():.3f}, "
              f"{constrained['total'].max():.3f}]")
        
        # ====================================================================
        # 10. CORRELATION PROPAGATION
        # ====================================================================
        print("\n10. Correlation propagation to derived parameters")
        print("-" * 70)
        
        # Generate samples
        corr_samples = sampler.generate(n=2000, return_type='dataframe')
        
        # Check correlations
        corr_matrix = corr_samples[['stocks', 'bonds', 'total']].corr()
        print("\nCorrelation matrix:")
        print(corr_matrix)
        
        print(f"\nObservations:")
        print(f"  stocks-bonds correlation: {corr_matrix.loc['stocks', 'bonds']:.3f} "
              f"(specified: 0.4)")
        print(f"  stocks-total correlation: {corr_matrix.loc['stocks', 'total']:.3f} "
              f"(automatically propagated)")
        print(f"  bonds-total correlation: {corr_matrix.loc['bonds', 'total']:.3f} "
              f"(automatically propagated)")
        print("\n  → Correlations automatically propagate to derived parameters!")
        
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
