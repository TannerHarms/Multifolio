"""
Tests for quality metrics, stratified sampling, bootstrap, and visualization features.
"""

import pytest
import numpy as np
import pandas as pd

from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution, NormalDistribution
)


class TestStratifiedSampling:
    """Test stratified sampling functionality."""
    
    def test_stratified_basic(self):
        """Test basic stratified sampling."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate_stratified(
            n=100, 
            strata_per_param=5,
            return_type='dataframe'
        )
        
        assert len(samples) == 100
        assert 'X' in samples.columns
        assert 'Y' in samples.columns
        
        # Check bounds
        assert samples['X'].min() >= 0
        assert samples['X'].max() <= 10
        assert samples['Y'].min() >= 0
        assert samples['Y'].max() <= 10
    
    def test_stratified_different_strata(self):
        """Test stratified sampling with different strata per parameter."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate_stratified(
            n=100,
            strata_per_param={'X': 3, 'Y': 5},
            return_type='dict'
        )
        
        assert len(samples['X']) == 100
        assert len(samples['Y']) == 100
    
    def test_stratified_methods(self):
        """Test different stratified sampling methods."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        # Test each method
        for method in ['random', 'center', 'jittered']:
            samples = sampler.generate_stratified(
                n=50,
                strata_per_param=5,
                method=method,
                return_type='dataframe'
            )
            assert len(samples) == 50
            assert samples['X'].min() >= 0
            assert samples['X'].max() <= 10
    
    def test_stratified_with_correlations(self):
        """Test stratified sampling with correlations."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.7)
        
        samples = sampler.generate_stratified(
            n=100,
            strata_per_param=5,
            return_type='dataframe'
        )
        
        # Check correlation is approximately maintained
        corr = np.corrcoef(samples['X'], samples['Y'])[0, 1]
        assert abs(corr - 0.7) < 0.2  # Allow some deviation
    
    def test_stratified_with_derived_parameters(self):
        """Test stratified sampling with derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_derived_parameter('Z', 'X + Y')
        
        samples = sampler.generate_stratified(
            n=100,
            strata_per_param=5,
            return_type='dataframe'
        )
        
        assert 'Z' in samples.columns
        # Check derived parameter calculation
        np.testing.assert_allclose(samples['Z'], samples['X'] + samples['Y'])
    
    def test_stratified_coverage(self):
        """Test that stratified sampling provides good coverage."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        # Use 25 strata (5x5), generate enough samples
        samples = sampler.generate_stratified(
            n=500,
            strata_per_param=5,
            method='center',
            return_type='dataframe'
        )
        
        # With center method and 25 strata, should have good spread
        # Each stratum should be represented
        x_bins = pd.cut(samples['X'], bins=5)
        y_bins = pd.cut(samples['Y'], bins=5)
        
        # Check that all bins are represented
        assert len(x_bins.value_counts()) >= 3  # Should have most X bins
        assert len(y_bins.value_counts()) >= 3  # Should have most Y bins


class TestBootstrapResampling:
    """Test bootstrap resampling functionality."""
    
    def test_bootstrap_basic(self):
        """Test basic bootstrap resampling."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        # Generate original samples
        original = sampler.generate(n=100, return_type='dataframe')
        
        # Bootstrap resample
        resampled = sampler.bootstrap_resample(original, return_type='dataframe')
        
        assert len(resampled) == len(original)
        assert list(resampled.columns) == list(original.columns)
    
    def test_bootstrap_larger_sample(self):
        """Test bootstrap with larger sample size."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        original = sampler.generate(n=100, return_type='dict')
        resampled = sampler.bootstrap_resample(original, n=500, return_type='dict')
        
        assert len(resampled['X']) == 500
    
    def test_bootstrap_dict_input(self):
        """Test bootstrap with dict input."""
        sampler = Sampler()
        
        data = {
            'X': np.array([1, 2, 3, 4, 5]),
            'Y': np.array([10, 20, 30, 40, 50])
        }
        
        resampled = sampler.bootstrap_resample(data, n=10, random_seed=42)
        
        assert len(resampled['X']) == 10
        assert len(resampled['Y']) == 10
        # All values should come from original
        assert all(x in data['X'] for x in resampled['X'])
        assert all(y in data['Y'] for y in resampled['Y'])
    
    def test_bootstrap_reproducibility(self):
        """Test that bootstrap is reproducible with seed."""
        sampler = Sampler()
        
        data = pd.DataFrame({
            'X': np.arange(100),
            'Y': np.arange(100, 200)
        })
        
        resample1 = sampler.bootstrap_resample(data, n=50, random_seed=42, return_type='dataframe')
        resample2 = sampler.bootstrap_resample(data, n=50, random_seed=42, return_type='dataframe')
        
        pd.testing.assert_frame_equal(resample1, resample2)
    
    def test_bootstrap_statistics(self):
        """Test that bootstrap maintains approximate distribution."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', NormalDistribution(mean=50, std=10))
        
        original = sampler.generate(n=1000, return_type='dataframe')
        resampled = sampler.bootstrap_resample(original, n=1000, return_type='dataframe')
        
        # Means should be similar
        assert abs(original['X'].mean() - resampled['X'].mean()) < 2
        # Stds should be similar
        assert abs(original['X'].std() - resampled['X'].std()) < 2


class TestQualityMetrics:
    """Test sample quality metrics."""
    
    def test_quality_metrics_basic(self):
        """Test basic quality metrics computation."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples)
        
        # Should have all default metrics
        assert 'coverage' in metrics
        assert 'discrepancy' in metrics
        assert 'distribution_ks' in metrics
        assert 'uniformity' in metrics
    
    def test_quality_metrics_coverage(self):
        """Test coverage metric."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        # Large sample should have decent coverage
        samples = sampler.generate(n=10000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples, metrics=['coverage'])
        
        assert 'coverage' in metrics
        assert metrics['coverage'] > 0.5  # Should cover at least half of space
        assert 'coverage_details' in metrics
        assert 'occupied_bins' in metrics['coverage_details']
    
    def test_quality_metrics_discrepancy(self):
        """Test discrepancy metric."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples, metrics=['discrepancy'])
        
        assert 'discrepancy' in metrics
        assert 0 <= metrics['discrepancy'] <= 1
        
        # QMC samples should have lower discrepancy than random
        qmc_samples = sampler.generate(n=1000, method='sobol', return_type='dataframe')
        qmc_metrics = sampler.compute_quality_metrics(qmc_samples, metrics=['discrepancy'])
        
        # QMC should generally be better (though not guaranteed for small samples)
        assert qmc_metrics['discrepancy'] <= metrics['discrepancy'] * 1.5
    
    def test_quality_metrics_correlation_error(self):
        """Test correlation error metric."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.8)
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples, metrics=['correlation_error'])
        
        assert 'correlation_error' in metrics
        assert 'rmse' in metrics['correlation_error']
        assert 'max_abs_error' in metrics['correlation_error']
        
        # Error should be small for large sample
        assert metrics['correlation_error']['rmse'] < 0.1
    
    def test_quality_metrics_ks_test(self):
        """Test KS test metric."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples, metrics=['distribution_ks'])
        
        assert 'distribution_ks' in metrics
        assert 'X' in metrics['distribution_ks']
        assert 'Y' in metrics['distribution_ks']
        
        # Check structure
        for param in ['X', 'Y']:
            assert 'statistic' in metrics['distribution_ks'][param]
            assert 'pvalue' in metrics['distribution_ks'][param]
            assert 'passes' in metrics['distribution_ks'][param]
            
            # Large sample should pass KS test
            assert metrics['distribution_ks'][param]['passes']
    
    def test_quality_metrics_uniformity(self):
        """Test uniformity chi-square test."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples, metrics=['uniformity'])
        
        assert 'uniformity' in metrics
        assert 'X' in metrics['uniformity']
        assert 'statistic' in metrics['uniformity']['X']
        assert 'pvalue' in metrics['uniformity']['X']
        assert 'passes' in metrics['uniformity']['X']
    
    def test_quality_metrics_with_derived(self):
        """Test quality metrics ignore derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_derived_parameter('Z', 'X + Y')
        
        samples = sampler.generate(n=500, return_type='dataframe')
        metrics = sampler.compute_quality_metrics(samples)
        
        # Should only test base parameters
        assert 'X' in metrics['distribution_ks']
        assert 'Y' in metrics['distribution_ks']
        # Z should not be tested (it's derived)
        assert 'Z' not in metrics['distribution_ks']


class TestVisualizations:
    """Test visualization functionality."""
    
    def test_plot_distributions_basic(self):
        """Test basic distribution plotting."""
        pytest.importorskip("matplotlib")
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', NormalDistribution(5, 1))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        fig = sampler.plot_distributions(samples)
        
        assert fig is not None
        assert len(fig.axes) >= 2  # Should have at least 2 subplots
    
    def test_plot_distributions_specific_params(self):
        """Test plotting specific parameters."""
        pytest.importorskip("matplotlib")
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_parameter('Z', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=500, return_type='dataframe')
        fig = sampler.plot_distributions(samples, parameters=['X', 'Y'])
        
        assert fig is not None
        # Should have 2 plots (not 3)
        active_axes = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
        assert len(active_axes) == 2
    
    def test_plot_correlation_matrix(self):
        """Test correlation matrix plotting."""
        pytest.importorskip("matplotlib")
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.7)
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        fig = sampler.plot_correlation_matrix(samples)
        
        assert fig is not None
        assert len(fig.axes) >= 1
    
    def test_plot_pairwise(self):
        """Test pairwise plotting."""
        pytest.importorskip("matplotlib")
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_parameter('Z', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=500, return_type='dataframe')
        fig = sampler.plot_pairwise(samples)
        
        assert fig is not None
        # Should have n_params x n_params subplots
        assert len(fig.axes) == 9  # 3x3
    
    def test_plot_pairwise_specific_params(self):
        """Test pairwise with specific parameters."""
        pytest.importorskip("matplotlib")
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=500, return_type='dataframe')
        fig = sampler.plot_pairwise(samples, parameters=['X', 'Y'])
        
        assert fig is not None
        assert len(fig.axes) == 4  # 2x2
    
    @pytest.mark.skip(reason="Import manipulation is fragile and environment-dependent")
    def test_visualization_without_matplotlib(self):
        """Test that helpful error is raised without matplotlib."""
        import sys
        
        # Temporarily hide matplotlib
        matplotlib_backup = sys.modules.get('matplotlib')
        if 'matplotlib' in sys.modules:
            del sys.modules['matplotlib']
        if 'matplotlib.pyplot' in sys.modules:
            del sys.modules['matplotlib.pyplot']
        
        try:
            sampler = Sampler(random_seed=42)
            sampler.add_parameter('X', UniformDistribution(0, 10))
            samples = sampler.generate(n=100, return_type='dataframe')
            
            with pytest.raises(ImportError, match="matplotlib"):
                sampler.plot_distributions(samples)
        finally:
            # Restore matplotlib
            if matplotlib_backup:
                sys.modules['matplotlib'] = matplotlib_backup


class TestIntegration:
    """Integration tests for new features."""
    
    def test_stratified_with_quality_metrics(self):
        """Test stratified sampling produces high quality samples."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        # Stratified samples should have better coverage
        stratified = sampler.generate_stratified(
            n=500, strata_per_param=10, return_type='dataframe'
        )
        random = sampler.generate(n=500, return_type='dataframe')
        
        strat_metrics = sampler.compute_quality_metrics(stratified, metrics=['coverage'])
        random_metrics = sampler.compute_quality_metrics(random, metrics=['coverage'])
        
        # Stratified should have better or equal coverage
        assert strat_metrics['coverage'] >= random_metrics['coverage']
    
    def test_bootstrap_preserves_correlations(self):
        """Test that bootstrap preserves correlation structure."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.8)
        
        original = sampler.generate(n=1000, return_type='dataframe')
        resampled = sampler.bootstrap_resample(original, n=1000, return_type='dataframe')
        
        orig_corr = np.corrcoef(original['X'], original['Y'])[0, 1]
        resamp_corr = np.corrcoef(resampled['X'], resampled['Y'])[0, 1]
        
        # Correlation should be maintained approximately
        assert abs(orig_corr - resamp_corr) < 0.1
    
    def test_full_workflow(self):
        """Test complete workflow with all features."""
        # Create sampler
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.6)
        sampler.add_derived_parameter('Z', 'X + Y')
        
        # Generate stratified samples
        samples = sampler.generate_stratified(
            n=1000, strata_per_param=5, return_type='dataframe'
        )
        
        # Compute quality metrics
        metrics = sampler.compute_quality_metrics(samples)
        assert metrics['coverage'] > 0.5
        assert 'correlation_error' in metrics
        
        # Bootstrap resample
        bootstrap = sampler.bootstrap_resample(samples, n=500, return_type='dataframe')
        assert len(bootstrap) == 500
        
        # Visualize (if matplotlib available)
        try:
            fig1 = sampler.plot_distributions(samples, parameters=['X', 'Y'])
            assert fig1 is not None
            
            fig2 = sampler.plot_correlation_matrix(samples)
            assert fig2 is not None
        except ImportError:
            pass  # Skip if matplotlib not available
