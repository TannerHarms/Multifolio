"""Unit tests for Sampler class."""

import pytest
import numpy as np
import pandas as pd

from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution,
    NormalDistribution,
    ConstantDistribution,
)


class TestSamplerBasics:
    """Test basic Sampler functionality."""
    
    def test_initialization(self):
        sampler = Sampler()
        assert sampler.list_parameters() == []
    
    def test_add_parameter(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution())
        
        assert 'x' in sampler.list_parameters()
        assert isinstance(sampler.get_distribution('x'), UniformDistribution)
    
    def test_add_multiple_parameters(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution())
        sampler.add_parameter('y', NormalDistribution())
        sampler.add_parameter('z', ConstantDistribution(5))
        
        params = sampler.list_parameters()
        assert params == ['x', 'y', 'z']
    
    def test_duplicate_parameter_error(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution())
        
        with pytest.raises(ValueError):
            sampler.add_parameter('x', NormalDistribution())
    
    def test_remove_parameter(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution())
        sampler.add_parameter('y', NormalDistribution())
        
        sampler.remove_parameter('x')
        
        assert 'x' not in sampler.list_parameters()
        assert 'y' in sampler.list_parameters()
    
    def test_remove_nonexistent_parameter(self):
        sampler = Sampler()
        
        with pytest.raises(KeyError):
            sampler.remove_parameter('x')
    
    def test_get_distribution(self):
        sampler = Sampler()
        dist = UniformDistribution(low=5, high=10)
        sampler.add_parameter('x', dist)
        
        retrieved = sampler.get_distribution('x')
        assert retrieved is dist
    
    def test_get_nonexistent_distribution(self):
        sampler = Sampler()
        
        with pytest.raises(KeyError):
            sampler.get_distribution('x')
    
    def test_method_chaining(self):
        sampler = Sampler()
        result = (sampler
                  .add_parameter('x', UniformDistribution())
                  .add_parameter('y', NormalDistribution())
                  .remove_parameter('x'))
        
        assert result is sampler
        assert sampler.list_parameters() == ['y']


class TestSamplerGeneration:
    """Test sample generation."""
    
    def test_generate_no_parameters_error(self):
        sampler = Sampler()
        
        with pytest.raises(ValueError):
            sampler.generate(n=10)
    
    def test_generate_dict_format(self):
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=5, return_type='dict')
        
        assert isinstance(samples, dict)
        assert 'x' in samples
        assert 'y' in samples
        assert len(samples['x']) == 5
        assert len(samples['y']) == 5
    
    def test_generate_dataframe_format(self):
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        
        df = sampler.generate(n=10, return_type='dataframe')
        
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ['x', 'y']
        assert len(df) == 10
    
    def test_generate_array_format(self):
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        sampler.add_parameter('z', ConstantDistribution(5))
        
        arr = sampler.generate(n=20, return_type='array')
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (20, 3)
    
    def test_invalid_return_type(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution())
        
        with pytest.raises(ValueError):
            sampler.generate(n=5, return_type='invalid')
    
    def test_parameter_order_preserved(self):
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('z', ConstantDistribution(3))
        sampler.add_parameter('a', ConstantDistribution(1))
        sampler.add_parameter('m', ConstantDistribution(2))
        
        df = sampler.generate(n=1, return_type='dataframe')
        
        # Order should match insertion order, not alphabetical
        assert list(df.columns) == ['z', 'a', 'm']
    
    def test_reproducibility_with_seed(self):
        sampler1 = Sampler(random_seed=42)
        sampler1.add_parameter('x', UniformDistribution(random_seed=42))
        sampler1.add_parameter('y', NormalDistribution(random_seed=43))
        
        sampler2 = Sampler(random_seed=42)
        sampler2.add_parameter('x', UniformDistribution(random_seed=42))
        sampler2.add_parameter('y', NormalDistribution(random_seed=43))
        
        samples1 = sampler1.generate(n=10, return_type='dict')
        samples2 = sampler2.generate(n=10, return_type='dict')
        
        np.testing.assert_array_equal(samples1['x'], samples2['x'])
        np.testing.assert_array_equal(samples1['y'], samples2['y'])
    
    def test_large_sample_generation(self):
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=10000, return_type='dict')
        
        # Check that distributions are approximately correct
        assert np.mean(samples['x']) == pytest.approx(0.5, abs=0.05)
        assert np.std(samples['y']) == pytest.approx(1.0, abs=0.05)


class TestSamplerRepr:
    """Test string representation."""
    
    def test_repr_empty(self):
        sampler = Sampler()
        repr_str = repr(sampler)
        
        assert 'Sampler' in repr_str
        assert '(none)' in repr_str
    
    def test_repr_with_parameters(self):
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 10))
        sampler.add_parameter('y', NormalDistribution(5, 2))
        
        repr_str = repr(sampler)
        
        assert 'Sampler' in repr_str
        assert 'x' in repr_str
        assert 'UniformDistribution' in repr_str
        assert 'y' in repr_str
        assert 'NormalDistribution' in repr_str


# QMC Methods Tests

class TestQMCMethods:
    """Tests for Quasi-Monte Carlo sampling methods."""
    
    def test_sobol_sampling(self):
        """Test Sobol quasi-random sampling."""
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='dataframe', method='sobol')
        
        assert isinstance(samples, pd.DataFrame)
        assert len(samples) == 100
        assert list(samples.columns) == ['x', 'y']
        assert samples['x'].between(0, 1).all()
        assert samples['y'].between(0, 1).all()
    
    def test_halton_sampling(self):
        """Test Halton quasi-random sampling."""
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=50, return_type='dict', method='halton')
        
        assert 'x' in samples and 'y' in samples
        assert len(samples['x']) == 50
        assert len(samples['y']) == 50
        assert samples['x'].min() >= 0 and samples['x'].max() <= 1
    
    def test_lhs_sampling(self):
        """Test Latin Hypercube Sampling."""
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        sampler.add_parameter('z', UniformDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='array', method='lhs')
        
        assert samples.shape == (100, 3)
        # LHS should provide good coverage - check that samples span the range
        for col in range(3):
            assert samples[:, col].min() < 0.2  # Should have samples in lower range
            assert samples[:, col].max() > 0.8  # Should have samples in upper range
    
    def test_qmc_invalid_method(self):
        """Test that invalid QMC method raises error."""
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 1))
        
        with pytest.raises(ValueError, match="method must be"):
            sampler.generate(n=10, method='invalid_method')
    
    def test_qmc_with_all_distribution_types(self):
        """Test QMC methods work with all distribution types."""
        from multifolio.core.sampling.distributions.continuous import (
            BetaDistribution,
            TruncatedNormalDistribution,
        )
        from multifolio.core.sampling.distributions.discrete import (
            ConstantDistribution,
            PoissonDistribution,
            UniformDiscreteDistribution,
        )
        
        sampler = Sampler()
        sampler.add_parameter('uniform', UniformDistribution(0, 10))
        sampler.add_parameter('normal', NormalDistribution(5, 2))
        sampler.add_parameter('truncnorm', TruncatedNormalDistribution(0, 1, -2, 2))
        sampler.add_parameter('beta', BetaDistribution(2, 5, low=0, high=100))
        sampler.add_parameter('constant', ConstantDistribution(42))
        sampler.add_parameter('poisson', PoissonDistribution(3))
        sampler.add_parameter('discrete', UniformDiscreteDistribution(1, 10))
        
        # Test with each QMC method
        for method in ['sobol', 'halton', 'lhs']:
            samples = sampler.generate(n=20, return_type='dataframe', method=method)
            
            assert len(samples) == 20
            assert len(samples.columns) == 7
            
            # Verify ranges
            assert samples['uniform'].between(0, 10).all()
            assert samples['truncnorm'].between(-2, 2).all()
            assert samples['beta'].between(0, 100).all()
            assert (samples['constant'] == 42).all()
            assert (samples['poisson'] >= 0).all()
            assert samples['discrete'].between(1, 10).all()
    
    def test_qmc_reproducibility(self):
        """Test that QMC methods are reproducible with same seed."""
        sampler1 = Sampler(random_seed=123)
        sampler1.add_parameter('x', UniformDistribution(0, 1))
        sampler1.add_parameter('y', NormalDistribution(0, 1))
        
        sampler2 = Sampler(random_seed=123)
        sampler2.add_parameter('x', UniformDistribution(0, 1))
        sampler2.add_parameter('y', NormalDistribution(0, 1))
        
        for method in ['sobol', 'halton', 'lhs']:
            samples1 = sampler1.generate(n=50, return_type='array', method=method)
            samples2 = sampler2.generate(n=50, return_type='array', method=method)
            
            np.testing.assert_array_almost_equal(samples1, samples2)
    
    def test_qmc_better_coverage_than_random(self):
        """Test that QMC methods provide better space coverage than random sampling."""
        n_samples = 100
        
        # Create sampler with 2D uniform distribution
        sampler_random = Sampler(random_seed=42)
        sampler_random.add_parameter('x', UniformDistribution(0, 1))
        sampler_random.add_parameter('y', UniformDistribution(0, 1))
        
        sampler_sobol = Sampler(random_seed=42)
        sampler_sobol.add_parameter('x', UniformDistribution(0, 1))
        sampler_sobol.add_parameter('y', UniformDistribution(0, 1))
        
        random_samples = sampler_random.generate(n=n_samples, return_type='array', method='random')
        sobol_samples = sampler_sobol.generate(n=n_samples, return_type='array', method='sobol')
        
        # Measure coverage by dividing space into grid and checking filled cells
        n_bins = 10
        
        def count_filled_bins(samples):
            """Count number of grid cells that contain at least one sample."""
            x_bins = np.digitize(samples[:, 0], np.linspace(0, 1, n_bins + 1))
            y_bins = np.digitize(samples[:, 1], np.linspace(0, 1, n_bins + 1))
            filled = set(zip(x_bins, y_bins))
            return len(filled)
        
        random_coverage = count_filled_bins(random_samples)
        sobol_coverage = count_filled_bins(sobol_samples)
        
        # Sobol should generally have better coverage
        # (This is probabilistic, but Sobol is designed for this)
        assert sobol_coverage >= random_coverage * 0.9  # Allow some variance
    
    def test_qmc_distribution_statistics(self):
        """Test that QMC methods preserve distribution statistics."""
        n_samples = 1000
        
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('uniform', UniformDistribution(0, 10))
        sampler.add_parameter('normal', NormalDistribution(5, 2))
        
        for method in ['random', 'sobol', 'halton', 'lhs']:
            samples = sampler.generate(n=n_samples, return_type='dataframe', method=method)
            
            # Check uniform distribution
            assert abs(samples['uniform'].mean() - 5.0) < 0.5  # Should be around midpoint
            
            # Check normal distribution
            assert abs(samples['normal'].mean() - 5.0) < 0.5  # Mean should be close to 5
            assert abs(samples['normal'].std() - 2.0) < 0.5   # Std should be close to 2
