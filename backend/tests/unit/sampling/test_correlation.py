"""Unit tests for correlation structures and correlated sampling."""

import pytest
import numpy as np
from scipy import stats

from multifolio.core.sampling.correlation import (
    GaussianCopula,
    CorrelationManager
)
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution,
    NormalDistribution,
    ExponentialDistribution
)


class TestGaussianCopula:
    """Test Gaussian copula functionality."""
    
    def test_initialization(self):
        """Test copula can be initialized with valid correlation matrix."""
        corr = np.array([[1.0, 0.7], [0.7, 1.0]])
        copula = GaussianCopula(corr)
        
        assert copula.n_variables == 2
        np.testing.assert_array_equal(copula.correlation_matrix, corr)
    
    def test_invalid_correlation_matrix(self):
        """Test that invalid correlation matrices are rejected."""
        # Not square
        with pytest.raises(ValueError, match="must be square"):
            GaussianCopula(np.array([[1.0, 0.5]]))
        
        # Wrong diagonal
        with pytest.raises(ValueError, match="diagonal must be all ones"):
            GaussianCopula(np.array([[1.0, 0.5], [0.5, 0.9]]))
        
        # Not symmetric
        with pytest.raises(ValueError, match="must be symmetric"):
            GaussianCopula(np.array([[1.0, 0.5], [0.7, 1.0]]))
        
        # Perfect correlation (not positive definite - singular)
        with pytest.raises(ValueError, match="positive definite"):
            GaussianCopula(np.array([[1.0, 1.0], [1.0, 1.0]]))
    
    def test_transform_uniform_preserves_range(self):
        """Test that transformed samples stay in [0, 1]."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        copula = GaussianCopula(corr)
        
        uniform_samples = np.random.uniform(0, 1, (1000, 2))
        transformed = copula.transform_uniform(uniform_samples)
        
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
    
    def test_transform_uniform_creates_correlation(self):
        """Test that transformation induces correlation."""
        np.random.seed(42)
        target_corr = 0.8
        corr = np.array([[1.0, target_corr], [target_corr, 1.0]])
        copula = GaussianCopula(corr)
        
        # Independent uniforms
        uniform_samples = np.random.uniform(0, 1, (5000, 2))
        
        # Check independence before
        corr_before = np.corrcoef(uniform_samples.T)[0, 1]
        assert abs(corr_before) < 0.1  # Should be near zero
        
        # Apply correlation
        transformed = copula.transform_uniform(uniform_samples)
        
        # Check correlation after (rank correlation)
        corr_after = stats.spearmanr(transformed)[0]
        
        # Spearman correlation should be close to target
        # (not exact due to Gaussian copula transformation)
        expected_spearman = (6 / np.pi) * np.arcsin(target_corr / 2)
        assert abs(corr_after - expected_spearman) < 0.1
    
    def test_near_perfect_correlation(self):
        """Test near-perfect positive correlation (avoids singularity)."""
        # Use 0.999 instead of 1.0 to avoid singular matrix
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        copula = GaussianCopula(corr)
        
        uniform_samples = np.random.uniform(0, 1, (1000, 2))
        transformed = copula.transform_uniform(uniform_samples)
        
        # With near-perfect correlation, both columns should be nearly identical
        spearman_corr = stats.spearmanr(transformed)[0]
        assert spearman_corr > 0.95  # Very high correlation
    
    def test_negative_correlation(self):
        """Test negative correlation."""
        np.random.seed(42)
        target_corr = -0.7
        corr = np.array([[1.0, target_corr], [target_corr, 1.0]])
        copula = GaussianCopula(corr)
        
        uniform_samples = np.random.uniform(0, 1, (5000, 2))
        transformed = copula.transform_uniform(uniform_samples)
        
        # Check negative correlation
        corr_after = stats.spearmanr(transformed)[0]
        expected_spearman = (6 / np.pi) * np.arcsin(target_corr / 2)
        assert corr_after < 0
        assert abs(corr_after - expected_spearman) < 0.1
    
    def test_multivariate_correlation(self):
        """Test 3-variable correlation."""
        corr = np.array([
            [1.0, 0.6, -0.3],
            [0.6, 1.0, 0.0],
            [-0.3, 0.0, 1.0]
        ])
        copula = GaussianCopula(corr)
        
        uniform_samples = np.random.uniform(0, 1, (5000, 3))
        transformed = copula.transform_uniform(uniform_samples)
        
        # Check correlation structure is preserved
        spearman_matrix = stats.spearmanr(transformed)[0]
        
        # Check key correlations (allowing some tolerance)
        assert spearman_matrix[0, 1] > 0.4  # Positive
        assert spearman_matrix[0, 2] < -0.1  # Negative
        assert abs(spearman_matrix[1, 2]) < 0.15  # Near zero


class TestCorrelationManager:
    """Test correlation manager functionality."""
    
    def test_initialization(self):
        """Test manager initialization."""
        params = ['a', 'b', 'c']
        manager = CorrelationManager(params)
        
        assert manager.parameter_names == params
        assert manager.n_parameters == 3
        assert not manager.has_correlations()
    
    def test_set_correlation(self):
        """Test setting pairwise correlation."""
        manager = CorrelationManager(['x', 'y'])
        manager.set_correlation('x', 'y', 0.7)
        
        assert manager.has_correlations()
        assert manager.get_correlation('x', 'y') == 0.7
        assert manager.get_correlation('y', 'x') == 0.7
    
    def test_set_correlation_invalid_params(self):
        """Test error on invalid parameter names."""
        manager = CorrelationManager(['x', 'y'])
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            manager.set_correlation('x', 'z', 0.5)
    
    def test_set_correlation_invalid_value(self):
        """Test error on invalid correlation value."""
        manager = CorrelationManager(['x', 'y'])
        
        with pytest.raises(ValueError, match="must be in"):
            manager.set_correlation('x', 'y', 1.5)
    
    def test_set_correlation_matrix(self):
        """Test setting full correlation matrix."""
        manager = CorrelationManager(['a', 'b', 'c'])
        
        corr = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0]
        ])
        manager.set_correlation_matrix(corr)
        
        np.testing.assert_array_equal(manager.get_correlation_matrix(), corr)
    
    def test_transform_samples(self):
        """Test transforming samples."""
        manager = CorrelationManager(['x', 'y'])
        manager.set_correlation('x', 'y', 0.8)
        
        np.random.seed(42)
        uniform_samples = np.random.uniform(0, 1, (1000, 2))
        transformed = manager.transform_samples(uniform_samples)
        
        assert transformed.shape == uniform_samples.shape
        corr_after = stats.spearmanr(transformed)[0]
        assert corr_after > 0.5  # Should be positively correlated


class TestSamplerWithCorrelations:
    """Test Sampler class with correlations."""
    
    def test_set_correlation(self):
        """Test setting correlation between parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        sampler.set_correlation('x', 'y', 0.7)
        
        assert sampler.has_correlations()
        corr_matrix = sampler.get_correlation_matrix()
        assert corr_matrix[0, 1] == 0.7
    
    def test_set_correlation_matrix(self):
        """Test setting full correlation matrix."""
        sampler = Sampler()
        sampler.add_parameter('a', NormalDistribution(0, 1))
        sampler.add_parameter('b', NormalDistribution(0, 1))
        sampler.add_parameter('c', NormalDistribution(0, 1))
        
        corr = np.array([
            [1.0, 0.6, -0.4],
            [0.6, 1.0, 0.3],
            [-0.4, 0.3, 1.0]
        ])
        sampler.set_correlation_matrix(corr)
        
        np.testing.assert_array_equal(sampler.get_correlation_matrix(), corr)
    
    def test_generate_with_correlation_random(self):
        """Test random sampling with correlations."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        sampler.set_correlation('x', 'y', 0.8)
        
        samples = sampler.generate(n=2000, method='random')
        
        # Check correlation is preserved
        corr = stats.spearmanr(samples['x'], samples['y'])[0]
        expected = (6 / np.pi) * np.arcsin(0.8 / 2)
        assert abs(corr - expected) < 0.1
    
    def test_generate_with_correlation_sobol(self):
        """Test Sobol sampling with correlations."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        sampler.set_correlation('x', 'y', 0.7)
        
        samples = sampler.generate(n=2000, method='sobol')
        
        # Check correlation is preserved
        corr = stats.spearmanr(samples['x'], samples['y'])[0]
        expected = (6 / np.pi) * np.arcsin(0.7 / 2)
        assert abs(corr - expected) < 0.15
    
    def test_generate_with_correlation_different_distributions(self):
        """Test correlations work with different distribution types."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('uniform', UniformDistribution(0, 10))
        sampler.add_parameter('normal', NormalDistribution(50, 10))
        sampler.add_parameter('exponential', ExponentialDistribution(2.0))
        
        # Set correlations
        sampler.set_correlation('uniform', 'normal', 0.6)
        sampler.set_correlation('normal', 'exponential', -0.4)
        
        samples = sampler.generate(n=3000, method='random')
        
        # Check correlations exist
        corr_12 = stats.spearmanr(samples['uniform'], samples['normal'])[0]
        corr_23 = stats.spearmanr(samples['normal'], samples['exponential'])[0]
        
        assert corr_12 > 0.3  # Positive
        assert corr_23 < -0.2  # Negative
    
    def test_generate_without_correlation(self):
        """Test that samples are independent when no correlation set."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        
        samples = sampler.generate(n=1000, method='random')
        
        # Should be nearly uncorrelated
        corr = stats.spearmanr(samples['x'], samples['y'])[0]
        assert abs(corr) < 0.1
    
    def test_correlation_preserved_dataframe_format(self):
        """Test correlations work with DataFrame output."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('x', NormalDistribution(0, 1))
        sampler.add_parameter('y', NormalDistribution(0, 1))
        sampler.set_correlation('x', 'y', 0.75)
        
        df = sampler.generate(n=2000, return_type='dataframe', method='random')
        
        # Check correlation in DataFrame
        corr = stats.spearmanr(df['x'], df['y'])[0]
        expected = (6 / np.pi) * np.arcsin(0.75 / 2)
        assert abs(corr - expected) < 0.1
    
    def test_correlation_preserved_array_format(self):
        """Test correlations work with array output."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('a', UniformDistribution(0, 1))
        sampler.add_parameter('b', UniformDistribution(0, 1))
        sampler.set_correlation('a', 'b', 0.65)
        
        arr = sampler.generate(n=2000, return_type='array', method='random')
        
        # Check correlation in array
        corr = stats.spearmanr(arr[:, 0], arr[:, 1])[0]
        expected = (6 / np.pi) * np.arcsin(0.65 / 2)
        assert abs(corr - expected) < 0.1
    
    def test_repr_with_correlations(self):
        """Test that repr shows correlation info."""
        sampler = Sampler()
        sampler.add_parameter('x', UniformDistribution(0, 1))
        sampler.add_parameter('y', UniformDistribution(0, 1))
        sampler.set_correlation('x', 'y', 0.5)
        
        repr_str = repr(sampler)
        assert 'Correlations' in repr_str or 'correlation' in repr_str
    
    def test_method_chaining_with_correlations(self):
        """Test that correlation methods support chaining."""
        sampler = (Sampler()
                   .add_parameter('x', UniformDistribution(0, 1))
                   .add_parameter('y', UniformDistribution(0, 1))
                   .set_correlation('x', 'y', 0.7))
        
        assert sampler.has_correlations()
        samples = sampler.generate(n=100)
        assert 'x' in samples
        assert 'y' in samples
