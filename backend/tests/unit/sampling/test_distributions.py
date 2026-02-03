"""Unit tests for distribution classes."""

import pytest
import numpy as np
from multifolio.core.sampling.distributions import (
    UniformDistribution,
    NormalDistribution,
    TruncatedNormalDistribution,
    BetaDistribution,
    ConstantDistribution,
    PoissonDistribution,
    UniformDiscreteDistribution,
)


class TestUniformDistribution:
    """Test continuous uniform distribution."""
    
    def test_initialization(self):
        dist = UniformDistribution(low=0, high=10)
        assert dist.low == 0
        assert dist.high == 10
    
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            UniformDistribution(low=10, high=5)
    
    def test_sample_shape(self):
        dist = UniformDistribution(random_seed=42)
        
        # Single sample
        sample = dist.sample(1)
        assert sample.shape == (1,)
        
        # Multiple samples
        samples = dist.sample(100)
        assert samples.shape == (100,)
        
        # 2D samples
        samples_2d = dist.sample((10, 5))
        assert samples_2d.shape == (10, 5)
    
    def test_sample_range(self):
        dist = UniformDistribution(low=5, high=15, random_seed=42)
        samples = dist.sample(1000)
        
        assert np.all(samples >= 5)
        assert np.all(samples < 15)
        assert np.mean(samples) == pytest.approx(10, abs=0.5)
    
    def test_reproducibility(self):
        dist1 = UniformDistribution(random_seed=42)
        dist2 = UniformDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_equal(samples1, samples2)


class TestNormalDistribution:
    """Test normal distribution."""
    
    def test_initialization(self):
        dist = NormalDistribution(mean=5, std=2)
        assert dist.mean == 5
        assert dist.std == 2
    
    def test_invalid_std(self):
        with pytest.raises(ValueError):
            NormalDistribution(mean=0, std=-1)
        
        with pytest.raises(ValueError):
            NormalDistribution(mean=0, std=0)
    
    def test_sample_statistics(self):
        dist = NormalDistribution(mean=10, std=3, random_seed=42)
        samples = dist.sample(10000)
        
        assert np.mean(samples) == pytest.approx(10, abs=0.1)
        assert np.std(samples) == pytest.approx(3, abs=0.1)
    
    def test_reproducibility(self):
        dist1 = NormalDistribution(random_seed=42)
        dist2 = NormalDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_equal(samples1, samples2)


class TestTruncatedNormalDistribution:
    """Test truncated normal distribution."""
    
    def test_initialization(self):
        dist = TruncatedNormalDistribution(mean=0, std=1, low=-2, high=2)
        assert dist.mean == 0
        assert dist.std == 1
        assert dist.low == -2
        assert dist.high == 2
    
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            TruncatedNormalDistribution(mean=0, std=1, low=5, high=2)
    
    def test_truncation(self):
        dist = TruncatedNormalDistribution(
            mean=0, std=10, low=-1, high=1, random_seed=42
        )
        samples = dist.sample(1000)
        
        # All samples should be within bounds
        assert np.all(samples >= -1)
        assert np.all(samples <= 1)
    
    def test_reproducibility(self):
        dist1 = TruncatedNormalDistribution(random_seed=42)
        dist2 = TruncatedNormalDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_almost_equal(samples1, samples2)


class TestBetaDistribution:
    """Test beta distribution."""
    
    def test_initialization(self):
        dist = BetaDistribution(alpha=2, beta=5)
        assert dist.alpha == 2
        assert dist.beta == 5
        assert dist.low == 0.0
        assert dist.high == 1.0
    
    def test_initialization_with_scaling(self):
        dist = BetaDistribution(alpha=2, beta=5, low=10, high=20)
        assert dist.alpha == 2
        assert dist.beta == 5
        assert dist.low == 10
        assert dist.high == 20
    
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            BetaDistribution(alpha=-1, beta=2)
        
        with pytest.raises(ValueError):
            BetaDistribution(alpha=2, beta=0)
        
        with pytest.raises(ValueError):
            BetaDistribution(alpha=2, beta=2, low=10, high=5)
    
    def test_sample_range_default(self):
        dist = BetaDistribution(alpha=2, beta=2, random_seed=42)
        samples = dist.sample(1000)
        
        # Beta distribution is in [0, 1] by default
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
    
    def test_sample_range_scaled(self):
        dist = BetaDistribution(alpha=2, beta=2, low=10, high=20, random_seed=42)
        samples = dist.sample(1000)
        
        # Should be scaled to [10, 20]
        assert np.all(samples >= 10)
        assert np.all(samples <= 20)
        
        # Mean should be approximately in the middle for alpha=beta
        assert np.mean(samples) == pytest.approx(15, abs=0.5)
    
    def test_reproducibility(self):
        dist1 = BetaDistribution(random_seed=42)
        dist2 = BetaDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_equal(samples1, samples2)
    
    def test_scaling_correctness(self):
        # Test that scaling works correctly
        dist_01 = BetaDistribution(alpha=3, beta=3, low=0, high=1, random_seed=42)
        dist_scaled = BetaDistribution(alpha=3, beta=3, low=100, high=200, random_seed=42)
        
        samples_01 = dist_01.sample(5)
        samples_scaled = dist_scaled.sample(5)
        
        # Manually scale [0,1] samples to [100, 200] and compare
        expected_scaled = 100 + samples_01 * 100
        
        np.testing.assert_array_almost_equal(samples_scaled, expected_scaled)


class TestConstantDistribution:
    """Test constant distribution."""
    
    def test_initialization(self):
        dist = ConstantDistribution(value=42)
        assert dist.value == 42
    
    def test_sample_constant(self):
        dist = ConstantDistribution(value=7.5)
        samples = dist.sample(100)
        
        assert np.all(samples == 7.5)
        assert samples.shape == (100,)
    
    def test_sample_shape(self):
        dist = ConstantDistribution(value=1)
        
        samples_2d = dist.sample((5, 3))
        assert samples_2d.shape == (5, 3)
        assert np.all(samples_2d == 1)


class TestPoissonDistribution:
    """Test Poisson distribution."""
    
    def test_initialization(self):
        dist = PoissonDistribution(lam=5)
        assert dist.lam == 5
    
    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            PoissonDistribution(lam=0)
        
        with pytest.raises(ValueError):
            PoissonDistribution(lam=-1)
    
    def test_sample_properties(self):
        dist = PoissonDistribution(lam=10, random_seed=42)
        samples = dist.sample(10000)
        
        # All samples should be non-negative integers
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))
        
        # Mean should be approximately lambda
        assert np.mean(samples) == pytest.approx(10, abs=0.3)
    
    def test_reproducibility(self):
        dist1 = PoissonDistribution(random_seed=42)
        dist2 = PoissonDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_equal(samples1, samples2)


class TestUniformDiscreteDistribution:
    """Test discrete uniform distribution."""
    
    def test_initialization(self):
        dist = UniformDiscreteDistribution(low=0, high=10)
        assert dist.low == 0
        assert dist.high == 10
    
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            UniformDiscreteDistribution(low=10, high=5)
    
    def test_sample_range(self):
        dist = UniformDiscreteDistribution(low=1, high=6, random_seed=42)
        samples = dist.sample(1000)
        
        # All samples should be integers in [low, high]
        assert np.all(samples >= 1)
        assert np.all(samples <= 6)
        assert np.all(samples == samples.astype(int))
        
        # Should see all values with roughly equal frequency
        unique_values = np.unique(samples)
        assert len(unique_values) == 6  # Values 1, 2, 3, 4, 5, 6
    
    def test_reproducibility(self):
        dist1 = UniformDiscreteDistribution(random_seed=42)
        dist2 = UniformDiscreteDistribution(random_seed=42)
        
        samples1 = dist1.sample(10)
        samples2 = dist2.sample(10)
        
        np.testing.assert_array_equal(samples1, samples2)


class TestDistributionReset:
    """Test random seed reset functionality."""
    
    def test_reset_seed(self):
        dist = UniformDistribution(random_seed=42)
        
        # Generate samples
        samples1 = dist.sample(5)
        
        # Reset with same seed
        dist.reset_seed(42)
        samples2 = dist.sample(5)
        
        # Should get same samples
        np.testing.assert_array_equal(samples1, samples2)
    
    def test_reset_different_seed(self):
        dist = UniformDistribution(random_seed=42)
        
        samples1 = dist.sample(5)
        
        # Reset with different seed
        dist.reset_seed(123)
        samples2 = dist.sample(5)
        
        # Should get different samples
        assert not np.array_equal(samples1, samples2)
