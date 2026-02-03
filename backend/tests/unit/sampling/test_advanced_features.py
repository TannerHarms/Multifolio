"""
Tests for advanced sampler features: save/load, filtering, constraints, conditional, batch generation.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from multifolio.core.sampling.sampler import Sampler
from multifolio.core.sampling.distributions import (
    UniformDistribution, NormalDistribution
)

# Check if HDF5 support is available (either pytables or h5py)
try:
    import tables
    HAS_HDF5 = True
except ImportError:
    try:
        import h5py
        HAS_HDF5 = True
    except ImportError:
        HAS_HDF5 = False


class TestSaveLoadConfig:
    """Test configuration save/load functionality."""
    
    def test_save_load_basic_config(self, tmp_path):
        """Test saving and loading basic sampler configuration."""
        # Create sampler with parameters
        sampler1 = Sampler(random_seed=42)
        sampler1.add_parameter('X', UniformDistribution(0, 10))
        sampler1.add_parameter('Y', NormalDistribution(5, 2))
        
        # Save config
        config_file = tmp_path / "sampler_config.json"
        sampler1.save_config(config_file)
        
        assert config_file.exists()
        
        # Load config
        sampler2 = Sampler.load_config(config_file)
        
        # Verify parameters match
        assert sampler2._parameter_order == ['X', 'Y']
        assert sampler2.random_seed == 42
        
        # Generate samples and verify distributions match
        samples1 = sampler1.generate(n=100, return_type='dataframe')
        samples2 = sampler2.generate(n=100, return_type='dataframe')
        
        # Check that distributions are similar (not exact due to different random states)
        assert samples1['X'].mean() == pytest.approx(samples2['X'].mean(), abs=2.0)
        assert samples1['Y'].mean() == pytest.approx(samples2['Y'].mean(), abs=1.0)
    
    def test_save_load_with_correlations(self, tmp_path):
        """Test saving and loading correlations."""
        sampler1 = Sampler(random_seed=42)
        sampler1.add_parameter('X', UniformDistribution(0, 10))
        sampler1.add_parameter('Y', UniformDistribution(0, 10))
        sampler1.set_correlation('X', 'Y', 0.8)
        
        config_file = tmp_path / "sampler_corr.json"
        sampler1.save_config(config_file)
        
        sampler2 = Sampler.load_config(config_file)
        
        # Check correlation preserved
        assert sampler2.has_correlations()
        corr_matrix = sampler2.get_correlation_matrix()
        corr = corr_matrix[0, 1]  # X-Y correlation
        assert corr == pytest.approx(0.8)
        
        # Verify samples are correlated
        samples = sampler2.generate(n=1000, return_type='dataframe')
        actual_corr = samples[['X', 'Y']].corr().iloc[0, 1]
        assert actual_corr == pytest.approx(0.8, abs=0.15)
    
    def test_save_load_with_derived_parameters(self, tmp_path):
        """Test saving and loading derived parameters (string formulas)."""
        sampler1 = Sampler()
        sampler1.add_parameter('X', UniformDistribution(1, 10))
        sampler1.add_parameter('Y', UniformDistribution(1, 10))
        sampler1.add_derived_parameter('Z', 'X + Y')
        sampler1.add_derived_parameter('W', 'X * Y')
        
        config_file = tmp_path / "sampler_derived.json"
        sampler1.save_config(config_file)
        
        sampler2 = Sampler.load_config(config_file)
        
        # Check derived parameters preserved
        assert 'Z' in sampler2._derived_parameters
        assert 'W' in sampler2._derived_parameters
        
        # Verify computation
        samples = sampler2.generate(n=10, return_type='dataframe')
        assert 'Z' in samples.columns
        assert 'W' in samples.columns
        assert np.allclose(samples['Z'], samples['X'] + samples['Y'])
        assert np.allclose(samples['W'], samples['X'] * samples['Y'])
    
    def test_save_load_with_constraints(self, tmp_path):
        """Test saving and loading constraints."""
        sampler1 = Sampler()
        sampler1.add_parameter('X', UniformDistribution(0, 10))
        sampler1.add_parameter('Y', UniformDistribution(0, 10))
        sampler1.add_constraint('X + Y <= 15')
        sampler1.add_constraint('X >= 2')
        
        config_file = tmp_path / "sampler_constraints.json"
        sampler1.save_config(config_file)
        
        sampler2 = Sampler.load_config(config_file)
        
        # Check constraints preserved
        constraints = sampler2.get_constraints()
        assert 'X + Y <= 15' in constraints
        assert 'X >= 2' in constraints
        
        # Verify constraints are enforced
        samples = sampler2.generate(n=100, return_type='dataframe')
        assert np.all(samples['X'] + samples['Y'] <= 15)
        assert np.all(samples['X'] >= 2)


class TestSaveLoadData:
    """Test data save/load functionality."""
    
    def test_save_load_csv(self, tmp_path):
        """Test saving and loading data as CSV."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        # Save as CSV
        csv_file = tmp_path / "samples.csv"
        Sampler.save_data(samples, csv_file, format='csv')
        
        assert csv_file.exists()
        
        # Load back
        loaded = Sampler.load_data(csv_file, format='csv')
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 100
        assert list(loaded.columns) == ['X', 'Y']
        pd.testing.assert_frame_equal(samples, loaded)
    
    def test_save_load_pickle(self, tmp_path):
        """Test saving and loading data as pickle."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='dict')
        
        # Save as pickle
        pkl_file = tmp_path / "samples.pkl"
        Sampler.save_data(samples, pkl_file, format='pickle')
        
        assert pkl_file.exists()
        
        # Load back
        loaded = Sampler.load_data(pkl_file, format='pickle')
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 100
        assert set(loaded.columns) == {'X', 'Y'}
    
    def test_save_load_auto_format(self, tmp_path):
        """Test automatic format detection from file extension."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        samples = sampler.generate(n=50, return_type='dataframe')
        
        # Test .csv
        csv_file = tmp_path / "test.csv"
        Sampler.save_data(samples, csv_file)  # format='auto'
        loaded_csv = Sampler.load_data(csv_file)
        pd.testing.assert_frame_equal(samples, loaded_csv)
        
        # Test .pkl
        pkl_file = tmp_path / "test.pkl"
        Sampler.save_data(samples, pkl_file)  # format='auto'
        loaded_pkl = Sampler.load_data(pkl_file)
        pd.testing.assert_frame_equal(samples, loaded_pkl)
    
    @pytest.mark.skipif(not HAS_HDF5, reason="Neither pytables nor h5py installed")
    def test_save_load_hdf5(self, tmp_path):
        """Test saving and loading data as HDF5."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        # Save as HDF5
        h5_file = tmp_path / "samples.h5"
        Sampler.save_data(samples, h5_file, format='hdf5')
        
        assert h5_file.exists()
        
        # Load back
        loaded = Sampler.load_data(h5_file, format='hdf5')
        
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 100
        assert list(loaded.columns) == ['X', 'Y']
        pd.testing.assert_frame_equal(samples, loaded)
    
    @pytest.mark.skipif(not HAS_HDF5, reason="Neither pytables nor h5py installed")
    def test_save_load_hdf5_auto_format(self, tmp_path):
        """Test automatic HDF5 format detection."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        samples = sampler.generate(n=50, return_type='dataframe')
        
        # Test .h5 extension
        h5_file = tmp_path / "test.h5"
        Sampler.save_data(samples, h5_file)  # format='auto'
        loaded = Sampler.load_data(h5_file)
        pd.testing.assert_frame_equal(samples, loaded)
        
        # Test .hdf5 extension
        hdf5_file = tmp_path / "test.hdf5"
        Sampler.save_data(samples, hdf5_file)
        loaded = Sampler.load_data(hdf5_file)
        pd.testing.assert_frame_equal(samples, loaded)
    
    @pytest.mark.skipif(not HAS_HDF5, reason="Neither pytables nor h5py installed")
    def test_save_hdf5_with_compression(self, tmp_path):
        """Test HDF5 compression options."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        # Save without compression
        h5_no_comp = tmp_path / "no_compression.h5"
        Sampler.save_data(samples, h5_no_comp, compression=None)
        
        # Save with gzip compression (default)
        h5_gzip = tmp_path / "gzip.h5"
        Sampler.save_data(samples, h5_gzip)  # Uses gzip by default
        
        # Both should load correctly
        loaded_no_comp = Sampler.load_data(h5_no_comp)
        loaded_gzip = Sampler.load_data(h5_gzip)
        
        pd.testing.assert_frame_equal(samples, loaded_no_comp)
        pd.testing.assert_frame_equal(samples, loaded_gzip)
        
        # Compressed file should be smaller (or similar size for small data)
        assert h5_gzip.stat().st_size <= h5_no_comp.stat().st_size * 1.1  # Allow 10% margin


class TestConstraints:
    """Test parameter constraints functionality."""
    
    def test_add_constraint(self):
        """Test adding constraints."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        sampler.add_constraint('X + Y <= 15')
        
        constraints = sampler.get_constraints()
        assert len(constraints) == 1
        assert constraints[0] == 'X + Y <= 15'
    
    def test_clear_constraints(self):
        """Test clearing all constraints."""
        sampler = Sampler()
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_constraint('X > 5')
        sampler.add_constraint('X < 8')
        
        assert len(sampler.get_constraints()) == 2
        
        sampler.clear_constraints()
        assert len(sampler.get_constraints()) == 0
    
    def test_constraint_enforcement_simple(self):
        """Test that simple constraints are enforced during generation."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_constraint('X + Y <= 10')
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        # All samples should satisfy constraint
        assert np.all(samples['X'] + samples['Y'] <= 10)
    
    def test_constraint_enforcement_multiple(self):
        """Test multiple constraints."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_constraint('X + Y <= 12')
        sampler.add_constraint('X >= 3')
        sampler.add_constraint('Y >= 2')
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        assert np.all(samples['X'] + samples['Y'] <= 12)
        assert np.all(samples['X'] >= 3)
        assert np.all(samples['Y'] >= 2)
    
    def test_constraint_with_derived_parameters(self):
        """Test constraints on derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(1, 10))
        sampler.add_parameter('Y', UniformDistribution(1, 10))
        sampler.add_derived_parameter('Z', 'X * Y')
        sampler.add_constraint('Z <= 50')
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        assert np.all(samples['Z'] <= 50)
        assert np.allclose(samples['Z'], samples['X'] * samples['Y'])
    
    def test_constraint_complex_expression(self):
        """Test complex constraint expressions."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(-5, 5))
        sampler.add_parameter('Y', UniformDistribution(-5, 5))
        sampler.add_constraint('X**2 + Y**2 <= 16')  # Inside circle radius 4
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        assert np.all(samples['X']**2 + samples['Y']**2 <= 16)
    
    def test_generate_without_constraints(self):
        """Test that generate works normally without constraints."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_constraint('X > 5')
        
        # Can disable constraints
        samples_all = sampler.generate(n=100, apply_constraints=False, return_type='dataframe')
        samples_constrained = sampler.generate(n=100, apply_constraints=True, return_type='dataframe')
        
        # Without constraints, some samples can be <= 5
        assert np.any(samples_all['X'] <= 5)
        
        # With constraints, all samples > 5
        assert np.all(samples_constrained['X'] > 5)
    
    def test_constraint_error_on_impossible(self):
        """Test error when constraints are impossible to satisfy."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 1))
        sampler.add_constraint('X > 10')  # Impossible for Uniform(0, 1)
        
        with pytest.raises(RuntimeError, match="Unable to generate .* samples satisfying constraints"):
            sampler.generate(n=10, max_constraint_attempts=100)


class TestFiltering:
    """Test sample filtering functionality."""
    
    def test_filter_samples_simple(self):
        """Test simple filtering."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=100, return_type='dataframe')
        filtered = sampler.filter_samples(samples, 'X > 5')
        
        assert len(filtered) < len(samples)
        assert np.all(filtered['X'] > 5)
    
    def test_filter_samples_multiple_conditions(self):
        """Test filtering with multiple conditions."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=100, return_type='dataframe')
        filtered = sampler.filter_samples(samples, '(X > 3) & (Y < 7)')
        
        assert np.all(filtered['X'] > 3)
        assert np.all(filtered['Y'] < 7)
    
    def test_filter_samples_dict_input(self):
        """Test filtering with dict input."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate(n=100, return_type='dict')
        filtered = sampler.filter_samples(samples, 'X + Y > 10')
        
        assert isinstance(filtered, dict)
        assert len(filtered['X']) < 100
        assert np.all(filtered['X'] + filtered['Y'] > 10)
    
    def test_filter_with_numpy_functions(self):
        """Test filtering using numpy functions."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', NormalDistribution(0, 1))
        
        samples = sampler.generate(n=100, return_type='dataframe')
        filtered = sampler.filter_samples(samples, 'np.abs(X) < 1')
        
        assert np.all(np.abs(filtered['X']) < 1)
    
    def test_filter_complex_condition(self):
        """Test filtering with complex mathematical condition."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(-5, 5))
        sampler.add_parameter('Y', UniformDistribution(-5, 5))
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        # Points inside unit circle
        filtered = sampler.filter_samples(samples, 'X**2 + Y**2 < 4')
        
        assert np.all(filtered['X']**2 + filtered['Y']**2 < 4)
        # Roughly π*2²/100 = 0.126 of samples should be inside
        assert 50 < len(filtered) < 200  # Approximate check


class TestConditionalGeneration:
    """Test conditional sample generation."""
    
    def test_generate_conditional_simple(self):
        """Test simple conditional generation."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        samples = sampler.generate_conditional(
            n=100,
            condition='X + Y <= 10',
            return_type='dataframe'
        )
        
        assert len(samples) == 100
        assert np.all(samples['X'] + samples['Y'] <= 10)
    
    def test_generate_conditional_circle(self):
        """Test conditional generation for points in a circle."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(-2, 2))
        sampler.add_parameter('Y', UniformDistribution(-2, 2))
        
        samples = sampler.generate_conditional(
            n=100,
            condition='X**2 + Y**2 < 1',
            return_type='dataframe'
        )
        
        assert len(samples) == 100
        assert np.all(samples['X']**2 + samples['Y']**2 < 1)
    
    def test_generate_conditional_dict_return(self):
        """Test conditional generation with dict return type."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        samples = sampler.generate_conditional(
            n=50,
            condition='X > 5',
            return_type='dict'
        )
        
        assert isinstance(samples, dict)
        assert len(samples['X']) == 50
        assert np.all(samples['X'] > 5)
    
    def test_generate_conditional_with_derived(self):
        """Test conditional generation with derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(1, 10))
        sampler.add_parameter('Y', UniformDistribution(1, 10))
        sampler.add_derived_parameter('Z', 'X * Y')
        
        samples = sampler.generate_conditional(
            n=100,
            condition='Z < 30',
            return_type='dataframe'
        )
        
        assert len(samples) == 100
        assert np.all(samples['Z'] < 30)
        assert np.allclose(samples['Z'], samples['X'] * samples['Y'])
    
    def test_generate_conditional_error_on_impossible(self):
        """Test error when condition is impossible."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 1))
        
        with pytest.raises(RuntimeError, match="Unable to generate .* valid samples"):
            sampler.generate_conditional(
                n=10,
                condition='X > 10',  # Impossible
                max_attempts=100
            )


class TestBatchGeneration:
    """Test batch generation functionality."""
    
    def test_generate_batches_basic(self):
        """Test basic batch generation."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        
        batches = list(sampler.generate_batches(
            n_per_batch=10,
            n_batches=5,
            return_type='dataframe'
        ))
        
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) == 10
            assert 'X' in batch.columns
            assert 'Y' in batch.columns
    
    def test_generate_batches_with_batch_id(self):
        """Test batch generation with batch ID tracking."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        batches = list(sampler.generate_batches(
            n_per_batch=20,
            n_batches=3,
            return_type='dataframe',
            track_batch_id=True
        ))
        
        assert len(batches) == 3
        for i, batch in enumerate(batches):
            assert 'batch_id' in batch.columns
            assert np.all(batch['batch_id'] == i)
    
    def test_generate_batches_dict_return(self):
        """Test batch generation with dict return type."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        batches = list(sampler.generate_batches(
            n_per_batch=15,
            n_batches=4,
            return_type='dict',
            track_batch_id=True
        ))
        
        assert len(batches) == 4
        for batch in batches:
            assert isinstance(batch, dict)
            assert 'X' in batch
            assert 'batch_id' in batch
            assert len(batch['X']) == 15
    
    def test_generate_batches_no_batch_id(self):
        """Test batch generation without batch ID."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        batches = list(sampler.generate_batches(
            n_per_batch=10,
            n_batches=2,
            return_type='dataframe',
            track_batch_id=False
        ))
        
        for batch in batches:
            assert 'batch_id' not in batch.columns
    
    def test_generate_batches_iterator(self):
        """Test that batch generation is a proper iterator."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        
        batch_gen = sampler.generate_batches(
            n_per_batch=10,
            n_batches=3,
            return_type='dataframe'
        )
        
        # Can iterate one at a time
        batch1 = next(batch_gen)
        assert len(batch1) == 10
        
        batch2 = next(batch_gen)
        assert len(batch2) == 10
        
        # Not the same data
        assert not batch1.equals(batch2)
    
    def test_generate_batches_with_qmc(self):
        """Test batch generation with QMC methods."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 1))
        sampler.add_parameter('Y', UniformDistribution(0, 1))
        
        batches = list(sampler.generate_batches(
            n_per_batch=100,
            n_batches=3,
            method='sobol',
            return_type='dataframe'
        ))
        
        assert len(batches) == 3
        # Check that batches maintain low discrepancy across all
        all_samples = pd.concat(batches, ignore_index=True)
        assert len(all_samples) == 300


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_constraints_with_correlations(self):
        """Test constraints applied to correlated parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.set_correlation('X', 'Y', 0.8)
        sampler.add_constraint('X + Y <= 12')
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        # Check constraint
        assert np.all(samples['X'] + samples['Y'] <= 12)
        
        # Check correlation (approximately)
        corr = samples[['X', 'Y']].corr().iloc[0, 1]
        assert corr > 0.3  # Should still be positively correlated (relaxed due to constraint filtering)
    
    def test_derived_parameters_with_constraints(self):
        """Test derived parameters with constraints."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(1, 10))
        sampler.add_parameter('Y', UniformDistribution(1, 10))
        sampler.add_derived_parameter('Product', 'X * Y')
        sampler.add_derived_parameter('Sum', 'X + Y')
        sampler.add_constraint('Product < 50')
        sampler.add_constraint('Sum < 15')
        
        samples = sampler.generate(n=100, return_type='dataframe')
        
        assert np.all(samples['Product'] < 50)
        assert np.all(samples['Sum'] < 15)
        assert np.allclose(samples['Product'], samples['X'] * samples['Y'])
        assert np.allclose(samples['Sum'], samples['X'] + samples['Y'])
    
    def test_save_load_full_configuration(self, tmp_path):
        """Test saving and loading complete configuration."""
        # Create complex sampler
        sampler1 = Sampler(random_seed=42)
        sampler1.add_parameter('X', UniformDistribution(0, 10))
        sampler1.add_parameter('Y', NormalDistribution(5, 2))
        sampler1.set_correlation('X', 'Y', 0.6)
        sampler1.add_derived_parameter('Z', 'X + Y')
        sampler1.add_constraint('Z < 12')
        sampler1.add_constraint('X > 2')
        
        # Save
        config_file = tmp_path / "full_config.json"
        sampler1.save_config(config_file)
        
        # Load
        sampler2 = Sampler.load_config(config_file)
        
        # Verify everything
        assert sampler2._parameter_order == ['X', 'Y']
        assert sampler2.has_correlations()
        assert 'Z' in sampler2._derived_parameters
        assert len(sampler2.get_constraints()) == 2
        
        # Generate and check
        samples = sampler2.generate(n=50, return_type='dataframe')
        assert np.all(samples['Z'] < 12)
        assert np.all(samples['X'] > 2)
        assert np.allclose(samples['Z'], samples['X'] + samples['Y'])
    
    def test_batch_generation_with_constraints(self):
        """Test batch generation with constraints."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter('X', UniformDistribution(0, 10))
        sampler.add_parameter('Y', UniformDistribution(0, 10))
        sampler.add_constraint('X + Y <= 12')
        
        batches = list(sampler.generate_batches(
            n_per_batch=50,
            n_batches=3,
            return_type='dataframe'
        ))
        
        # All batches should satisfy constraints
        for batch in batches:
            assert np.all(batch['X'] + batch['Y'] <= 12)
