"""Tests for derived parameters functionality."""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import (
    NormalDistribution,
    UniformDistribution,
    ConstantDistribution,
)


class TestDerivedParametersBasics:
    """Test basic derived parameter functionality."""
    
    def test_add_derived_parameter_string_formula(self):
        """Test adding derived parameter with string formula."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(5, 2))
        
        # Add derived parameter
        sampler.add_derived_parameter("Z", formula="X * Y")
        
        assert "Z" in sampler._derived_parameters
        assert sampler._derived_parameters["Z"]["type"] == "string"
        assert sampler._derived_parameters["Z"]["formula"] == "X * Y"
    
    def test_add_derived_parameter_callable(self):
        """Test adding derived parameter with callable."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(5, 2))
        
        # Add derived parameter with callable
        sampler.add_derived_parameter(
            "Z",
            formula=lambda s: s["X"] * s["Y"],
            depends_on=["X", "Y"]
        )
        
        assert "Z" in sampler._derived_parameters
        assert sampler._derived_parameters["Z"]["type"] == "callable"
        assert callable(sampler._derived_parameters["Z"]["formula"])
    
    def test_infer_dependencies_from_formula(self):
        """Test that dependencies are inferred from string formula."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(5, 2))
        
        sampler.add_derived_parameter("Z", formula="X * Y + X**2")
        
        deps = sampler._derived_parameters["Z"]["depends_on"]
        assert set(deps) == {"X", "Y"}
    
    def test_derived_parameter_duplicate_name_error(self):
        """Test that duplicate names raise error."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        # Cannot add derived parameter with same name as base parameter
        with pytest.raises(ValueError, match="already exists"):
            sampler.add_derived_parameter("X", formula="X * 2")
        
        # Cannot add two derived parameters with same name
        sampler.add_derived_parameter("Z", formula="X * 2")
        with pytest.raises(ValueError, match="already exists"):
            sampler.add_derived_parameter("Z", formula="X * 3")
    
    def test_derived_parameter_missing_dependency(self):
        """Test that missing dependencies raise error at generation."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        # Reference non-existent parameter - error raised at generation time
        sampler.add_derived_parameter("Z", formula="X * Y")
        
        with pytest.raises(ValueError, match="not defined"):
            sampler.generate(n=100)
    
    def test_callable_requires_depends_on(self):
        """Test that callable requires explicit depends_on."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        with pytest.raises(ValueError, match="must be explicitly provided"):
            sampler.add_derived_parameter(
                "Z",
                formula=lambda s: s["X"] * 2
                # Missing depends_on
            )


class TestDerivedParametersComputation:
    """Test computation of derived parameters."""
    
    def test_simple_arithmetic_operations(self):
        """Test simple arithmetic formulas."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(mean=10, std=2))
        sampler.add_parameter("Y", NormalDistribution(mean=5, std=1))
        
        # Test various operations
        sampler.add_derived_parameter("sum", formula="X + Y")
        sampler.add_derived_parameter("diff", formula="X - Y")
        sampler.add_derived_parameter("product", formula="X * Y")
        sampler.add_derived_parameter("ratio", formula="X / Y")
        sampler.add_derived_parameter("power", formula="X**2")
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        # Verify computations
        np.testing.assert_allclose(samples["sum"], samples["X"] + samples["Y"])
        np.testing.assert_allclose(samples["diff"], samples["X"] - samples["Y"])
        np.testing.assert_allclose(samples["product"], samples["X"] * samples["Y"])
        np.testing.assert_allclose(samples["ratio"], samples["X"] / samples["Y"])
        np.testing.assert_allclose(samples["power"], samples["X"]**2)
    
    def test_numpy_functions_in_formula(self):
        """Test using numpy functions in formulas."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", UniformDistribution(low=1, high=10))
        
        sampler.add_derived_parameter("log_X", formula="np.log(X)")
        sampler.add_derived_parameter("sqrt_X", formula="np.sqrt(X)")
        sampler.add_derived_parameter("exp_X", formula="np.exp(X)")
        sampler.add_derived_parameter("sin_X", formula="np.sin(X)")
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        # Verify computations
        np.testing.assert_allclose(samples["log_X"], np.log(samples["X"]))
        np.testing.assert_allclose(samples["sqrt_X"], np.sqrt(samples["X"]))
        np.testing.assert_allclose(samples["exp_X"], np.exp(samples["X"]))
        np.testing.assert_allclose(samples["sin_X"], np.sin(samples["X"]))
    
    def test_complex_formula(self):
        """Test complex formula with multiple operations."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(5, 1))
        sampler.add_parameter("Y", NormalDistribution(10, 2))
        
        # Complex formula
        sampler.add_derived_parameter(
            "complex",
            formula="np.sqrt(X**2 + Y**2) + np.log(np.abs(X * Y) + 1)"
        )
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        expected = np.sqrt(samples["X"]**2 + samples["Y"]**2) + \
                   np.log(np.abs(samples["X"] * samples["Y"]) + 1)
        np.testing.assert_allclose(samples["complex"], expected)
    
    def test_callable_formula(self):
        """Test callable formula."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(0, 1))
        
        # Callable that computes Euclidean distance from origin
        sampler.add_derived_parameter(
            "distance",
            formula=lambda s: np.sqrt(s["X"]**2 + s["Y"]**2),
            depends_on=["X", "Y"]
        )
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        expected = np.sqrt(samples["X"]**2 + samples["Y"]**2)
        np.testing.assert_allclose(samples["distance"], expected)
    
    def test_column_order_in_dataframe(self):
        """Test that columns are ordered: base params, then derived."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("A", NormalDistribution(0, 1))
        sampler.add_parameter("B", NormalDistribution(0, 1))
        sampler.add_derived_parameter("C", formula="A + B")
        sampler.add_derived_parameter("D", formula="A * B")
        
        df = sampler.generate(n=1000, return_type='dataframe')
        
        # Check column order
        assert list(df.columns) == ["A", "B", "C", "D"]
    
    def test_dict_return_type(self):
        """Test that dict return includes derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_derived_parameter("Y", formula="X**2")
        
        samples = sampler.generate(return_type='dict')
        
        assert "X" in samples
        assert "Y" in samples
        np.testing.assert_allclose(samples["Y"], samples["X"]**2)
    
    def test_array_return_type(self):
        """Test that array return includes derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_derived_parameter("Y", formula="X * 2")
        
        samples = sampler.generate(n=100, return_type='array')
        
        # Should have 2 columns
        assert samples.shape == (100, 2)
        # Second column should be 2 * first column
        np.testing.assert_allclose(samples[:, 1], samples[:, 0] * 2)


class TestDerivedParametersDependencies:
    """Test dependency handling for derived parameters."""
    
    def test_chained_dependencies(self):
        """Test derived parameters that depend on other derived parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(5, 1))
        
        # Chain: X -> Y -> Z
        sampler.add_derived_parameter("Y", formula="X**2")
        sampler.add_derived_parameter("Z", formula="Y + X")  # Depends on both Y and X
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        np.testing.assert_allclose(samples["Y"], samples["X"]**2)
        np.testing.assert_allclose(samples["Z"], samples["Y"] + samples["X"])
    
    def test_dependency_ordering(self):
        """Test that derived parameters are computed in correct order."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("A", NormalDistribution(0, 1))
        
        # Define B first, then C (which depends on B)
        sampler.add_derived_parameter("B", formula="A + 1")  # Depends on A
        sampler.add_derived_parameter("C", formula="B * 2")  # Depends on B
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        np.testing.assert_allclose(samples["B"], samples["A"] + 1)
        np.testing.assert_allclose(samples["C"], samples["B"] * 2)
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        # Create circular dependency manually (bypassing normal validation)
        sampler._derived_parameters["A"] = {
            'formula': "B + 1",
            'depends_on': ["B"],
            'type': 'string'
        }
        sampler._derived_parameters["B"] = {
            'formula': "A + 1",
            'depends_on': ["A"],
            'type': 'string'
        }
        sampler._derived_order = ["A", "B"]
        
        # Should raise error when generating
        with pytest.raises(ValueError, match="Circular dependencies"):
            sampler._sort_derived_parameters()
    
    def test_multiple_independent_derived_parameters(self):
        """Test multiple derived parameters with no dependencies between them."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(5, 2))
        
        # Multiple independent derived parameters
        sampler.add_derived_parameter("X_squared", formula="X**2")
        sampler.add_derived_parameter("Y_squared", formula="Y**2")
        sampler.add_derived_parameter("sum", formula="X + Y")
        sampler.add_derived_parameter("product", formula="X * Y")
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        np.testing.assert_allclose(samples["X_squared"], samples["X"]**2)
        np.testing.assert_allclose(samples["Y_squared"], samples["Y"]**2)
        np.testing.assert_allclose(samples["sum"], samples["X"] + samples["Y"])
        np.testing.assert_allclose(samples["product"], samples["X"] * samples["Y"])


class TestDerivedParametersIntegration:
    """Test integration with other sampler features."""
    
    def test_derived_with_correlations(self):
        """Test derived parameters work with correlated base parameters."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_parameter("Y", NormalDistribution(0, 1))
        
        # Add correlation
        sampler.set_correlation("X", "Y", 0.7)
        
        # Add derived parameter
        sampler.add_derived_parameter("Z", formula="X + Y")
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        # Check that X and Y are correlated
        corr_xy = spearmanr(samples["X"], samples["Y"])[0]
        assert abs(corr_xy - 0.7) < 0.1
        
        # Check that Z is computed correctly
        np.testing.assert_allclose(samples["Z"], samples["X"] + samples["Y"])
        
        # Z should be highly correlated with both X and Y
        corr_xz = spearmanr(samples["X"], samples["Z"])[0]
        corr_yz = spearmanr(samples["Y"], samples["Z"])[0]
        assert corr_xz > 0.5
        assert corr_yz > 0.5
    
    def test_derived_with_qmc_methods(self):
        """Test derived parameters work with QMC sampling."""
        for method in ['sobol', 'halton', 'lhs']:
            sampler = Sampler(random_seed=42)
            sampler.add_parameter("X", UniformDistribution(0, 10))
            sampler.add_parameter("Y", UniformDistribution(0, 10))
            
            sampler.add_derived_parameter("product", formula="X * Y")
            
            n = 1024 if method == 'sobol' else 1000
            samples = sampler.generate(n=n, method=method, return_type='dataframe')
            
            np.testing.assert_allclose(
                samples["product"],
                samples["X"] * samples["Y"]
            )
    
    def test_method_chaining_with_derived(self):
        """Test method chaining with derived parameters."""
        samples = (Sampler(random_seed=42)
            .add_parameter("X", NormalDistribution(0, 1))
            .add_parameter("Y", NormalDistribution(5, 2))
            .add_derived_parameter("Z", formula="X * Y")
            .generate(n=1000, return_type='dataframe'))
        
        assert list(samples.columns) == ["X", "Y", "Z"]
        np.testing.assert_allclose(samples["Z"], samples["X"] * samples["Y"])
    
    def test_repr_with_derived_parameters(self):
        """Test that __repr__ shows derived parameters."""
        sampler = Sampler()
        sampler.add_parameter("X", NormalDistribution(0, 1))
        sampler.add_derived_parameter("Y", formula="X**2")
        sampler.add_derived_parameter("Z", formula=lambda s: s["X"] * 2, depends_on=["X"])
        
        repr_str = repr(sampler)
        
        assert "Derived Parameters:" in repr_str
        assert "Y = X**2" in repr_str
        assert "Z = <callable>" in repr_str


class TestDerivedParametersErrorHandling:
    """Test error handling for derived parameters."""
    
    def test_invalid_formula_syntax(self):
        """Test that invalid formula syntax is caught."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        sampler.add_derived_parameter("Y", formula="X + +")  # Invalid syntax
        
        with pytest.raises(ValueError, match="Error evaluating formula"):
            sampler.generate()
    
    def test_callable_error(self):
        """Test that errors in callable are caught."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        def buggy_function(samples):
            return samples["NonExistent"]  # KeyError
        
        sampler.add_derived_parameter(
            "Y",
            formula=buggy_function,
            depends_on=["X"]
        )
        
        with pytest.raises(ValueError, match="Error computing derived parameter"):
            sampler.generate()
    
    def test_no_formula_provided(self):
        """Test that formula is required."""
        sampler = Sampler(random_seed=42)
        sampler.add_parameter("X", NormalDistribution(0, 1))
        
        with pytest.raises(ValueError, match="Formula must be provided"):
            sampler.add_derived_parameter("Y")  # No formula


class TestDerivedParametersUseCase:
    """Test realistic use cases for derived parameters."""
    
    def test_physical_system_example(self):
        """Test modeling a physical system with derived quantities."""
        sampler = Sampler(random_seed=42)
        
        # Base parameters
        sampler.add_parameter("temperature", NormalDistribution(mean=300, std=10))  # Kelvin
        sampler.add_parameter("pressure", NormalDistribution(mean=101325, std=1000))  # Pascal
        
        # Derived: Ideal gas law (assuming constant n*R/V)
        # P*V = n*R*T => V = n*R*T/P
        sampler.add_derived_parameter(
            "volume_normalized",
            formula="temperature / pressure"  # Proportional to volume
        )
        
        # Derived: Energy (proportional to temperature)
        sampler.add_derived_parameter("energy", formula="temperature * 1.5")
        
        samples = sampler.generate(n=1000, return_type='dataframe')
        
        assert all(samples["temperature"] > 0)
        assert all(samples["pressure"] > 0)
        np.testing.assert_allclose(
            samples["volume_normalized"],
            samples["temperature"] / samples["pressure"]
        )
    
    def test_financial_portfolio_example(self):
        """Test financial portfolio with derived metrics."""
        sampler = Sampler(random_seed=42)
        
        # Asset returns (%)
        sampler.add_parameter("stock_return", NormalDistribution(mean=8, std=15))
        sampler.add_parameter("bond_return", NormalDistribution(mean=3, std=5))
        
        # Portfolio allocation (constant for this test)
        stock_weight = 0.6
        
        # Derived: Portfolio return
        sampler.add_derived_parameter(
            "portfolio_return",
            formula=f"stock_return * {stock_weight} + bond_return * {1-stock_weight}"
        )
        
        # Correlate the returns
        sampler.set_correlation("stock_return", "bond_return", -0.3)
        
        samples = sampler.generate(n=2000, return_type='dataframe')
        
        # Verify portfolio return calculation
        expected_return = (samples["stock_return"] * stock_weight +
                          samples["bond_return"] * (1-stock_weight))
        np.testing.assert_allclose(samples["portfolio_return"], expected_return)
