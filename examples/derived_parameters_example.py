"""
Example: Derived Parameters in Multifolio

Demonstrates how to create parameters that are computed as functions of other parameters.
"""

import numpy as np
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import NormalDistribution, UniformDistribution

print("="*70)
print("DERIVED PARAMETERS EXAMPLE")
print("="*70)

# Example 1: Basic arithmetic operations
print("\n1. Basic Arithmetic Operations")
print("-" * 70)

sampler = Sampler(random_seed=42)
sampler.add_parameter("X", NormalDistribution(mean=10, std=2))
sampler.add_parameter("Y", NormalDistribution(mean=5, std=1))

# Derive new parameters from X and Y
sampler.add_derived_parameter("sum", formula="X + Y")
sampler.add_derived_parameter("product", formula="X * Y")
sampler.add_derived_parameter("ratio", formula="X / Y")

print(sampler)
print()

samples = sampler.generate(n=5, return_type='dataframe')
print(samples)

# Example 2: Using numpy functions
print("\n2. Using Numpy Functions")
print("-" * 70)

sampler2 = Sampler(random_seed=42)
sampler2.add_parameter("temperature", NormalDistribution(mean=300, std=10))
sampler2.add_parameter("pressure", NormalDistribution(mean=101325, std=1000))

# Derived parameters with numpy functions
sampler2.add_derived_parameter("log_temp", formula="np.log(temperature)")
sampler2.add_derived_parameter("sqrt_pressure", formula="np.sqrt(pressure)")
sampler2.add_derived_parameter("energy", formula="temperature * pressure")

samples2 = sampler2.generate(n=5, return_type='dataframe')
print(samples2[['temperature', 'log_temp', 'pressure', 'sqrt_pressure', 'energy']])

# Example 3: Chained dependencies
print("\n3. Chained Dependencies")
print("-" * 70)

sampler3 = Sampler(random_seed=42)
sampler3.add_parameter("radius", UniformDistribution(low=1, high=10))

# Chain: radius -> area -> volume
sampler3.add_derived_parameter("area", formula="np.pi * radius**2")
sampler3.add_derived_parameter("circumference", formula="2 * np.pi * radius")
sampler3.add_derived_parameter("volume", formula="(4/3) * np.pi * radius**3")

samples3 = sampler3.generate(n=5, return_type='dataframe')
print(samples3)

# Example 4: Using callable functions
print("\n4. Using Callable Functions")
print("-" * 70)

sampler4 = Sampler(random_seed=42)
sampler4.add_parameter("x", NormalDistribution(0, 1))
sampler4.add_parameter("y", NormalDistribution(0, 1))

# Custom function for distance from origin
def euclidean_distance(samples):
    return np.sqrt(samples["x"]**2 + samples["y"]**2)

sampler4.add_derived_parameter(
    "distance",
    formula=euclidean_distance,
    depends_on=["x", "y"]
)

samples4 = sampler4.generate(n=5, return_type='dataframe')
print(samples4)

# Example 5: Real-world example - Ideal Gas Law
print("\n5. Real-World Example: Ideal Gas Law")
print("-" * 70)

sampler5 = Sampler(random_seed=42)

# Base parameters
sampler5.add_parameter("temperature", NormalDistribution(mean=300, std=20))  # Kelvin
sampler5.add_parameter("pressure", NormalDistribution(mean=101325, std=5000))  # Pascal
sampler5.add_parameter("n_moles", UniformDistribution(low=1, high=5))  # moles

# Gas constant
R = 8.314  # J/(mol·K)

# Derived: Volume from ideal gas law (PV = nRT)
sampler5.add_derived_parameter(
    "volume",
    formula=f"{R} * n_moles * temperature / pressure"
)

# Derived: Density (assuming molecular mass of 29 g/mol for air)
M = 0.029  # kg/mol
sampler5.add_derived_parameter(
    "density",
    formula=f"{M} * n_moles / volume"
)

samples5 = sampler5.generate(n=5, return_type='dataframe')
print(samples5)

print("\n" + "="*70)
print("Examples completed successfully!")
print("="*70)
