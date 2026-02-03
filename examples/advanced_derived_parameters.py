"""
Advanced Formula Examples - Showing the full flexibility of derived parameters.
"""

import numpy as np
from multifolio.core.sampling import Sampler
from multifolio.core.sampling.distributions import NormalDistribution, UniformDistribution

print("="*70)
print("ADVANCED FORMULA EXAMPLES")
print("="*70)

# Example 1: Complex nested formulas with conditionals
print("\n1. Complex String Formulas with np.where (conditionals)")
print("-" * 70)

sampler1 = Sampler(random_seed=42)
sampler1.add_parameter("temperature", NormalDistribution(mean=25, std=10))
sampler1.add_parameter("humidity", UniformDistribution(low=0, high=100))

# Complex formula with conditional logic using np.where
sampler1.add_derived_parameter(
    "comfort_index",
    formula="np.where(temperature < 20, "
            "temperature * 0.5 + humidity * 0.1, "  # Cold: weight temp more
            "np.where(temperature > 30, "
            "temperature * 0.3 + humidity * 0.3, "  # Hot: weight both
            "temperature * 0.4 + humidity * 0.2))"  # Moderate
)

samples1 = sampler1.generate(n=5, return_type='dataframe')
print(samples1[['temperature', 'humidity', 'comfort_index']])

# Example 2: Using your own defined functions in formulas
print("\n2. Using Custom Defined Functions in Formulas")
print("-" * 70)

sampler2 = Sampler(random_seed=42)
sampler2.add_parameter("x", UniformDistribution(low=-10, high=10))
sampler2.add_parameter("y", UniformDistribution(low=-10, high=10))

# Define custom functions
def custom_sigmoid(z):
    """Custom sigmoid function."""
    return 1 / (1 + np.exp(-z))

def custom_rbf(x, y, center_x=0, center_y=0, sigma=1):
    """Radial basis function."""
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    return np.exp(-dist_sq / (2 * sigma**2))

# Use as callable (recommended for complex functions)
sampler2.add_derived_parameter(
    "sigmoid_x",
    formula=lambda s: custom_sigmoid(s["x"]),
    depends_on=["x"]
)

sampler2.add_derived_parameter(
    "rbf_distance",
    formula=lambda s: custom_rbf(s["x"], s["y"], sigma=5),
    depends_on=["x", "y"]
)

samples2 = sampler2.generate(n=5, return_type='dataframe')
print(samples2)

# Example 3: Multi-step calculations with intermediate results
print("\n3. Complex Multi-Step Calculations")
print("-" * 70)

sampler3 = Sampler(random_seed=42)
sampler3.add_parameter("velocity", UniformDistribution(low=0, high=100))  # m/s
sampler3.add_parameter("angle", UniformDistribution(low=0, high=np.pi/2))  # radians

# Projectile motion - chain of derived parameters
sampler3.add_derived_parameter("v_x", formula="velocity * np.cos(angle)")
sampler3.add_derived_parameter("v_y", formula="velocity * np.sin(angle)")

g = 9.81  # gravity
sampler3.add_derived_parameter("time_of_flight", formula=f"2 * v_y / {g}")
sampler3.add_derived_parameter("max_height", formula=f"(v_y**2) / (2 * {g})")
sampler3.add_derived_parameter("range", formula="v_x * time_of_flight")

samples3 = sampler3.generate(n=5, return_type='dataframe')
print(samples3[['velocity', 'angle', 'range', 'max_height', 'time_of_flight']])

# Example 4: Statistical transformations
print("\n4. Statistical Transformations (Percentiles, Ranks)")
print("-" * 70)

sampler4 = Sampler(random_seed=42)
sampler4.add_parameter("score", NormalDistribution(mean=75, std=15))

# Complex lambda with percentile ranking
def percentile_rank(samples):
    """Convert scores to percentile ranks."""
    scores = samples["score"]
    ranks = np.argsort(np.argsort(scores)) + 1
    percentiles = (ranks / len(scores)) * 100
    return percentiles

sampler4.add_derived_parameter(
    "percentile",
    formula=percentile_rank,
    depends_on=["score"]
)

# Grade assignment using complex logic
def assign_grade(samples):
    """Assign letter grades based on percentile."""
    p = samples["percentile"]
    return np.select(
        [p >= 90, p >= 80, p >= 70, p >= 60, p < 60],
        ['A', 'B', 'C', 'D', 'F'],
        default='F'
    )

sampler4.add_derived_parameter(
    "grade",
    formula=assign_grade,
    depends_on=["percentile"]
)

samples4 = sampler4.generate(n=10, return_type='dataframe')
print(samples4.sort_values('score'))

# Example 5: Using external libraries in formulas
print("\n5. Using External Library Functions (scipy.stats)")
print("-" * 70)

from scipy import stats

sampler5 = Sampler(random_seed=42)
sampler5.add_parameter("data_point", NormalDistribution(mean=0, std=1))

# Use scipy functions in callable
sampler5.add_derived_parameter(
    "cdf_value",
    formula=lambda s: stats.norm.cdf(s["data_point"]),
    depends_on=["data_point"]
)

sampler5.add_derived_parameter(
    "z_score_category",
    formula=lambda s: np.select(
        [np.abs(s["data_point"]) < 1, 
         np.abs(s["data_point"]) < 2,
         np.abs(s["data_point"]) >= 2],
        ['common', 'uncommon', 'rare']
    ),
    depends_on=["data_point"]
)

samples5 = sampler5.generate(n=8, return_type='dataframe')
print(samples5)

# Example 6: Machine Learning Model Predictions
print("\n6. Using ML Models as Formulas")
print("-" * 70)

# Simulate a simple trained model (in reality, you'd load a real model)
def simple_ml_model(features):
    """
    Simulate a trained ML model.
    In practice, this could be sklearn, tensorflow, pytorch, etc.
    """
    # Simple weighted sum + nonlinearity (like a neural network)
    weights = np.array([0.3, 0.7, -0.2])
    bias = 0.5
    linear = np.dot(features, weights) + bias
    return 1 / (1 + np.exp(-linear))  # sigmoid activation

sampler6 = Sampler(random_seed=42)
sampler6.add_parameter("feature1", NormalDistribution(0, 1))
sampler6.add_parameter("feature2", NormalDistribution(0, 1))
sampler6.add_parameter("feature3", NormalDistribution(0, 1))

# Use ML model as derived parameter
def ml_prediction(samples):
    features = np.column_stack([
        samples["feature1"],
        samples["feature2"],
        samples["feature3"]
    ])
    return simple_ml_model(features)

sampler6.add_derived_parameter(
    "prediction",
    formula=ml_prediction,
    depends_on=["feature1", "feature2", "feature3"]
)

# Decision based on prediction
sampler6.add_derived_parameter(
    "decision",
    formula=lambda s: np.where(s["prediction"] > 0.5, "Accept", "Reject"),
    depends_on=["prediction"]
)

samples6 = sampler6.generate(n=5, return_type='dataframe')
print(samples6)

# Example 7: Time-series-like dependencies
print("\n7. Complex Domain-Specific Calculations")
print("-" * 70)

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option pricing.
    S: stock price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

sampler7 = Sampler(random_seed=42)
sampler7.add_parameter("stock_price", NormalDistribution(mean=100, std=20))
sampler7.add_parameter("volatility", UniformDistribution(low=0.1, high=0.5))

K = 100  # Strike price
T = 1    # 1 year
r = 0.05 # 5% risk-free rate

# Option pricing as derived parameter
sampler7.add_derived_parameter(
    "call_option_price",
    formula=lambda s: black_scholes_call(
        s["stock_price"], K, T, r, s["volatility"]
    ),
    depends_on=["stock_price", "volatility"]
)

# Profit/loss if you buy the option
option_premium = 10  # You paid $10 for the option
sampler7.add_derived_parameter(
    "profit_loss",
    formula=f"call_option_price - {option_premium}",
    depends_on=["call_option_price"]
)

samples7 = sampler7.generate(n=5, return_type='dataframe')
print(samples7)

print("\n" + "="*70)
print("KEY TAKEAWAYS:")
print("="*70)
print("""
1. String formulas: Can use any numpy function, math operations, np.where
2. Callable formulas: Can be ANY Python function - no limits!
3. You can import and use: scipy, sklearn, tensorflow, pytorch, etc.
4. Complex logic: conditionals, percentiles, rankings, classifications
5. Domain models: physics, finance, ML predictions, statistical tests
6. Chaining: Build complex pipelines by chaining derived parameters

Limitations:
- String formulas have restricted namespace (numpy + parameters only)
- For complex logic, use callable formulas (lambdas or def functions)
- Callables must return numpy arrays or array-like objects
""")
