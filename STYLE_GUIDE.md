# Style Guide & Coding Standards

## Code Formatting

### General Principles

- Write clear, readable, and maintainable code
- Favor explicitness over cleverness
- Keep functions and methods focused and small
- Follow DRY (Don't Repeat Yourself) principle
- Use meaningful variable and function names

### Indentation & Spacing

- **Indentation:** 4 spaces
- **Line Length:** Maximum 120 characters
- **Blank Lines:** Use blank lines to separate logical sections

### Naming Conventions

- **Variables:** camelCase
  - Example: `userCount`, `total_items`
- **Functions/Methods:** snake_case
  - Example: `calculateTotal()`, `process_data()`
- **Classes:** PascalCase
  - Example: `DataManager`, `UserService`
- **Constants:** UPPER_SNAKE_CASE
  - Example: `MAX_RETRY_COUNT`, `API_BASE_URL`
- **Private Members:**  _leadingUnderscore
  - Example: `_privateMethod()`, `_internalState`

## Comments & Documentation

### When to Comment

- Complex algorithms that aren't immediately obvious
- Business logic that requires context
- Workarounds or non-obvious solutions
- Public APIs and interfaces

### Comment Style

```
# Single-line comments for brief explanations

"""
Multi-line comments or docstrings for detailed explanations,
function documentation, and module descriptions.
"""
```

### Function/Method Documentation

```
function exampleFunction(param1, param2):
    """
    Brief description of what the function does.
  
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
  
    Returns:
        type: Description of return value
  
    Raises:
        ExceptionType: When this exception is raised
    """
```

### Manual Files

Keep a software manual that details:

1. Internal functions that are used throughout the code.
2. External functions that are used in the software api
3. GUI functionality.

## File Organization

### File Structure

1. Imports/Dependencies (grouped logically)
2. Constants
3. Type definitions/Interfaces
4. Main class/function definitions
5. Helper functions
6. Exports (if applicable)

### Import Order

1. Standard library imports
2. Third-party library imports
3. Local application imports

## Best Practices

### Error Handling

- Always handle expected errors gracefully
- Use specific exception types
- Log errors with appropriate context
- Don't swallow exceptions silently

### Security

- Never commit sensitive data (API keys, passwords)
- Validate all user input
- Use parameterized queries for database operations
- Sanitize output to prevent injection attacks

### Performance

- Optimize for readability first, performance second
- Profile before optimizing
- Cache expensive operations when appropriate
- Use appropriate data structures

### Testing

- Write unit tests for all business logic
- Aim for 80% code coverage
- Test edge cases and error conditions
- Keep tests independent and repeatable

## Code Review Checklist

- [ ] Code follows naming conventions
- [ ] Functions are appropriately sized and focused
- [ ] Code is properly documented
- [ ] Error handling is implemented
- [ ] Tests are included and passing
- [ ] No sensitive data is exposed
- [ ] Performance considerations addressed
- [ ] Code is readable and maintainable

## Git Conventions

### Commit Messages

Format: `<type>(<scope>): <subject>`

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**

```
feat(auth): add JWT token validation
fix(api): resolve data parsing issue
docs(readme): update installation instructions
```

### Branch Naming

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `refactor/what-is-refactored` - Refactoring
- `docs/documentation-update` - Documentation

## Language-Specific Guidelines

### Python Guidelines
- **Follow PEP 8** style guide with 4-space indentation
- **Type hints required** for all public functions and methods
- **Docstrings required** for all public APIs (NumPy style)
- Use **context managers** for file operations and resource management
- Prefer **NumPy vectorized operations** over Python loops for data processing
- Use **list/dict comprehensions** for concise transformations
- **Avoid external dependencies** when internal implementation is reasonable
  - Favor NumPy/SciPy over specialized libraries when possible
  - Implement algorithms internally for long-term maintainability

### Data Science Specific
- **Document array shapes** in comments: `# shape: (n_samples, n_features)`
- **Validate inputs** early: check array dimensions, data types, NaN handling
- **Memory-aware**: Consider memory footprint for large datasets
- **Reproducibility**: Use random seeds for stochastic operations
- **Numerical stability**: Watch for overflow, underflow, division by zero

### Example Function Template
```python
import numpy as np
from typing import Optional

def process_data(
    data: np.ndarray,
    threshold: float,
    normalize: bool = True,
    axis: Optional[int] = None
) -> np.ndarray:
    """
    Process input data with normalization and thresholding.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array, shape (n_samples, n_features)
    threshold : float
        Threshold value for filtering
    normalize : bool, optional
        Whether to normalize data, by default True
    axis : int or None, optional
        Axis along which to operate, by default None
    
    Returns
    -------
    np.ndarray
        Processed data array, same shape as input
    
    Raises
    ------
    ValueError
        If data contains NaN or infinite values
    """
    # Validate input
    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or infinite values")
    
    # Process
    result = data.copy()
    if normalize:
        result = (result - result.mean(axis=axis)) / result.std(axis=axis)
    result[result < threshold] = 0
    
    return result
```

## Tools & Linters
- **Formatter:** Black (line length: 120)
- **Linter:** Flake8 with extensions:
  - flake8-docstrings (enforce docstrings)
  - flake8-type-checking (validate type hints)
  - flake8-bugbear (catch common bugs)
- **Type Checker:** mypy (strict mode)
- **Import Sorter:** isort (compatible with Black)
- **Security:** bandit (scan for security issues)
- **Pre-commit Hooks:**
  - Black formatting
  - Flake8 linting
  - mypy type checking
  - pytest (run unit tests)
  - No large files or binary data in commits

## Resources
