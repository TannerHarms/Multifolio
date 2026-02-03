# Project Goals

## Project Overview

**Project Name:** Multifolio
**Purpose:** Multifolio provides seamless tools for interrogating and generating tabular datasets that are accessible to analysts of all categories, regardless of their coding experience.  

## Primary Objectives

1. **Objective 1:** Tabular data interpretation
   1. Read, modify, visualize, and interrogate tabular data.
   2. Include tools for dealing with many datatypes (e.g., table cells may include strings, ints, floats, arrays, etc.)
   3. Support graph/network visualization where adjacency relationships can be functionally defined by the user.
2. **Objective 2:** Experimental parameter sampling
   1. Generate parametric samples from common or custom-defined density functions.
   2. Optionally train ML models to sample from the distribution of a dataset (for appropriate data types).
3. **Objective 3:** Maintain an api for users to integrate code into software
4. **Objective 4:** Maintain multiple GUI options for different use cases
   1. Desktop GUI (PyQt6) for standalone application use
   2. Web GUI (React + D3, future) for browser-based access and maximum visualization flexibility  

## Target Audience

- Analysts who frequently need to extract insights from tabular data.
- Scientists and statisticians who need to build mathematical and statistical models based on parametric variations.

## Key Features

- [ ] Feature 1: Data analysis GUI with interactive visualization
  - Click on data points to inspect underlying parameters
  - Support for graph/network visualization with custom adjacency
- [ ] Feature 2: Statistical parameter sampling API
- [ ] Feature 3: Statistical ML learning module
- [ ] Feature 4: Multiple data source support (files, SQL databases)

## Success Metrics

- **Metric 1:** Flexibility - Ability to use multiple file types and data types
- **Metric 2:** Speed - Should be able to handle large datasets (up to millions of cells) quickly
- **Metric 3:** Visualization quality - Should be able to generate presentation-quality figures.  Should give extensive flexibility in data interrogation and visualization. Interactive features (tooltips, click-to-inspect, zoom) are essential.

## Constraints & Requirements

### Technical Requirements

- **Programming language:** Python 3.9+
  - Justification: Rich data science ecosystem, rapid development
  - Performance mitigation: NumPy vectorization, Numba JIT, multiprocessing
  - Option to rewrite critical sections in Cython/Rust if needed
- **Data sources:** File-based (CSV, Excel, Parquet, HDF5) + optional SQL database support
  - SQL support for large datasets that exceed memory
  - File-based for portability and simplicity
- **Visualization:** Interactive plotting library (Plotly or PyQtGraph)
  - Must support click events to inspect data point parameters
  - Must handle graph/network visualization
- **API requirements:** Pythonic, fluent API with type hints
- **Performance target:** Handle datasets with millions of cells with <1s response time
- Keep code robust to changes
  - Favor internal implementations.
  - Avoid using bespoke libraries.

### Business Requirements

- [Timeline constraints]
- [Budget constraints]
- [Compliance requirements]

## Out of Scope

- This project is not designed for the analysis of scientific field data or geospatial data.

## Future Considerations

- [Potential future enhancements]
- [Long-term vision]
