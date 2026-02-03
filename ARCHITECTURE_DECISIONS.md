# Architecture Decisions: Learning Software Engineering Through Multifolio

*A personal guide to understanding the structural choices in this project*

---

## Table of Contents
1. [Core Architectural Principles](#core-architectural-principles)
2. [Backend/Frontend Separation](#backendfrontend-separation)
3. [File Format and Data Structure Decisions](#file-format-and-data-structure-decisions)
4. [Module Organization Strategy](#module-organization-strategy)
5. [Technology Selection Rationale](#technology-selection-rationale)
6. [API Design Philosophy](#api-design-philosophy)
7. [Scalability and Maintainability Patterns](#scalability-and-maintainability-patterns)
8. [Key Software Engineering Lessons](#key-software-engineering-lessons)

---

## Core Architectural Principles

### 1. Separation of Concerns (SoC)

**What it means:** Different parts of the system should handle different responsibilities, with minimal overlap.

**How we applied it:**
```
Backend/          → Business logic, data processing, computations
Frontend/desktop/ → User interface, user interactions (PyQt6)
Frontend/web/     → User interface, user interactions (React)
```

**Why this matters:**
- **Testability**: You can test backend logic without needing a GUI
- **Reusability**: Backend can serve multiple frontends (desktop, web, CLI, mobile)
- **Team parallelization**: Backend and frontend developers can work independently
- **Easier debugging**: Problems are isolated to specific layers

**Real-world example:** When you discover a bug in data filtering, you only need to fix `backend/multifolio/core/analysis/filters/` - both GUIs automatically benefit.

### 2. API-First Design

**What it means:** Design your system's interface (API) before implementing the internals.

**How we applied it:**
- Backend exposes **REST API** (HTTP endpoints) and **Python API** (library imports)
- Both APIs provide the same functionality with different interfaces
- Frontends *only* interact through these APIs, never directly with core modules

**Why this matters:**
- **Contracts**: APIs define clear contracts between components
- **Flexibility**: Swap implementations without breaking consumers
- **Documentation**: API surface is well-defined and documentable
- **Versioning**: Can evolve APIs with backward compatibility (v1, v2)

**Anti-pattern to avoid:** Tight coupling where frontend directly imports backend internals:
```python
# BAD: Frontend importing core internals
from backend.multifolio.core.data.loaders.csv import CSVLoader
loader = CSVLoader()  # Frontend knows too much!

# GOOD: Frontend uses API
from multifolio import Dataset
data = Dataset.from_csv('file.csv')  # API abstracts implementation
```

### 3. Dependency Inversion Principle

**What it means:** High-level modules shouldn't depend on low-level modules. Both should depend on abstractions.

**How we applied it:**
- Frontend depends on **API interface** (abstraction), not concrete implementations
- Core modules depend on **interfaces/protocols**, not specific file formats
- Example: Visualization module returns Plotly JSON (standard format), not Python-specific objects

**Why this matters:**
```
Desktop GUI (PyQt6) ─┐
                     ├─→ Backend API ─→ Core Logic
Web GUI (React)   ───┘

Both GUIs depend on the same API abstraction.
If we swap PyQt6 for Tkinter, backend doesn't change.
If we rewrite backend in Rust, API contract stays the same.
```

---

## Backend/Frontend Separation

### Why Split Into Separate Directories?

**Directory structure:**
```
Multifolio/
├── backend/          # Server-side logic
└── frontend/         # Client-side interfaces
    ├── desktop/      # Python GUI
    └── web/          # JavaScript SPA
```

### Benefits of This Split

#### 1. **Independent Deployment**
- Desktop app can be packaged as standalone `.exe` or `.app`
- Web frontend can be deployed to CDN (Netlify, Vercel)
- Backend can be deployed as Docker container on server
- Each component has independent release cycles

#### 2. **Different Dependency Management**
```
backend/requirements.txt       → numpy, fastapi, pytorch
frontend/desktop/requirements.txt → PyQt6
frontend/web/package.json      → react, d3, typescript
```

Each part manages its own dependencies without conflicts.

#### 3. **Technology Agnosticism**
Want to add a mobile app? Just add `frontend/mobile/` that consumes the same backend API.

#### 4. **Security Boundaries**
- Backend handles sensitive operations (file system, databases, ML training)
- Frontend only has UI permissions
- Clear boundary for security auditing

### When NOT to Separate

If you're building a simple script or single-purpose tool, this separation is overkill. Use it when:
- ✅ Multiple interfaces to same logic
- ✅ Different deployment targets
- ✅ Team working on different parts
- ❌ Simple one-off script
- ❌ Proof of concept

---

## File Format and Data Structure Decisions

### Why Plotly JSON as Intermediate Format?

**Key decision:** Backend generates Plotly JSON instead of rendering plots directly.

#### The Problem
Different frontends need different rendering:
- Desktop: Matplotlib/PyQt widgets
- Web: D3, Canvas, SVG
- Export: PNG, PDF, SVG files

#### Traditional Approach (Bad)
```python
# Backend generates platform-specific objects
def create_plot(data):
    fig = plt.figure()  # Matplotlib object (Python-only!)
    plt.scatter(data.x, data.y)
    return fig  # Web frontend can't use this!
```

#### Our Approach (Good)
```python
# Backend generates JSON specification
def create_plot(data):
    return {
        "data": [
            {
                "type": "scatter",
                "x": data.x.tolist(),
                "y": data.y.tolist(),
                "mode": "markers"
            }
        ],
        "layout": {"title": "My Plot"}
    }  # Pure JSON - works everywhere!
```

#### Benefits of JSON Specifications

1. **Language Agnostic**
   - Python can generate it
   - JavaScript can consume it
   - Any language can read JSON

2. **Wire-Friendly**
   - Easily transmitted over HTTP
   - Can be cached
   - Can be stored in database

3. **Standard Format**
   - Plotly is widely supported
   - Has Python library (plotly.py) and JavaScript library (plotly.js)
   - Documentation is universal

4. **Flexibility**
   - Desktop GUI: `plotly.py` renders JSON in PyQt WebEngine
   - Web GUI: `plotly.js` renders JSON in browser
   - Advanced users: Take JSON and use D3 for custom rendering

### Why Support Multiple Input File Formats?

**Supported:** CSV, Excel, Parquet, HDF5, SQL databases

#### Principle: "Liberal in what you accept, conservative in what you send"

**Liberal in acceptance:**
- Users have data in many formats
- Converting is friction
- Each format has advantages:
  - CSV: Universal, text-based, version-control friendly
  - Excel: Business users love it, multi-sheet support
  - Parquet: Columnar storage, compressed, very fast for large data
  - HDF5: Nested structures, scientific standard
  - SQL: Data already in database, huge datasets

**Conservative in output:**
- Internal representation is **NumPy arrays** (fast, standard)
- Export options are well-defined and tested

#### Implementation Pattern: Loaders

```python
# Each format has its own loader
backend/multifolio/core/data/loaders/
├── csv_loader.py
├── excel_loader.py
├── parquet_loader.py
└── hdf5_loader.py

# All implement common interface
class DataLoader(Protocol):
    def load(self, path: str) -> InternalDataFrame:
        ...

# API abstracts which loader to use
data = Dataset.from_csv("file.csv")   # Uses CSVLoader
data = Dataset.from_excel("file.xlsx") # Uses ExcelLoader
```

**Why this pattern?**
- **Open/Closed Principle**: Open for extension (add new loaders), closed for modification (core logic doesn't change)
- **Single Responsibility**: Each loader handles one format
- **Easy testing**: Test each loader independently

### SQL Database Support: When and Why

**Decision:** Make SQL support **optional**, not required.

#### When SQL Makes Sense
- ✅ Data too large for memory (billions of rows)
- ✅ Data already in database (don't duplicate)
- ✅ Need concurrent access (multiple users)
- ✅ Need transactional integrity
- ✅ Data updates frequently

#### When Files Make Sense
- ✅ Dataset fits in memory
- ✅ Single-user analysis
- ✅ Portability (email a CSV)
- ✅ Version control (git tracks CSV changes)
- ✅ Simplicity (no database setup)

#### Our Approach: Support Both
```python
# File-based (simple)
data = Dataset.from_csv('data.csv')

# SQL-based (for large data)
data = Dataset.from_sql(
    'SELECT * FROM big_table WHERE date > "2024-01-01"',
    connection='postgresql://...'
)
```

**Key insight:** Most users start with files. SQL support shouldn't be mandatory, but should be there when needed.

---

## Module Organization Strategy

### The Core/API/Frontend Pattern

```
backend/multifolio/
├── core/          # Business logic (pure Python, no web/GUI concerns)
├── api/           # Interface layer
│   ├── python/    # Direct Python imports
│   └── server/    # REST API (FastAPI)
```

### Why Separate Core from API?

#### Core = Pure Logic
```python
# multifolio/core/analysis/statistics.py
def calculate_mean(data: np.ndarray) -> float:
    """Pure function, no I/O, no side effects"""
    return np.mean(data)
```

**Characteristics:**
- No HTTP knowledge
- No database connections
- No file I/O (delegates to loaders)
- Pure computations
- Highly testable

#### API = Interface Adapter
```python
# multifolio/api/server/routes/analysis.py
@app.post("/api/v1/analysis/mean")
async def compute_mean(request: MeanRequest):
    """Adapts HTTP to core logic"""
    data = load_dataset(request.dataset_id)
    result = calculate_mean(data)  # Calls core
    return {"mean": result}
```

**Characteristics:**
- Handles HTTP request/response
- Authentication/authorization
- Input validation (Pydantic)
- Error handling and status codes
- Converts between wire format (JSON) and internal format

### Hexagonal Architecture (Ports and Adapters)

This is a professional architecture pattern we're using:

```
        ┌─────────────────────────────┐
        │      Core Business Logic    │
        │  (Data, Analysis, Sampling) │
        └──────────┬───────┬──────────┘
                   │       │
        ┌──────────┴───┐   └──────────┐
        │ Python API   │   │ REST API │  ← Adapters/Ports
        └──────┬───────┘   └────┬─────┘
               │                │
        ┌──────┴───────┐   ┌────┴─────┐
        │ Desktop GUI  │   │ Web GUI  │  ← Clients
        └──────────────┘   └──────────┘
```

**Benefits:**
1. Core logic is independent of delivery mechanism
2. Can test core without spinning up web server
3. Can swap adapters (REST → GraphQL) without changing core
4. Clear boundaries for team ownership

---

## Technology Selection Rationale

### Backend: Why Python?

**Pros:**
- Rich data science ecosystem (NumPy, SciPy, Pandas)
- ML libraries (PyTorch, TensorFlow)
- Quick development and iteration
- Type hints for safety (Python 3.9+)

**Cons:**
- Slower than compiled languages (C++, Rust)
- GIL limits multi-threading

**Mitigation strategies:**
- NumPy operations are in C (fast)
- Numba JIT compilation for hot paths
- Multiprocessing for parallelism (bypasses GIL)
- Option to rewrite bottlenecks in Cython/Rust

**When to choose Python:**
- ✅ Data science / scientific computing
- ✅ Rapid prototyping
- ✅ Rich library ecosystem needed
- ❌ Hard real-time requirements
- ❌ Extremely performance-critical (game engines, HFT)

### API Framework: Why FastAPI?

**Alternatives considered:**
- Flask: Simpler but less features
- Django: Too heavy, includes ORM/templates we don't need
- gRPC: More complex, better for microservices

**Why FastAPI:**
- **Automatic OpenAPI docs** (Swagger UI at `/docs`)
- **Type-based validation** (Pydantic models)
- **Async support** for concurrent requests
- **WebSocket support** for real-time updates
- **Modern Python** (uses type hints)

```python
# FastAPI example
from pydantic import BaseModel

class PlotRequest(BaseModel):
    dataset_id: str
    x_column: str
    y_column: str

@app.post("/api/v1/plot")
async def create_plot(request: PlotRequest):
    # request is automatically validated!
    # API docs are automatically generated!
    ...
```

### Frontend Desktop: Why PyQt6?

**Alternatives:**
- Tkinter: Built-in but limited styling
- Kivy: Cross-platform but different paradigm
- Electron: Web tech but 100MB+ bundle
- wxPython: Good but older

**Why PyQt6:**
- **Professional appearance**
- **Rich widget library**
- **QtWebEngine** for rendering Plotly
- **Cross-platform** (Windows, Mac, Linux)
- **Active development** and community

**Trade-off:** GPL license (or commercial license needed for proprietary). For open-source project, this is fine.

### Frontend Web: Why React + D3?

**React:**
- Component-based architecture (reusable UI pieces)
- Huge ecosystem and community
- Good TypeScript support
- Virtual DOM for performance

**D3:**
- Most powerful visualization library
- Full control over SVG rendering
- Can implement click-to-inspect easily
- Works well with React (React manages components, D3 manages SVG)

**Alternative considered:** Plotly.js alone
- Pros: Consistent with backend
- Cons: Less flexibility for custom interactions

**Our approach:** Use both!
- Plotly.js for standard charts (fast development)
- D3 for custom visualizations (full control)

### Data Format: Why Plotly JSON?

Covered earlier, but key principle: **Choose formats that work across boundaries**

**Good boundary-crossing formats:**
- JSON (universal)
- Protocol Buffers (efficient, typed)
- Arrow (columnar data, zero-copy)

**Bad boundary-crossing formats:**
- Python pickle (Python-only, security risk)
- Python objects (can't serialize)
- Platform-specific binaries

---

## API Design Philosophy

### REST API Principles

#### 1. Resource-Based URLs
```
✅ GOOD
POST   /api/v1/datasets          # Create dataset
GET    /api/v1/datasets/{id}     # Retrieve dataset
PUT    /api/v1/datasets/{id}     # Update dataset
DELETE /api/v1/datasets/{id}     # Delete dataset

❌ BAD
POST /api/v1/createDataset
POST /api/v1/getDataset
POST /api/v1/updateDataset
```

**Principle:** URLs represent resources (nouns), HTTP verbs represent actions.

#### 2. Versioning
```
/api/v1/datasets
```

**Why version?**
- Can evolve API without breaking old clients
- Desktop app v1.0 still works when server is on v2
- Clear migration path

#### 3. Pagination for Large Results
```
GET /api/v1/datasets?page=1&page_size=50
```

**Why?**
- Can't return 1M rows in one response
- Client can fetch incrementally
- Improves performance

#### 4. HTTP Status Codes with Meaning
```python
200 OK          # Success
201 Created     # Resource created
400 Bad Request # Client error (invalid input)
401 Unauthorized # Authentication required
404 Not Found   # Resource doesn't exist
500 Internal Server Error # Server problem
```

**Principle:** Use standard HTTP semantics. Clients know what to expect.

### Python API Principles

#### 1. Fluent Interface (Method Chaining)
```python
# Chainable
result = (dataset
    .filter(lambda x: x > 0)
    .aggregate(['mean', 'std'])
    .sort_by('mean'))

# vs Non-chainable
filtered = dataset.filter(lambda x: x > 0)
aggregated = aggregate(filtered, ['mean', 'std'])
result = sort_by(aggregated, 'mean')
```

**When to use:** Data transformation pipelines

#### 2. Sensible Defaults
```python
# Minimal required parameters
plot = vis.scatter(x='col1', y='col2')

# With optional customization
plot = vis.scatter(
    x='col1', 
    y='col2',
    color='category',
    size='importance',
    opacity=0.7,
    title='My Plot'
)
```

**Principle:** "Make simple things simple, complex things possible"

#### 3. Type Hints Everywhere
```python
def load_dataset(path: str, format: str = 'csv') -> Dataset:
    ...

# IDE knows return type, provides autocomplete
data = load_dataset('file.csv')
data.  # ← IDE suggests .filter(), .aggregate(), etc.
```

**Benefits:**
- IDE autocomplete
- Catch errors at development time (mypy)
- Self-documenting code

---

## Scalability and Maintainability Patterns

### 1. Strategy Pattern for File Loaders

**Problem:** Need to support multiple file formats

**Bad approach:** Giant if/elif
```python
def load_file(path, format):
    if format == 'csv':
        # 50 lines of CSV logic
    elif format == 'excel':
        # 50 lines of Excel logic
    elif format == 'parquet':
        # 50 lines of Parquet logic
```

**Good approach:** Strategy pattern
```python
class CSVLoader:
    def load(self, path): ...

class ExcelLoader:
    def load(self, path): ...

LOADERS = {
    'csv': CSVLoader(),
    'excel': ExcelLoader(),
    'parquet': ParquetLoader(),
}

def load_file(path, format):
    loader = LOADERS[format]
    return loader.load(path)
```

**Benefits:**
- Easy to add new format: just add new loader class
- Each loader is independently testable
- No risk of breaking other formats when modifying one

### 2. Configuration Over Code

**Bad:**
```python
# Hardcoded in source
MAX_ROWS = 1000000
CACHE_DIR = '/tmp/multifolio'
```

**Good:**
```python
# backend/multifolio/config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    max_rows: int = 1000000
    cache_dir: str = '/tmp/multifolio'
    
    class Config:
        env_file = '.env'

# Can override via environment variables
# MULTIFOLIO_MAX_ROWS=5000000
```

**Benefits:**
- Different configs for dev/staging/prod
- No code changes to adjust behavior
- Can be overridden by users

### 3. Dependency Injection

**Problem:** Testing code that uses database

**Bad:**
```python
def process_data():
    db = PostgreSQLConnection()  # Hardcoded!
    data = db.query("SELECT ...")
    return transform(data)

# Can't test without database!
```

**Good:**
```python
def process_data(db: DatabaseConnection):
    data = db.query("SELECT ...")
    return transform(data)

# Production
process_data(PostgreSQLConnection())

# Testing
process_data(MockDatabase())  # No real database needed!
```

### 4. Lazy Loading for Performance

**Problem:** Loading huge datasets upfront is slow

**Bad:**
```python
class Dataset:
    def __init__(self, path):
        self.data = load_entire_file(path)  # 10GB loaded!
```

**Good:**
```python
class Dataset:
    def __init__(self, path):
        self._path = path
        self._data = None  # Not loaded yet
    
    @property
    def data(self):
        if self._data is None:
            self._data = load_file(self._path)  # Load on first access
        return self._data
```

**Benefits:**
- Fast initialization
- Only load what you need
- Can implement chunked loading for huge files

---

## Key Software Engineering Lessons

### 1. "Premature optimization is the root of all evil"

**Mistake:** Optimizing before knowing where the bottleneck is

**Better approach:**
1. Write clear, correct code first
2. Profile to find actual bottlenecks
3. Optimize hot paths only

**Example in this project:**
- Start with pure Python loops
- If slow, profile with `cProfile`
- Optimize proven bottleneck with NumPy vectorization or Numba
- Document why optimization was needed

### 2. "Make it work, make it right, make it fast"

**Phases:**
1. **Make it work:** Get functionality working, even if ugly
2. **Make it right:** Refactor for clarity, add tests, fix design issues
3. **Make it fast:** Optimize based on profiling

Most projects die in phase 1 because developers try to do all three at once.

### 3. "You aren't gonna need it" (YAGNI)

**Mistake:** Adding features "just in case"

```python
# Don't do this unless you need it!
class Dataset:
    def to_csv(self): ...
    def to_excel(self): ...
    def to_parquet(self): ...
    def to_json(self): ...
    def to_xml(self): ...      # Do you really need XML?
    def to_yaml(self): ...     # Or YAML?
    def to_protobuf(self): ... # Really?
```

**Better:** Add formats when users actually request them.

**Exception:** Architecture decisions (like backend/frontend split) that are hard to change later.

### 4. "Don't Repeat Yourself" (DRY)

**Bad:**
```python
# Copy-pasted in 5 places
data = pd.read_csv(path)
data = data[data['value'] > 0]
data = data.dropna()
```

**Good:**
```python
def load_and_clean(path):
    data = pd.read_csv(path)
    data = data[data['value'] > 0]
    data = data.dropna()
    return data
```

**But don't over-DRY:** If two pieces of code are similar by coincidence, not by design, don't force them together.

### 5. "Explicit is better than implicit" (Zen of Python)

**Bad:**
```python
def process(data):
    # Magic! What does this do?
    return data.apply(lambda x: x * 2 if x > 0 else x)
```

**Good:**
```python
def double_positive_values(data: pd.Series) -> pd.Series:
    """Double all positive values, leave negative unchanged."""
    return data.apply(lambda x: x * 2 if x > 0 else x)
```

### 6. Code for Humans, Not Computers

**Bad:**
```python
def f(x, y, z=True):
    return [i**2 for i in x if i>y] if z else [i for i in x if i>y]
```

**Good:**
```python
def filter_and_square_values(
    values: List[float],
    threshold: float,
    should_square: bool = True
) -> List[float]:
    """
    Filter values above threshold and optionally square them.
    
    Args:
        values: List of numbers to process
        threshold: Minimum value to include
        should_square: If True, square the values
    
    Returns:
        Filtered (and possibly squared) values
    """
    filtered = [v for v in values if v > threshold]
    
    if should_square:
        return [v ** 2 for v in filtered]
    
    return filtered
```

### 7. Favor Composition Over Inheritance

**Bad (deep inheritance):**
```python
class DataProcessor:
    pass

class CSVProcessor(DataProcessor):
    pass

class CleanedCSVProcessor(CSVProcessor):
    pass

class FilteredCleanedCSVProcessor(CleanedCSVProcessor):
    pass  # This is getting ridiculous
```

**Good (composition):**
```python
class CSVLoader:
    def load(self, path): ...

class DataCleaner:
    def clean(self, data): ...

class DataFilter:
    def filter(self, data, condition): ...

# Compose behaviors
loader = CSVLoader()
cleaner = DataCleaner()
filter = DataFilter()

data = loader.load(path)
data = cleaner.clean(data)
data = filter.filter(data, lambda x: x > 0)
```

### 8. Test at the Right Level

**Test pyramid:**
```
      ┌─────────┐
      │   E2E   │  ← Few (slow, brittle)
      ├─────────┤
      │Integration│  ← Some (medium speed)
      ├─────────┤
      │  Unit   │  ← Many (fast, focused)
      └─────────┘
```

**Unit tests:** Test individual functions
```python
def test_calculate_mean():
    assert calculate_mean([1, 2, 3]) == 2.0
```

**Integration tests:** Test components together
```python
def test_load_and_filter_csv():
    data = Dataset.from_csv('test.csv')
    filtered = data.filter(lambda x: x['value'] > 10)
    assert len(filtered) == expected_count
```

**E2E tests:** Test through GUI/API
```python
def test_plot_creation_via_api():
    response = client.post('/api/v1/plot', json={...})
    assert response.status_code == 200
```

---

## Practical Application: How to Use These Principles

### When Starting a New Module

1. **Define the interface first** (API-first)
   ```python
   # Write this first (the contract)
   class DataLoader(Protocol):
       def load(self, path: str) -> DataFrame:
           ...
   ```

2. **Write a failing test**
   ```python
   def test_csv_loader():
       loader = CSVLoader()
       data = loader.load('test.csv')
       assert len(data) == 100
   ```

3. **Implement the simplest thing that works**
   ```python
   class CSVLoader:
       def load(self, path):
           return pd.read_csv(path)
   ```

4. **Refactor for clarity and patterns**

5. **Optimize only if needed (profile first)**

### When Designing a New Feature

Ask these questions:

1. **Where does this belong?**
   - Core logic? → `backend/multifolio/core/`
   - API endpoint? → `backend/multifolio/api/server/`
   - UI? → `frontend/desktop/` or `frontend/web/`

2. **What are the inputs and outputs?**
   - Write type hints
   - Document with docstrings

3. **How will this be tested?**
   - If hard to test, design is probably wrong

4. **Could this be reused elsewhere?**
   - If yes, make it generic
   - If no, keep it specific

5. **What could go wrong?**
   - Add error handling
   - Validate inputs

### Code Review Checklist

Use this when reviewing your own code:

- [ ] Does each function do one thing?
- [ ] Are names descriptive?
- [ ] Are there type hints?
- [ ] Is there error handling?
- [ ] Could someone understand this in 6 months?
- [ ] Are there tests?
- [ ] Is configuration extracted from code?
- [ ] Are dependencies injected (not hardcoded)?
- [ ] Is the happy path obvious?
- [ ] Are edge cases handled?

---

## Conclusion: The Meta-Lesson

**The biggest lesson:** There are no silver bullets.

Every architectural decision is a **trade-off**:

- Backend/frontend separation → More files, but better organization
- API-first → More boilerplate, but better flexibility
- Multiple file formats → More code, but better user experience
- Type hints → More typing, but better safety

**Good engineering** is about understanding these trade-offs and making conscious decisions based on your context:

- Small project? → Simpler structure
- Team of 10? → Strict boundaries
- Performance-critical? → Compiled language
- Fast iteration? → Python/JavaScript

**Most important:** Be consistent within a project. Pick patterns and stick to them. Inconsistency is worse than imperfect architecture.

---

## Further Learning Resources

### Books
- *Clean Code* by Robert Martin - Writing maintainable code
- *Design Patterns* by Gang of Four - Classic patterns
- *Domain-Driven Design* by Eric Evans - Organizing complex logic
- *Building Microservices* by Sam Newman - Service architecture

### For This Project Specifically
- FastAPI docs: https://fastapi.tiangolo.com/
- Plotly docs: https://plotly.com/python/
- React + TypeScript: https://react-typescript-cheatsheet.netlify.app/
- D3 tutorials: https://observablehq.com/@d3/learn-d3

### Key Takeaway

**Software engineering is more about people and maintainability than computers and performance.** 

Code is read 10x more than it's written. Optimize for the next developer (which might be you in 6 months) who needs to understand and modify your code.

---

*Document created: February 3, 2026*  
*Author: Personal learning guide for Multifolio project*
