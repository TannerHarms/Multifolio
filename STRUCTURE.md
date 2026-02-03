# Project Structure

## Architecture Overview
Multifolio uses a **decoupled, API-first architecture** with clear separation between backend and frontend:

1. **Backend (Core + API Server)**
   - **Data Layer**: Handles reading/writing various tabular formats and SQL databases
   - **Processing Layer**: Core analysis, statistical sampling, and ML training modules
   - **API Layer**: RESTful API server that exposes all functionality
   
2. **Frontend (Multiple Options)**
   - **Desktop GUI (Python)**: PyQt6-based application (current implementation)
   - **Web GUI (Future)**: D3/React SPA communicating with backend API
   - Both frontends consume the same backend API

The design prioritizes:
- Internal implementations over external dependencies for robustness
- Clear API contracts between backend and frontend
- Backend can run standalone as API server or embedded in desktop app

## Directory Structure
```
Multifolio/
├── backend/                    # Backend core and API server
│   ├── multifolio/            # Main Python package
│   │   ├── core/              # Core business logic
│   │   │   ├── data/          # Data loading and handling
│   │   │   │   ├── loaders/   # File format readers (CSV, Excel, etc.)
│   │   │   │   ├── sql/       # SQL database connectors (optional)
│   │   │   │   ├── processors/ # Data transformation and cleaning
│   │   │   │   └── validators/ # Data type validation
│   │   │   ├── analysis/      # Data analysis and interrogation
│   │   │   │   ├── statistics/ # Statistical analysis tools
│   │   │   │   ├── aggregation/ # Data aggregation functions
│   │   │   │   └── filters/   # Data filtering utilities
│   │   │   ├── visualization/ # Visualization data preparation
│   │   │   │   ├── plot_data/ # Prepare data for plotting
│   │   │   │   ├── graph_data/ # Graph structure generation
│   │   │   │   └── layouts/   # Layout algorithms
│   │   │   ├── sampling/      # Parameter sampling module
│   │   │   │   ├── distributions/ # Common density functions
│   │   │   │   ├── custom/    # Custom distribution definitions
│   │   │   │   └── generators/ # Sample generation engines
│   │   │   └── ml/            # Machine learning module
│   │   │       ├── models/    # ML model definitions
│   │   │       ├── training/  # Training pipelines
│   │   │       └── inference/ # Model inference and sampling
│   │   ├── api/               # API layer (can run as server or embedded)
│   │   │   ├── server/        # FastAPI/Flask REST API server
│   │   │   │   ├── routes/    # API route definitions
│   │   │   │   ├── schemas/   # Pydantic models for request/response
│   │   │   │   └── middleware/ # CORS, auth, etc.
│   │   │   ├── python/        # Python API (direct library usage)
│   │   │   │   ├── dataset.py # Dataset management API
│   │   │   │   ├── sampler.py # Sampling API
│   │   │   │   └── visualizer.py # Visualization API
│   │   │   └── websocket/     # WebSocket support for real-time updates
│   │   ├── utils/             # Utility functions
│   │   └── config/            # Configuration management
│   ├── tests/                 # Backend tests
│   │   ├── unit/              # Unit tests
│   │   ├── integration/       # Integration tests
│   │   ├── api/               # API endpoint tests
│   │   ├── fixtures/          # Test data fixtures
│   │   └── benchmarks/        # Performance benchmarks
│   ├── requirements.txt       # Backend Python dependencies
│   ├── setup.py               # Package setup
│   └── pyproject.toml         # Modern Python project config
├── frontend/                   # Frontend applications
│   ├── desktop/               # Python desktop GUI (current)
│   │   ├── multifolio_gui/    # Desktop app package
│   │   │   ├── widgets/       # PyQt6 widgets
│   │   │   ├── views/         # Application views/screens
│   │   │   ├── controllers/   # Event handlers
│   │   │   ├── rendering/     # Plotly rendering in Qt
│   │   │   └── main.py        # Desktop app entry point
│   │   ├── assets/            # Icons, styles, etc.
│   │   ├── requirements.txt   # Desktop GUI dependencies
│   │   └── build/             # Build configs for packaging
│   └── web/                   # Web-based GUI (future)
│       ├── client/            # React frontend
│       │   ├── src/
│       │   │   ├── components/ # React components
│       │   │   ├── d3/        # D3 visualizations
│       │   │   ├── hooks/     # React hooks
│       │   │   ├── services/  # API client
│       │   │   └── App.tsx    # Main app component
│       │   ├── public/
│       │   ├── package.json
│       │   └── tsconfig.json
│       └── README.md          # Web GUI setup instructions
├── examples/                   # Example scripts and notebooks
│   ├── python_api/            # Python API usage examples
│   ├── notebooks/             # Jupyter notebooks
│   └── rest_api/              # REST API usage examples (curl, Python requests)
├── docs/                       # Documentation
│   ├── api/                   # API documentation
│   ├── guides/                # User guides
│   └── architecture/          # Architecture docs
├── data/                       # Sample datasets
├── scripts/                    # Utility scripts
│   ├── start_server.py        # Launch API server
│   └── build_desktop.py       # Build desktop app
├── docker/                     # Docker configurations
│   ├── Dockerfile.backend     # Backend API container
│   └── docker-compose.yml     # Full stack setup
├── .gitignore
├── README.md
├── PROJECT_GOALS.md
├── STRUCTURE.md
└── STYLE_GUIDE.md
```

## Technology Stack

### Backend Technologies
- **Language:** Python 3.9+
- **API Framework:** FastAPI (REST API server with automatic OpenAPI docs)
- **Data Processing:** NumPy, Pandas
- **Visualization Core:** Plotly (JSON-based, works with both Python and web)
- **Graph Processing:** NetworkX (graph structures and algorithms)
- **ML Framework:** PyTorch or TensorFlow (for distribution learning)
- **File I/O:** openpyxl (Excel), pyarrow (Parquet), h5py (HDF5)
- **Database (optional):** SQLAlchemy (SQL abstraction), psycopg2 (PostgreSQL), sqlite3 (built-in)
- **WebSocket:** python-socketio (real-time updates for web GUI)

### Desktop Frontend Technologies (Current)
- **GUI Framework:** PyQt6
- **Visualization Rendering:** Plotly rendered in Qt WebEngine
- **Graph Rendering:** PyVis or NetworkX with Plotly

### Web Frontend Technologies (Future)
- **Framework:** React 18+ with TypeScript
- **Visualization:** D3.js (custom interactive visualizations)
- **Graph Visualization:** D3-force, Cytoscape.js, or React-force-graph
- **State Management:** Redux Toolkit or Zustand
- **API Client:** Axios with TypeScript types
- **Build Tool:** Vite
- **UI Components:** Material-UI or Ant Design

### Development Tools
- **Version Control:** Git
- **Backend Package Manager:** pip / conda
- **Frontend Package Manager:** npm / yarn
- **Testing:** pytest (backend), Jest + React Testing Library (web frontend)
- **Code Quality:** 
  - Backend: black, flake8, mypy
  - Frontend: ESLint, Prettier, TypeScript strict mode
- **Documentation:** 
  - Backend: Sphinx with autodoc
  - API: FastAPI automatic OpenAPI/Swagger docs
  - Frontend: Storybook for component documentation

### Infrastructure
- **Deployment Options:**
  - Desktop: Standalone PyQt6 app (PyInstaller or Nuitka)
  - Web: Backend as containerized API (Docker) + React SPA (static hosting)
- **CI/CD:** GitHub Actions for automated testing and builds
- **Performance Optimization:**
  - Numba for JIT compilation of critical paths
  - Multiprocessing for parallel operations
  - Optional: Cython or PyO3 (Rust bindings) for bottlenecks
  - GPU acceleration via CuPy or PyTorch for large operations

## Data Flow
### Analysis Workflow
1. **Input**: User loads tabular data via GUI or API (CSV, Excel, Parquet, etc.)
2. **Validation**: System validates data types and structure
3. **Processing**: Data is parsed into internal representation (optimized DataFrame-like structure)
4. **Analysis**: User applies filters, aggregations, statistical operations
5. **Visualization**: Results displayed in GUI or exported as publication-quality figures

### Sampling Workflow
1. **Configuration**: User defines parameter distributions (built-in or custom)
2. **Generation**: Sampling engine generates parameter sets
3. **Optional ML**: Train model on existing dataset to learn distribution
4. **Export**: Samples exported as tabular data or integrated into analysis pipeline

## Module Descriptions

### Core Modules (Backend)

#### Data Module (`multifolio.core.data`)
**Purpose:** Handle loading, validating, and converting various tabular data formats  
**Key Components:** 
- Format-specific loaders (CSV, Excel, Parquet, HDF5)
- SQL database connectors (PostgreSQL, MySQL, SQLite) for large datasets
- Type validators for strings, ints, floats, arrays, nested structures
- Data transformation pipelines
**Dependencies:** NumPy, Pandas (minimal usage), format-specific libraries, SQLAlchemy (optional)

#### Analysis Module (`multifolio.core.analysis`)
**Purpose:** Provide tools for data interrogation and statistical analysis  
**Key Components:**
- Statistical functions (mean, median, distributions, correlations)
- Filtering and aggregation engines
- Query interface for data interrogation
**Dependencies:** NumPy, SciPy

#### Visualization Module (`multifolio.core.visualization`)
**Purpose:** Prepare visualization data structures (not rendering - that's frontend-specific)  
**Key Components:**
- Generate Plotly JSON specifications for plots (works with both Python and web)
- Graph structure generation with NetworkX
- Layout algorithms for graph visualization
- Data preparation for D3 (for web frontend)
**Dependencies:** Plotly (for JSON specs), NetworkX (graph algorithms)

#### Sampling Module (`multifolio.core.sampling`)
**Purpose:** Generate parametric samples from statistical distributions  
**Key Components:**
- Built-in distributions (normal, uniform, beta, gamma, etc.)
- Custom distribution definitions
- Multi-parameter sampling with correlations
**Dependencies:** NumPy, SciPy

#### ML Module (`multifolio.core.ml`)
**Purpose:** Learn distributions from data and generate samples  
**Key Components:**
- Distribution learning models (VAE, normalizing flows, GANs)
- Training pipelines with validation
- Inference engines for sample generation
**Dependencies:** PyTorch/TensorFlow, scikit-learn

### API Module (`multifolio.api`)

#### REST API Server (`multifolio.api.server`)
**Purpose:** Expose backend functionality via HTTP/WebSocket for web frontend  
**Key Components:**
- FastAPI application with route handlers
- Pydantic schemas for request/response validation
- CORS middleware for cross-origin requests
- WebSocket handlers for real-time updates
**Dependencies:** FastAPI, Uvicorn, Pydantic, python-socketio

#### Python API (`multifolio.api.python`)
**Purpose:** Programmatic Python interface for scripts and desktop GUI  
**Key Components:**
- Dataset class for data manipulation
- Sampler interface for parameter generation
- Visualizer for plot specification generation
**Dependencies:** All core modules

### Frontend Modules

#### Desktop GUI (`frontend/desktop/multifolio_gui`)
**Purpose:** PyQt6-based desktop application (current implementation)  
**Key Components:**
- Data table viewer with sorting/filtering
- Plotly chart rendering in Qt WebEngine
- Parameter sampling configuration panels
- Graph visualization viewer
**Dependencies:** PyQt6, PyQtWebEngine, multifolio.api.python

#### Web GUI (`frontend/web` - Future)
**Purpose:** React + D3 web application  
**Key Components:**
- React components for UI
- D3 custom visualizations with full interactivity
- API client for backend communication
- Real-time updates via WebSocket
**Dependencies:** React, D3, TypeScript, Axios

## API Design

### REST API (FastAPI)
The backend exposes a RESTful API that can be consumed by any frontend:

```python
# Start API server
# uvicorn multifolio.api.server.main:app --reload

# API Endpoints (examples)
POST   /api/v1/datasets/upload          # Upload dataset
GET    /api/v1/datasets/{id}            # Get dataset info
POST   /api/v1/datasets/{id}/filter     # Apply filter
GET    /api/v1/datasets/{id}/stats      # Get statistics
POST   /api/v1/visualizations/scatter   # Generate scatter plot data
POST   /api/v1/visualizations/graph     # Generate graph visualization
POST   /api/v1/sampling/generate        # Generate parameter samples
POST   /api/v1/ml/train                 # Train distribution model
POST   /api/v1/ml/sample                # Sample from trained model
WS     /api/v1/ws                       # WebSocket for real-time updates
```

#### Example REST API Usage (from web frontend)
```javascript
// TypeScript/JavaScript (React frontend)
import axios from 'axios';

// Upload dataset
const formData = new FormData();
formData.append('file', file);
const response = await axios.post('/api/v1/datasets/upload', formData);
const datasetId = response.data.dataset_id;

// Get scatter plot data
const plotData = await axios.post(
  `/api/v1/visualizations/scatter`,
  { dataset_id: datasetId, x: 'param1', y: 'param2', color: 'category' }
);

// Render with D3
const svg = d3.select('#plot');
svg.selectAll('circle')
  .data(plotData.data.points)
  .enter()
  .append('circle')
  .attr('cx', d => xScale(d.x))
  .attr('cy', d => yScale(d.y))
  .attr('fill', d => colorScale(d.color))
  .on('click', (event, d) => {
    // Show parameters on click
    console.log('Point data:', d.parameters);
  });
```

### Python API (Direct Library Usage)
For Python scripts and desktop GUI - uses the same backend core:

```python
# Python API (programmatic usage)
from multifolio import Dataset

# From file
data = Dataset.from_csv('data.csv')

# From SQL database (optional)
data = Dataset.from_sql(
    'SELECT * FROM experiments WHERE date > "2025-01-01"',
    connection='postgresql://user:pass@localhost/mydb'
)

filtered = data.filter(lambda row: row['value'] > 10)
summary = filtered.aggregate(['mean', 'std'])

# Interactive visualization with click-to-inspect
from multifolio import Visualizer
vis = Visualizer(data)

# Scatter plot with interactive point inspection
plot = vis.scatter(x='param1', y='param2', color='category')
plot.on_click(lambda point: print(f"Parameters: {point.data}"))
plot.show()  # Opens in PyQt6 window or returns Plotly JSON for web

# Graph visualization with custom adjacency
def adjacency_func(node1, node2):
    # Define when two nodes should be connected
    return abs(node1['value'] - node2['value']) < threshold

graph = vis.graph(
    nodes=data,
    adjacency=adjacency_func,
    node_color='category',
    node_size='importance'
)
graph.show()

# Parameter sampling
from multifolio import Sampler
sampler = Sampler()
sampler.add_param('x', distribution='normal', mean=0, std=1)
sampler.add_param('y', distribution='uniform', low=-1, high=1)
samples = sampler.generate(n=1000)

# ML-based sampling
from multifolio.ml import DistributionLearner
learner = DistributionLearner(data)
learner.train(epochs=100)
generated_samples = learner.sample(n=1000)
```

### Design Principles
- **API-first**: Backend designed to serve multiple frontends
- **Consistent data format**: Plotly JSON works across Python and web
- **Type safety**: FastAPI uses Pydantic for validation, Python API uses type hints
- **WebSocket support**: Real-time updates for long-running operations (ML training, large data processing)
- **CORS-enabled**: Web frontend can communicate with local or remote backend
- **Authentication-ready**: Can add JWT auth for multi-user deployments

## Data Storage
Multifolio supports both file-based and database data sources:

### Supported File Formats
- **CSV**: Comma-separated values (most common)
- **Excel**: .xlsx and .xls files with multi-sheet support
- **Parquet**: Columnar format for large datasets
- **HDF5**: Hierarchical format for complex nested data
- **JSON**: For configuration and metadata

### SQL Database Support (Optional)
- **PostgreSQL**: For large relational datasets
- **MySQL/MariaDB**: Alternative SQL backend
- **SQLite**: Lightweight, file-based SQL database
- **Use case**: When data is too large for memory or needs concurrent access
- **Integration**: SQLAlchemy provides unified interface

### Internal Data Representation
- Uses optimized in-memory structures based on NumPy arrays
- Supports heterogeneous types (mixed strings, numbers, arrays in columns)
- Lazy loading for large datasets that exceed memory
- Chunked processing for operations on massive files

## Configuration Management
### User Configuration
- **Config file**: `~/.multifolio/config.json` or `config/settings.yaml`
- **Settings include**:
  - Default file paths and formats
  - Visualization themes and defaults
  - ML model hyperparameters
  - Performance tuning (memory limits, chunk sizes)
  - GUI preferences

### Environment Variables (Optional)
- `MULTIFOLIO_DATA_DIR`: Default directory for datasets
- `MULTIFOLIO_CACHE_DIR`: Cache location for large file operations
- `MULTIFOLIO_GPU`: Enable/disable GPU acceleration for ML

### No Secrets Required
- Desktop application with no external authentication
- All data processing is local

## Scalability Considerations
### Performance Targets
- Handle datasets up to millions of cells (rows × columns)
- Responsive GUI with <100ms interaction latency
- Visualization rendering <1 second for typical plots

### Optimization Strategies
1. **Vectorization**: Use NumPy operations instead of Python loops
2. **Lazy Evaluation**: Load and process data on-demand when possible
3. **Chunked Processing**: Break large datasets into manageable chunks
4. **Caching**: Cache computed statistics and transformed data
5. **JIT Compilation**: Use Numba for performance-critical functions
6. **Parallel Processing**: Use multiprocessing for independent operations
7. **GPU Acceleration**: Leverage GPU for ML training and large matrix operations

### Known Bottlenecks
- **Memory**: Large datasets may exceed available RAM (use chunking)
- **File I/O**: Reading massive files (use streaming parsers)
- **Visualization**: Plotting millions of points (use downsampling/aggregation)
- **ML Training**: Distribution learning can be slow (GPU acceleration helps)
