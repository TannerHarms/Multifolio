# Multifolio

Multifolio provides seamless tools for interrogating and generating tabular datasets, accessible to analysts of all skill levels regardless of coding experience. It combines data analysis capabilities with ML-powered statistical sampling to help scientists, statisticians, and analysts extract insights and build models from parametric data.

## Architecture

Multifolio uses a **decoupled, API-first architecture**:
- **Backend**: Python core with FastAPI REST API server
- **Frontends**: 
  - Desktop GUI (PyQt6) - *current implementation*
  - Web GUI (React + D3) - *planned for future*

Both frontends consume the same backend API, ensuring consistency and flexibility.

## Quick Links
- [Project Goals](PROJECT_GOALS.md) - Objectives, features, and success criteria
- [Structure](STRUCTURE.md) - Architecture, tech stack, and organization
- [Style Guide](STYLE_GUIDE.md) - Coding standards and conventions

## Getting Started

### Prerequisites
- Python 3.9+
- pip or conda package manager
- (Optional) GPU support for ML training acceleration

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd Multifolio

# Install backend dependencies
cd backend
pip install -r requirements.txt
# Or: pip install -e .  # Install in development mode

# Install desktop GUI dependencies (optional)
cd ../frontend/desktop
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Desktop GUI (Current)
```bash
# From project root
python frontend/desktop/multifolio_gui/main.py

# Or if installed as package
multifolio-desktop
```

#### Option 2: Backend API Server (for web frontend or API usage)
```bash
# Start FastAPI server
cd backend
uvicorn multifolio.api.server.main:app --reload --port 8000

# API documentation available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

#### Option 3: Python API (in scripts)
```python
from multifolio import Dataset, Visualizer

data = Dataset.from_csv('mydata.csv')
vis = Visualizer(data)
vis.scatter(x='col1', y='col2').show()
```

#### Option 4: Web GUI (Future)
```bash
# Start backend API
cd backend
uvicorn multifolio.api.server.main:app --port 8000

# In another terminal, start React dev server
cd frontend/web/client
npm install
npm run dev  # Opens at http://localhost:5173
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/test_data_loader.py

# Run with coverage
pytest --cov=multifolio --cov-report=html
```

## Project Status
🚧 **Status:** In Development

### Current Version
v0.1.0 (Initial Development)

### Roadmap
- [ ] Phase 1: Core backend + data analysis + Python API
- [ ] Phase 2: Desktop GUI (PyQt6) + statistical sampling
- [ ] Phase 3: ML distribution learning module
- [ ] Phase 4: REST API server (FastAPI)
- [ ] Phase 5: Web GUI (React + D3)

## Contributing
1. Review the [Style Guide](STYLE_GUIDE.md)
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

## Documentation
Detailed documentation can be found in the `/docs` directory:
- [API Documentation](docs/api.md) (when applicable)
- [User Guide](docs/user-guide.md)
- [Development Guide](docs/development.md)

## License
[License type, e.g., MIT, Apache 2.0]

## Contact
[Contact information or links]

## Acknowledgments
[Credits and acknowledgments]
