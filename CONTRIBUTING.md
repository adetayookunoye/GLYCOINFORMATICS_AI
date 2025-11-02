# Contributing to GLYCOINFORMATICS_AI

## 🤝 Welcome Contributors!

Thank you for your interest in contributing to the GlycoInformatics AI Platform! This project aims to advance glycobiology research through comprehensive data integration and AI-powered analysis.

## 📋 Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/adetayookunoye/GLYCOINFORMATICS_AI.git
cd GLYCOINFORMATICS_AI

# Start all services
docker-compose up -d

# Verify installation
curl http://localhost:8000/healthz
```

## 💻 Development Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 3. Database Initialization
```bash
# Start infrastructure services
docker-compose up -d postgres redis graphdb

# Run database migrations
python scripts/init_sample_data.py
```

## 🤝 Contributing Guidelines

### Types of Contributions Welcome
- 🐛 **Bug fixes** and issue reports
- ✨ **New features** and enhancements
- 📚 **Documentation** improvements
- 🧪 **Tests** and quality improvements
- 🎨 **UI/UX** enhancements
- 📈 **Performance** optimizations

### Contribution Process
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Pull Request Guidelines
- Clear, descriptive title
- Detailed description of changes
- Reference related issues
- Include tests for new features
- Update documentation as needed
- Ensure all CI checks pass

## 📝 Code Standards

### Python Code Style
- Follow [PEP 8](https://pep8.org/) guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [mypy](http://mypy-lang.org/) for type checking
- Maximum line length: 88 characters

### Code Quality Tools
```bash
# Format code
black .

# Check types
mypy glycokg/ glycollm/ glycogot/

# Lint code
flake8 .

# Run all quality checks
make lint
```

### Documentation Standards
- All public functions must have docstrings
- Use Google-style docstrings
- Include type hints
- Update README.md for major changes

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=glycokg --cov=glycollm --cov=glycogot

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
```

### Test Requirements
- All new features require tests
- Maintain >80% code coverage
- Include both unit and integration tests
- Test edge cases and error conditions

## 📚 Documentation

### Documentation Structure
- `documentations/` - Technical documentation
- `README.md` - Project overview and quickstart
- Code comments - Inline documentation
- API docs - Auto-generated from code

### Building Documentation
```bash
# Build MkDocs site
mkdocs build

# Serve locally
mkdocs serve
```

## 🏷️ Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to docs
- `good-first-issue` - Good for newcomers
- `help-wanted` - Extra attention needed
- `priority:high` - High priority items

## 🔬 Research Contributions

### Data Sources
We integrate with major glycoinformatics databases:
- **GlyTouCan** - Glycan structure repository
- **GlyGen** - Protein glycosylation data
- **GlycoPOST** - MS/MS spectra database

### Research Areas
- Glycan structure prediction
- Protein glycosylation analysis
- Pathway reconstruction
- Disease biomarker discovery

## 🌟 Recognition

Contributors will be:
- Listed in the project README
- Acknowledged in academic publications
- Invited to join the core team (for significant contributions)

## 📞 Getting Help

- 📖 Check the [documentation](./documentations/)
- 🐛 Report bugs via [GitHub Issues](https://github.com/adetayookunoye/GLYCOINFORMATICS_AI/issues)
- 💬 Ask questions in [Discussions](https://github.com/adetayookunoye/GLYCOINFORMATICS_AI/discussions)
- 📧 Contact maintainers for major contributions

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to advancing glycoinformatics research! 🧬✨**