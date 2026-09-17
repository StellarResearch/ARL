# Contributing to AdaptiveRL

Thank you for your interest in contributing to **AdaptiveRL**! This document provides clear guidelines to help you set up your development environment, understand our architecture, and submit high-quality contributions.

---

## 1. Core Development Philosophy

AdaptiveRL is developed under strict quality and architectural rules:
1. **Incremental Milestones:** Work only on the requested phase or targeted issue. Do not attempt large, unfocused refactors.
2. **Interface Isolation:** Algorithms must remain independent of any single concrete environment. All environments conform to the Farama Gymnasium standard.
3. **No Placeholders:** Never commit mock methods that pretend to return valid results without real computation.
4. **Automated Verification:** Every contribution must be accompanied by automated tests.
5. **Clean Verification:** Never claim a feature works without actually running it and verifying test execution.

---

## 2. Getting Started

### 2.1 Fork & Clone
```bash
# Clone your fork or the repository
git clone https://github.com/AryanXCode646/ARL.git
cd ARL
```

### 2.2 Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate

# Install AdaptiveRL in editable mode with development dependencies
pip install -e ".[dev]"
```

Verify that AdaptiveRL is properly installed:
```bash
python -c "import adaptive_rl; print(adaptive_rl.__version__)"
adaptive-rl --help
```

---

## 3. Development Workflow

### 3.1 Branch Naming
Create a feature branch from `main`:
* Features: `feature/milestone-name` or `feature/short-description`
* Bug fixes: `fix/issue-description`
* Docs: `docs/topic-name`

Example:
```bash
git checkout -b feature/phase-2-environment-registry
```

### 3.2 Running Tests
Before opening a pull request, run the test suite:
```bash
# Run all unit and integration tests
pytest -v tests/

# Run with coverage report
pytest --cov=adaptive_rl --cov-report=term-missing tests/
```

### 3.3 Code Quality & Formatting
AdaptiveRL adheres to strict PEP 8 and static typing standards:
```bash
# Check linting
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Run type checker
mypy src/
```

---

## 4. Pull Request Guidelines

1. **Focused Scope:** Keep each pull request focused on one feature, fix, or milestone. Avoid bundling unrelated formatting changes.
2. **Test Coverage:** Every new function, class, or configuration schema must include corresponding tests in `tests/`.
3. **Documentation:** Update relevant documents (`README.md`, `docs/ARCHITECTURE.md`, docstrings) if your change modifies public interfaces.
4. **PR Description:** Describe:
   - What changed
   - Why the change was made
   - Commands executed to verify the change
   - Test results output

---

## 5. Community & Discussion
* Please open an issue to discuss significant changes or proposed environment additions before submitting a large pull request.
* Be respectful and constructive in code reviews and discussions.
