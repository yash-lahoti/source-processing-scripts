# Quick Start: Deploy to PyPI

## Before You Start

1. **Update `pyproject.toml`**:
   - Replace `"Your Name"` and `"your.email@example.com"` with your details
   - Update GitHub URLs in `[project.urls]` if you have a repository

2. **Create PyPI Accounts**:
   - TestPyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

3. **Get API Tokens** (recommended):
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/
   - Create tokens and save them securely

## Deploy Steps

### 1. Install Build Tools
```bash
pip install --upgrade build twine
```

### 2. Build Package
```bash
python -m build
```

This creates files in `dist/` directory.

### 3. Test on TestPyPI (Optional but Recommended)
```bash
python -m twine upload --repository testpypi dist/*
```

When prompted:
- Username: `__token__`
- Password: Your TestPyPI API token

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ patient-aggregator
```

### 4. Upload to PyPI
```bash
python -m twine upload dist/*
```

When prompted:
- Username: `__token__`
- Password: Your PyPI API token

### 5. Verify
```bash
pip install patient-aggregator
aggregate-patients --help
```

## Updating the Package

1. Update version in `pyproject.toml` (e.g., `0.1.0` → `0.1.1`)
2. Run `python -m build`
3. Run `python -m twine upload dist/*`

## Troubleshooting

- **"File already exists"**: Version already on PyPI - increment version number
- **Authentication errors**: Use API tokens, not passwords
- **Build errors**: Check `pyproject.toml` syntax

For detailed instructions, see `DEPLOYMENT.md` in this directory.

