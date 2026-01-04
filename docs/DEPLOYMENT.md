# Deploying to PyPI

This guide walks you through uploading the `patient-aggregator` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both:
   - **TestPyPI** (for testing): https://test.pypi.org/account/register/
   - **PyPI** (production): https://pypi.org/account/register/

2. **Install build tools**:
   ```bash
   pip install --upgrade build twine
   ```

## Step 1: Update Package Metadata

Before deploying, update `pyproject.toml` with your information:

1. Replace `"Your Name"` and `"your.email@example.com"` with your actual name and email
2. Update the GitHub URLs in `[project.urls]` with your repository URL (if you have one)

## Step 2: Build the Package

Build both source distribution and wheel:

```bash
python -m build
```

This creates:
- `dist/patient-aggregator-0.1.0.tar.gz` (source distribution)
- `dist/patient-aggregator-0.1.0-py3-none-any.whl` (wheel)

## Step 3: Test on TestPyPI (Recommended)

First, test your package on TestPyPI:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for:
# - Username: your TestPyPI username
# - Password: your TestPyPI password (or API token)
```

Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ patient-aggregator
```

## Step 4: Upload to PyPI

Once tested, upload to production PyPI:

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- Username: your PyPI username
- Password: your PyPI password (or API token)

**Note**: For security, use an API token instead of your password:
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Use `__token__` as username and the token as password

## Step 5: Verify Installation

After upload, verify it works:

```bash
pip install patient-aggregator
aggregate-patients --help
```

## Updating the Package

For future releases:

1. **Update version** in `pyproject.toml` (e.g., `0.1.0` → `0.1.1`)
2. **Build again**: `python -m build`
3. **Upload**: `python -m twine upload dist/*`

## Troubleshooting

### "File already exists" error
- Version number already exists on PyPI
- Increment the version in `pyproject.toml`

### Authentication errors
- Use API tokens instead of passwords
- Ensure you're using the correct credentials for TestPyPI vs PyPI

### Import errors after installation
- Verify all dependencies are listed in `pyproject.toml`
- Check that package structure is correct

## Additional Resources

- [PyPI Packaging Guide](https://packaging.python.org/en/latest/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging User Guide](https://packaging.python.org/)

