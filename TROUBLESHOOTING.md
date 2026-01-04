# Troubleshooting PyPI Upload Issues

## Common 400 Bad Request Errors

### 1. Package Name Already Exists
If the package name `patient-aggregator` already exists on TestPyPI, you have two options:

**Option A: Use a unique name with your username**
```toml
name = "patient-aggregator-yashlahoti"
```

**Option B: Delete the existing package** (if it's yours)
- Go to https://test.pypi.org/manage/projects/
- Find your package and delete it

### 2. Version Already Exists
If version `0.1.0` already exists, increment the version:
```toml
version = "0.1.1"
```

### 3. Invalid URLs
TestPyPI validates URLs. Make sure all URLs in `pyproject.toml` are:
- Valid URLs (not placeholders)
- Accessible (or at least valid format)
- Use HTTPS

### 4. Missing or Invalid Metadata
Ensure all required fields are present:
- `name` - must be unique
- `version` - must be valid version format
- `description` - required
- `authors` - at least one author
- `license` - valid SPDX expression

## Debugging Steps

1. **Check package validity:**
   ```bash
   python3 -m twine check dist/*
   ```

2. **Try verbose upload:**
   ```bash
   python3 -m twine upload --repository testpypi dist/* --verbose
   ```

3. **Check if package exists:**
   Visit: https://test.pypi.org/project/patient-aggregator/

4. **Verify API token:**
   - Make sure you're using the full token including `pypi-` prefix
   - Token should be for TestPyPI (not production PyPI)
   - Token scope should be "Entire account"

5. **Check package name availability:**
   Try searching: https://test.pypi.org/search/?q=patient-aggregator

## Quick Fix: Use Unique Name

If you want to upload immediately without checking, use a unique name:

```toml
name = "patient-aggregator-yashlahoti"
```

Then rebuild:
```bash
python3 -m build
python3 -m twine upload --repository testpypi dist/*
```

