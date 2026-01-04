# Quick Fix for 400 Bad Request Error

## Most Likely Causes & Solutions

### Solution 1: Package Name Already Exists (Most Common)

The package name `patient-aggregator` might already be taken on TestPyPI. 

**Quick Fix - Use a unique name:**

1. Edit `pyproject.toml` and change:
   ```toml
   name = "patient-aggregator-yashlahoti"
   ```

2. Rebuild:
   ```bash
   python3 -m build
   ```

3. Upload again:
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```

### Solution 2: Version Already Exists

If version `0.1.0` already exists, increment it:

1. Edit `pyproject.toml`:
   ```toml
   version = "0.1.1"
   ```

2. Rebuild and upload

### Solution 3: Check What's Already There

Visit these URLs to check:
- https://test.pypi.org/project/patient-aggregator/
- https://test.pypi.org/search/?q=patient-aggregator

## Recommended Action

**Use a unique package name** to avoid conflicts:

```toml
name = "patient-aggregator-yashlahoti"
```

This ensures your package name is unique and won't conflict with others.

## After Fixing

**macOS/Linux:**
1. Rebuild: `python3 -m build`
2. Check: `python3 -m twine check dist/*`
3. Upload: `python3 -m twine upload --repository testpypi dist/*`

**Windows:**
1. Rebuild: `python -m build`
2. Check: `python -m twine check dist\*` (PowerShell) or list files explicitly (Command Prompt)
3. Upload: `python -m twine upload --repository testpypi dist\*` (PowerShell)

Enter your TestPyPI API token when prompted (the full token including `pypi-` prefix).

**Note:** For complete Windows command reference, see `WINDOWS_COMMANDS.md`

