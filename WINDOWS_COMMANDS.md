# Windows Commands Reference

This document provides Windows equivalents for all the commands used in this project.

## Building the Package

**macOS/Linux:**
```bash
python3 -m build
```

**Windows:**
```cmd
python -m build
```
or
```cmd
py -m build
```

## Installing Build Tools

**macOS/Linux:**
```bash
python3 -m pip install --upgrade build twine
```

**Windows:**
```cmd
python -m pip install --upgrade build twine
```
or
```cmd
py -m pip install --upgrade build twine
```

## Checking Package Files

**macOS/Linux:**
```bash
python3 -m twine check dist/*
```

**Windows:**
```cmd
python -m twine check dist\patient_aggregator_yashlahoti-0.1.0-py3-none-any.whl dist\patient_aggregator_yashlahoti-0.1.0.tar.gz
```

Or use PowerShell (supports wildcards):
```powershell
python -m twine check dist\*
```

## Uploading to TestPyPI

**macOS/Linux:**
```bash
python3 -m twine upload --repository testpypi dist/*
```

**Windows (Command Prompt):**
```cmd
python -m twine upload --repository testpypi dist\patient_aggregator_yashlahoti-0.1.0-py3-none-any.whl dist\patient_aggregator_yashlahoti-0.1.0.tar.gz
```

**Windows (PowerShell):**
```powershell
python -m twine upload --repository testpypi dist\*
```

## Uploading to Production PyPI

**macOS/Linux:**
```bash
python3 -m twine upload dist/*
```

**Windows (Command Prompt):**
```cmd
python -m twine upload dist\patient_aggregator_yashlahoti-0.1.0-py3-none-any.whl dist\patient_aggregator_yashlahoti-0.1.0.tar.gz
```

**Windows (PowerShell):**
```powershell
python -m twine upload dist\*
```

## Installing from TestPyPI

**macOS/Linux:**
```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps patient-aggregator-yashlahoti
```

**Windows:**
```cmd
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps patient-aggregator-yashlahoti
```

## Installing Dependencies

**macOS/Linux:**
```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ pandas openpyxl pyyaml
```

**Windows:**
```cmd
python -m pip install --index-url https://test.pypi.org/simple/ pandas openpyxl pyyaml
```

## Using the CLI Command

**macOS/Linux:**
```bash
aggregate-patients --input-dir sample_data --output output.csv
```

**Windows:**
```cmd
aggregate-patients --input-dir sample_data --output output.csv
```

(Works the same on Windows!)

## Python API Usage

**macOS/Linux:**
```bash
python3 -c "from patient_aggregator import aggregate_patients; aggregate_patients('sample_data', 'output.csv')"
```

**Windows:**
```cmd
python -c "from patient_aggregator import aggregate_patients; aggregate_patients('sample_data', 'output.csv')"
```

## Cleaning Build Artifacts

**macOS/Linux:**
```bash
rm -rf dist build
```

**Windows (Command Prompt):**
```cmd
rmdir /s /q dist build
```

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force dist, build
```

## Key Differences

1. **Python command**: Use `python` or `py` instead of `python3` on Windows
2. **Path separators**: Use backslashes `\` instead of forward slashes `/` (though forward slashes often work too)
3. **Wildcards**: Command Prompt doesn't support `*` wildcards in some contexts; PowerShell does
4. **File deletion**: Use `rmdir` or `Remove-Item` instead of `rm -rf`

## Recommended: Use PowerShell

PowerShell on Windows is more similar to bash and supports:
- Wildcards (`dist\*`)
- Better error messages
- More Unix-like commands

To open PowerShell:
- Press `Win + X` and select "Windows PowerShell" or "Terminal"
- Or search for "PowerShell" in the Start menu

