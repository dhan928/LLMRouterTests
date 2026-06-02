# Test Guide

This simplified repo only needs one focused test file:

```powershell
python -m pytest
```

`pyproject.toml` points pytest at:

```text
tests/test_simple_routers.py
```

That file checks:

- `KNNRouter.fit()` and `KNNRouter.route()`
- `SVMRouter.fit()` and `SVMRouter.route()`
- `MLPRouter.fit()` and `MLPRouter.route()`
- the error raised when routing before fitting

Install test dependencies first:

```powershell
python -m pip install -e ".[dev]"
```
