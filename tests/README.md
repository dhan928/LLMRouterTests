# Test Guide

Run:

```powershell
python -m pytest
```

The tests cover:

- training `KNNRouter`, `SVMRouter`, and `MLPRouter` on real text queries
- routing a new plain-English query to `weak` or `strong`
- calling the selected OpenRouter model through a fake client
- rejecting route calls before the router is trained

The tests do not use your OpenRouter API key and do not spend credits.
