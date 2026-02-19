# CLAUDE.md — Semantic API Engine (Open Source)

## Overview
Open source version of Semantic API — natural language interface to any API.
Licensed under AGPL-3.0. This is the community/public version.

## Stack
- **Language:** Python 3.10+
- **Build:** setuptools, pip installable (`pip install semanticapi`)
- **Package:** `semanticapi/` module
- **Providers:** `providers/` directory

## Structure
- `semanticapi/` — Core engine module
- `providers/` — API provider definitions
- `examples/` — Usage examples
- `setup.py` / `pyproject.toml` — Package config

## Key Commands
```bash
# Install locally
pip install -e .

# Run tests
make test   # or pytest

# Build
make build
```

## Docs
- `README.md` — Main docs
- `CONTRIBUTING.md` — Contributor guide
- `CODE_OF_CONDUCT.md` — Community standards
- `SECURITY.md` — Security policy

## Status
🟢 ACTIVE — Open source companion to semanticapi-cloud.
Needs community engagement, good docs, and clean contributor experience.
