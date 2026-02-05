# Contributing to Semantic API Engine

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/petermtj/semanticapi-engine.git
cd semanticapi-engine
make setup
source .venv/bin/activate
```

## Running Tests

```bash
make test
```

## Adding a New Provider

The easiest way to contribute is by adding support for a new API provider. No code changes needed!

1. Create a JSON file in `providers/` (e.g., `providers/myapi.json`)
2. Follow the provider schema format:

```json
{
  "provider": "myapi",
  "name": "My API",
  "description": "What this API does",
  "base_url": "https://api.myapi.com/v1",
  "auth": {
    "type": "bearer",
    "header": "Authorization",
    "prefix": "Bearer"
  },
  "capabilities": [
    {
      "id": "list_items",
      "name": "List Items",
      "description": "Clear description of what this does",
      "semantic_tags": ["items", "list", "query"],
      "endpoint": {
        "method": "GET",
        "path": "/items",
        "params": {
          "limit": {
            "type": "integer",
            "required": false,
            "description": "Max results"
          }
        }
      }
    }
  ]
}
```

3. Test it:
```bash
python -c "from semanticapi.provider_loader import load_providers; print([p.name for p in load_providers()])"
```

4. Submit a PR!

## Code Style

- Python 3.10+ type hints
- Docstrings on all public functions
- Keep it simple — avoid unnecessary abstractions

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b add-sendgrid-provider`)
3. Make your changes
4. Run tests (`make test`)
5. Commit with a clear message
6. Push and open a PR

## Reporting Issues

- Use GitHub Issues
- Include steps to reproduce
- Include error messages and stack traces
- Mention your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 license.
