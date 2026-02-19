# AGENTS.md — Semantic API Engine Agent Teams

## Project Context
Open source Semantic API engine. Python package, AGPL-3.0. Read CLAUDE.md first.

## Team Roles
- **Core Agent** — `semanticapi/` module, core engine logic
- **Provider Agent** — `providers/` directory, adding/improving API providers
- **Docs Agent** — README, CONTRIBUTING, examples, docstrings
- **QA Agent** — Tests, CI, package build validation

## Rules
- This is OPEN SOURCE — code quality and documentation matter extra
- Every public function needs docstrings
- Changes should maintain backward compatibility
- Run tests before committing
- Keep `examples/` up to date with any API changes
- AGPL-3.0 license — ensure all contributions are compatible


## 🚨 Sub-Agent Rules (Universal)
- **Read this ENTIRE AGENTS.md before making ANY changes**
- **Do NOT delete existing functionality** — preserve everything that works
- **Do NOT create duplicate databases, endpoints, or files**
- **Read the ENTIRE file before editing ANY part of it**
- **PSG repos: NEVER `git push` to remote** — only Peter pushes manually
