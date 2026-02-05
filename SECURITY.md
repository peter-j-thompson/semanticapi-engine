# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Current          |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email **security@semanticapi.dev** with details
3. Include steps to reproduce if possible
4. We'll respond within 48 hours

## Security Considerations

### API Keys
- Never commit API keys to version control
- Use environment variables or `.env` files (included in `.gitignore`)
- The engine does NOT store credentials persistently — they're held in memory only

### Self-Hosting
- The engine has no built-in authentication. If exposing publicly, put it behind a reverse proxy with auth.
- CORS is permissive by default (`*`). Restrict in production.
- Consider running in a private network or Docker network.

### LLM Prompt Injection
- The agentic processor sends user queries to an LLM. Be aware of prompt injection risks.
- The engine only executes API calls to providers you've configured with credentials.
- It cannot access providers without credentials, limiting blast radius.
