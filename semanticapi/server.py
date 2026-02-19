"""
Semantic API Server
====================

A lightweight FastAPI server for the Semantic API Engine.

Provides a REST API for processing natural language queries against
configured API providers.

Run directly:
    python -m semanticapi.server

Or with uvicorn:
    uvicorn semanticapi.server:app --host 0.0.0.0 --port 8080
"""

import os
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from semanticapi.processor import AgenticProcessor
from semanticapi.provider_loader import load_providers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROVIDERS_DIR = os.environ.get(
    "PROVIDERS_DIR",
    str(Path(__file__).parent.parent / "providers"),
)

AI_PROVIDER = os.environ.get("AI_PROVIDER", "anthropic")
AI_MODEL = os.environ.get("AI_MODEL", None)
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semantic API Engine",
    description="Natural language interface to any API",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for self-hosted usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Provider credentials (set via /api/providers/{name}/configure)
provider_credentials: dict[str, dict] = {}

# Loaded provider definitions
loaded_providers = []


@app.on_event("startup")
def startup():
    """Load providers on startup."""
    global loaded_providers
    loaded_providers = load_providers(PROVIDERS_DIR)
    provider_names = [p.name for p in loaded_providers]
    print(f"[SemanticAPI] Loaded {len(loaded_providers)} providers: {', '.join(provider_names)}")
    print(f"[SemanticAPI] AI provider: {AI_PROVIDER}")
    print(f"[SemanticAPI] Providers dir: {PROVIDERS_DIR}")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /api/query."""

    query: str
    """Natural language query (e.g., 'get my stripe balance')."""

    credentials: Optional[dict[str, dict]] = None
    """Optional per-request credentials. Format: {"stripe": {"api_key": "sk_..."}}."""

    ai_provider: Optional[str] = None
    """Override AI provider for this request (anthropic, openai, groq, ollama)."""

    model: Optional[str] = None
    """Override AI model for this request."""


class QueryResponse(BaseModel):
    """Response body for /api/query."""

    success: bool
    response: str
    status: str  # "complete" | "needs_input" | "error"
    question: Optional[str] = None
    options: Optional[list[str]] = None
    steps: Optional[list[dict]] = None
    data: Optional[Any] = None


class ConfigureRequest(BaseModel):
    """Request body for /api/providers/{name}/configure."""

    credentials: dict[str, str]
    """Provider credentials (e.g., {"api_key": "sk_live_..."})."""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page."""
    providers_html = ""
    for p in loaded_providers:
        cap_count = len(p.capabilities)
        providers_html += f"<li><strong>{p.name}</strong> — {p.description} ({cap_count} capabilities)</li>\n"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Semantic API Engine</title>
    <style>
        body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 60px auto; padding: 0 20px; color: #333; }}
        h1 {{ font-size: 2.5em; margin-bottom: 0.2em; }}
        .subtitle {{ color: #666; font-size: 1.2em; margin-bottom: 2em; }}
        a {{ color: #2563eb; }}
        code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        pre {{ background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; }}
        .providers {{ margin: 1.5em 0; }}
        .links {{ margin-top: 2em; }}
        .links a {{ margin-right: 1.5em; }}
    </style>
</head>
<body>
    <h1>⚡ Semantic API Engine</h1>
    <p class="subtitle">Natural language interface to any API</p>

    <h2>Quick Start</h2>
    <pre>curl -X POST http://localhost:8080/api/query \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "get my stripe balance", "credentials": {{"stripe": {{"api_key": "sk_live_..."}}}}}}'</pre>

    <h2>Loaded Providers</h2>
    <ul class="providers">
        {providers_html}
    </ul>

    <div class="links">
        <a href="/docs">📖 API Docs</a>
        <a href="/redoc">📘 ReDoc</a>
        <a href="/health">💚 Health</a>
        <a href="/api/providers">📋 Providers</a>
    </div>

    <p style="margin-top: 3em; color: #999; font-size: 0.85em;">
        Powered by <a href="https://github.com/petermtj/semanticapi-engine">Semantic API Engine</a> •
        Hosted version at <a href="https://semanticapi.dev">semanticapi.dev</a>
    </p>
</body>
</html>"""


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "providers_loaded": len(loaded_providers),
        "ai_provider": AI_PROVIDER,
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Process a natural language query.

    Send a query like "get my stripe balance" and the engine will:
    1. Parse the intent
    2. Call the appropriate API(s)
    3. Return a human-friendly response

    Credentials can be passed per-request or pre-configured via
    POST /api/providers/{name}/configure.
    """
    ai_prov = req.ai_provider or AI_PROVIDER
    model = req.model or AI_MODEL

    try:
        processor = AgenticProcessor(
            providers_dir=PROVIDERS_DIR,
            ai_provider=ai_prov,
            model=model,
            debug=DEBUG,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize processor: {str(e)}",
        )

    # Merge credentials: pre-configured + per-request (per-request wins)
    all_creds = dict(provider_credentials)
    if req.credentials:
        all_creds.update(req.credentials)

    for provider_name, creds in all_creds.items():
        processor.add_provider(provider_name, **creds)

    try:
        result = processor.process(req.query)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}",
        )

    return QueryResponse(
        success=result.success,
        response=result.response,
        status=result.status,
        question=result.question,
        options=result.options,
        steps=result.steps if DEBUG else None,
        data=result.data,
    )


@app.get("/api/providers")
def list_providers():
    """List loaded providers and their capabilities."""
    result = []
    for p in loaded_providers:
        caps = []
        for cap in p.capabilities:
            caps.append(
                {
                    "id": cap.get("id"),
                    "name": cap.get("name"),
                    "description": cap.get("description"),
                    "method": cap.get("endpoint", {}).get("method"),
                }
            )
        result.append(
            {
                "id": p.provider_id,
                "name": p.name,
                "description": p.description,
                "connected": p.provider_id in provider_credentials,
                "capabilities": caps,
            }
        )
    return {"providers": result}


@app.post("/api/providers/{name}/configure")
def configure_provider(name: str, req: ConfigureRequest):
    """
    Set credentials for a provider.

    After configuring, the provider's credentials will be used for
    all subsequent queries (unless overridden per-request).
    """
    # Validate the provider exists
    provider_ids = [p.provider_id for p in loaded_providers]
    if name not in provider_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider: {name}. Available: {provider_ids}",
        )

    provider_credentials[name] = req.credentials
    return {
        "status": "configured",
        "provider": name,
        "message": f"Credentials set for {name}.",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Starting Semantic API Engine on {host}:{port}")
    uvicorn.run(
        "semanticapi.server:app",
        host=host,
        port=port,
        reload=DEBUG,
    )
