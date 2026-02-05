"""
Semantic API Engine
===================

Natural language interface to any API.

Usage:
    from semanticapi import SemanticAPI

    api = SemanticAPI()
    api.add_provider("stripe", api_key="sk_live_...")

    result = api.fetch("get my last 5 stripe payments")
    result = api.execute("send SMS to +1234567890 saying 'Hello!'")

Or use the agentic processor for multi-step reasoning:

    from semanticapi import AgenticProcessor

    processor = AgenticProcessor()
    processor.add_provider("stripe", api_key="sk_live_...")
    result = processor.process("get my stripe balance and list recent payments")
"""

__version__ = "0.1.0"
__author__ = "Semantic API Contributors"
__license__ = "AGPL-3.0"

from semanticapi.engine import (
    SemanticAPI,
    IntentType,
    ParsedIntent,
    APIEndpoint,
    ProviderConfig,
    SemanticResponse,
    ExecutionEngine,
    APIRegistry,
    SchemaTransformer,
    create_api,
)

from semanticapi.processor import AgenticProcessor, AgentResult, process_query

from semanticapi.provider_loader import (
    LoadedProvider,
    load_providers,
    load_json_providers,
    get_static_providers,
)

__all__ = [
    # Core SDK
    "SemanticAPI",
    "create_api",
    # Types
    "IntentType",
    "ParsedIntent",
    "APIEndpoint",
    "ProviderConfig",
    "SemanticResponse",
    # Engine components
    "ExecutionEngine",
    "APIRegistry",
    "SchemaTransformer",
    # Agentic processor
    "AgenticProcessor",
    "AgentResult",
    "process_query",
    # Provider loading
    "LoadedProvider",
    "load_providers",
    "load_json_providers",
    "get_static_providers",
]
