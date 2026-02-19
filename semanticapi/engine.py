"""
Semantic API Engine — Core Types and SDK
=========================================

The main SemanticAPI class and all core data types for natural language
API interaction.

Usage:
    from semanticapi import SemanticAPI

    api = SemanticAPI()
    api.add_provider("stripe", api_key="sk_live_...")

    payments = api.fetch("get my last 5 stripe payments")
    api.execute("send SMS to +1234567890 saying 'Hello!'")
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from urllib.parse import urljoin

import httpx


T = TypeVar("T")


# =============================================================================
# Core Data Types
# =============================================================================


class IntentType(Enum):
    """Types of API operations."""

    FETCH = "fetch"
    EXECUTE = "execute"
    STREAM = "stream"


@dataclass
class ParsedIntent:
    """Structured representation of a natural language request."""

    intent_type: IntentType
    provider: str | None
    action: str
    entities: dict[str, Any]
    filters: dict[str, Any]
    output_hints: dict[str, Any]
    confidence: float
    raw_query: str

    def to_dict(self) -> dict:
        return {
            "intent_type": self.intent_type.value,
            "provider": self.provider,
            "action": self.action,
            "entities": self.entities,
            "filters": self.filters,
            "output_hints": self.output_hints,
            "confidence": self.confidence,
        }


@dataclass
class APIEndpoint:
    """Definition of an API endpoint."""

    provider: str
    name: str
    method: str
    path: str
    description: str
    parameters: dict[str, dict]
    response_schema: dict | None = None
    rate_limit: int | None = None

    def matches_action(self, action: str) -> float:
        """Score how well this endpoint matches an action description."""
        action_lower = action.lower()
        name_lower = self.name.lower()
        desc_lower = self.description.lower()

        if action_lower in name_lower or name_lower in action_lower:
            return 0.9

        synonyms = {
            "get": ["list", "fetch", "show", "retrieve"],
            "list": ["get", "fetch", "show", "retrieve"],
            "create": ["make", "add", "new"],
            "send": ["create", "make"],
        }
        stop_words = {"to", "the", "a", "an", "my", "all"}

        action_words = [
            w for w in action_lower.replace("_", " ").split() if w not in stop_words
        ]
        name_words = [
            w for w in name_lower.replace("_", " ").split() if w not in stop_words
        ]

        word_matches = 0
        synonym_matches = 0

        for aw in action_words:
            for nw in name_words:
                if aw == nw:
                    word_matches += 1
                elif aw in synonyms and nw in synonyms.get(aw, []):
                    synonym_matches += 1
                elif nw in synonyms and aw in synonyms.get(nw, []):
                    synonym_matches += 1

        if word_matches >= 1:
            return 0.7 + (0.1 * min(word_matches + synonym_matches, 2))

        keywords = [
            w for w in action_lower.split("_") if len(w) > 3 and w not in stop_words
        ]
        matches = sum(1 for kw in keywords if kw in desc_lower)
        if matches > 0 and len(keywords) > 0:
            return 0.4 + (0.2 * matches / len(keywords))

        return 0.0


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""

    name: str
    base_url: str
    auth_type: str  # "api_key", "bearer", "basic", "oauth2"
    credentials: dict[str, str]
    endpoints: list[APIEndpoint] = field(default_factory=list)
    rate_limit: int = 60

    def get_auth_header(self) -> dict[str, str]:
        """Build authentication headers."""
        if self.auth_type == "bearer":
            return {
                "Authorization": f"Bearer {self.credentials.get('api_key', '')}"
            }
        elif self.auth_type == "api_key":
            return {
                "Authorization": f"Bearer {self.credentials.get('api_key', '')}"
            }
        elif self.auth_type == "basic":
            import base64

            user = self.credentials.get(
                "username", self.credentials.get("account_sid", "")
            )
            pwd = self.credentials.get(
                "password", self.credentials.get("auth_token", "")
            )
            encoded = base64.b64encode(f"{user}:{pwd}".encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        return {}


@dataclass
class SemanticResponse(Generic[T]):
    """Normalized response from any API."""

    success: bool
    data: T | None
    provider: str
    provider_id: str | None
    operation: str
    timestamp: datetime
    raw_response: dict | None = None
    error: str | None = None
    needs_auth: bool = False
    auth_url: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "provider": self.provider,
            "provider_id": self.provider_id,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


# =============================================================================
# Intent Parser (Pattern-based)
# =============================================================================


class IntentParser:
    """
    Parses natural language queries into structured intents.

    Uses pattern matching for common cases with optional LLM fallback.
    """

    PATTERNS = {
        r"(?:get|list|show|fetch).*?(?:last\s+)?(\d+)?\s*(?:stripe\s+)?payments?": (
            "stripe",
            "list_payments",
        ),
        r"(?:get|list|show|fetch).*?(?:stripe\s+)?customers?": (
            "stripe",
            "list_customers",
        ),
        r"(?:get|show|check|what(?:'?s)?|my).*?(?:stripe\s+)?balance|balance.*?(?:for\s+)?stripe": (
            "stripe",
            "get_balance",
        ),
        r"(?:create|make|charge).*?payment.*?\$?([\d.]+)": (
            "stripe",
            "create_payment",
        ),
        r"refund.*?payment.*?([a-zA-Z0-9_]+)": ("stripe", "create_refund"),
        r"send\s+(?:an?\s+)?(?:sms|text|message)\s+to\s+(\+?[\d\-\s]+)\s+(?:saying|with|:)\s*['\"]?(.+?)['\"]?\s*$": (
            "twilio",
            "send_sms",
        ),
        r"send\s+(?:an?\s+)?(?:sms|text|message)\s+to\s+(\+?[\d\-\s]+)": (
            "twilio",
            "send_sms",
        ),
        r"(?:text|sms|message)\s+(\+?[\d\-\s]+)": ("twilio", "send_sms"),
        r"(?:get|list|show|fetch).*?(?:sms|text|message).*?history": (
            "twilio",
            "list_messages",
        ),
        r"call\s+(\+?[\d\-\s]+)": ("twilio", "make_call"),
        r"generate.*?(?:an?\s+)?image.*?(?:of\s+)?(.+)": (
            "openai",
            "create_image",
        ),
        r"transcribe.*?(?:audio|recording|file)": ("openai", "transcribe"),
        r"embed(?:ding)?.*?['\"](.+)['\"]": ("openai", "create_embedding"),
        r"(?:ask|tell|chat|prompt|complete|write|create|explain).*?(?:openai|gpt|chatgpt|ai)\s*[:\-]?\s*(.+)": (
            "openai",
            "create_completion",
        ),
        r"(?:openai|gpt|chatgpt)\s+(?:ask|tell|write|explain|create)\s*[:\-]?\s*(.+)": (
            "openai",
            "create_completion",
        ),
        r"^(?:generate|write|create|make)\s+(?:a\s+)?(?:list|step.?by.?step|guide|summary|outline|essay|article|story|poem|code|script|recipe|plan)": (
            "openai",
            "create_completion",
        ),
        r"^(?:explain|describe|summarize|translate|rewrite|improve|edit)\s+": (
            "openai",
            "create_completion",
        ),
        r"^(?:what is|what are|how to|how do|why does|can you|could you|please)\s+": (
            "openai",
            "create_completion",
        ),
    }

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    def parse(
        self, query: str, intent_type: IntentType = IntentType.FETCH
    ) -> ParsedIntent:
        """Parse a natural language query into structured intent."""
        query_lower = query.lower().strip()

        for pattern, (provider, action) in self.PATTERNS.items():
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._build_intent_from_pattern(
                    query, match, provider, action, intent_type
                )

        if self.llm_client:
            return self._parse_with_llm(query, intent_type)

        return self._basic_parse(query, intent_type)

    def _build_intent_from_pattern(
        self,
        query: str,
        match: re.Match,
        provider: str,
        action: str,
        intent_type: IntentType,
    ) -> ParsedIntent:
        entities = {}
        filters = {}
        groups = match.groups()

        if action == "list_payments" and groups[0]:
            filters["limit"] = int(groups[0])
        elif action == "create_payment" and groups[0]:
            entities["amount"] = float(groups[0])
        elif action == "send_sms" and groups and groups[0]:
            entities["to"] = groups[0].strip()
            if len(groups) >= 2 and groups[1]:
                entities["body"] = groups[1].strip()
            else:
                entities["body"] = "Hello from Semantic API"
        elif action == "create_image" and groups[0]:
            entities["prompt"] = groups[0].strip()
        elif action == "create_refund" and groups[0]:
            entities["payment_id"] = groups[0]

        return ParsedIntent(
            intent_type=intent_type,
            provider=provider,
            action=action,
            entities=entities,
            filters=filters,
            output_hints={},
            confidence=0.85,
            raw_query=query,
        )

    def _parse_with_llm(self, query: str, intent_type: IntentType) -> ParsedIntent:
        prompt = f"""Parse this API request into structured format.

Query: "{query}"

Respond in JSON:
{{
    "provider": "provider_name or null",
    "action": "action_name",
    "entities": {{}},
    "filters": {{}},
    "confidence": 0.0-1.0
}}"""
        try:
            response = self.llm_client(prompt)
            data = json.loads(response)
            return ParsedIntent(
                intent_type=intent_type,
                provider=data.get("provider"),
                action=data.get("action", "unknown"),
                entities=data.get("entities", {}),
                filters=data.get("filters", {}),
                output_hints={},
                confidence=data.get("confidence", 0.5),
                raw_query=query,
            )
        except (json.JSONDecodeError, Exception):
            return self._basic_parse(query, intent_type)

    def _basic_parse(self, query: str, intent_type: IntentType) -> ParsedIntent:
        query_lower = query.lower()

        provider = None
        for p in ["stripe", "twilio", "openai", "github", "slack", "notion"]:
            if p in query_lower:
                provider = p
                break

        action = "unknown"
        if any(kw in query_lower for kw in ["get", "list", "show", "fetch"]):
            action = "list"
        elif any(kw in query_lower for kw in ["send", "create", "make"]):
            action = "create"

        # Detect SMS/text messaging — route to Twilio
        sms_keywords = ["sms", "text message", "send text", "send message"]
        if not provider and any(kw in query_lower for kw in sms_keywords):
            provider = "twilio"
            action = "send_sms"
            phone_match = re.search(r'(\+?[\d\-\s]{10,})', query)
            entities = {}
            if phone_match:
                entities["to"] = phone_match.group(1).strip()
            return ParsedIntent(
                intent_type=intent_type,
                provider=provider,
                action=action,
                entities=entities,
                filters={},
                output_hints={},
                confidence=0.7,
                raw_query=query,
            )

        # If no provider detected but looks like AI/text generation, route to OpenAI
        generation_keywords = [
            "generate", "write", "create", "make", "explain", "describe",
            "summarize", "translate", "what is", "how to", "list of",
            "step by step", "help me", "can you", "please", "tell me",
        ]
        if not provider and any(kw in query_lower for kw in generation_keywords):
            return ParsedIntent(
                intent_type=intent_type,
                provider="openai",
                action="create_completion",
                entities={"prompt": query},
                filters={},
                output_hints={},
                confidence=0.6,
                raw_query=query,
            )

        return ParsedIntent(
            intent_type=intent_type,
            provider=provider,
            action=action,
            entities={},
            filters={},
            output_hints={},
            confidence=0.3,
            raw_query=query,
        )


# =============================================================================
# API Registry
# =============================================================================


class APIRegistry:
    """Registry of available API providers and their endpoints."""

    def __init__(self):
        self.providers: dict[str, ProviderConfig] = {}
        self._load_builtin_providers()

    def _load_builtin_providers(self):
        """Load built-in provider definitions."""
        stripe = ProviderConfig(
            name="stripe",
            base_url="https://api.stripe.com/v1",
            auth_type="bearer",
            credentials={},
            endpoints=[
                APIEndpoint(
                    provider="stripe",
                    name="list_payments",
                    method="GET",
                    path="/charges",
                    description="List all payment charges",
                    parameters={
                        "limit": {
                            "type": "int",
                            "required": False,
                            "description": "Number of results",
                        },
                    },
                ),
                APIEndpoint(
                    provider="stripe",
                    name="list_customers",
                    method="GET",
                    path="/customers",
                    description="List all customers",
                    parameters={
                        "limit": {
                            "type": "int",
                            "required": False,
                            "description": "Number of results",
                        },
                        "email": {
                            "type": "string",
                            "required": False,
                            "description": "Filter by email",
                        },
                    },
                ),
                APIEndpoint(
                    provider="stripe",
                    name="create_payment",
                    method="POST",
                    path="/payment_intents",
                    description="Create a new payment intent",
                    parameters={
                        "amount": {
                            "type": "int",
                            "required": True,
                            "description": "Amount in cents",
                        },
                        "currency": {
                            "type": "string",
                            "required": True,
                            "description": "Currency code",
                        },
                    },
                ),
                APIEndpoint(
                    provider="stripe",
                    name="create_refund",
                    method="POST",
                    path="/refunds",
                    description="Refund a payment",
                    parameters={
                        "charge": {
                            "type": "string",
                            "required": True,
                            "description": "Charge ID to refund",
                        },
                    },
                ),
                APIEndpoint(
                    provider="stripe",
                    name="get_balance",
                    method="GET",
                    path="/balance",
                    description="Get account balance",
                    parameters={},
                ),
            ],
        )
        self.providers["stripe"] = stripe

        twilio = ProviderConfig(
            name="twilio",
            base_url="https://api.twilio.com/2010-04-01",
            auth_type="basic",
            credentials={},
            endpoints=[
                APIEndpoint(
                    provider="twilio",
                    name="send_sms",
                    method="POST",
                    path="/Accounts/{account_sid}/Messages.json",
                    description="Send an SMS message",
                    parameters={
                        "To": {
                            "type": "string",
                            "required": True,
                            "description": "Recipient phone number",
                        },
                        "From": {
                            "type": "string",
                            "required": True,
                            "description": "Sender phone number",
                        },
                        "Body": {
                            "type": "string",
                            "required": True,
                            "description": "Message text",
                        },
                    },
                ),
                APIEndpoint(
                    provider="twilio",
                    name="list_messages",
                    method="GET",
                    path="/Accounts/{account_sid}/Messages.json",
                    description="List SMS messages",
                    parameters={
                        "PageSize": {
                            "type": "int",
                            "required": False,
                            "description": "Results per page",
                        },
                    },
                ),
            ],
        )
        self.providers["twilio"] = twilio

        openai_provider = ProviderConfig(
            name="openai",
            base_url="https://api.openai.com/v1",
            auth_type="bearer",
            credentials={},
            endpoints=[
                APIEndpoint(
                    provider="openai",
                    name="create_completion",
                    method="POST",
                    path="/chat/completions",
                    description="Generate text completion",
                    parameters={
                        "model": {
                            "type": "string",
                            "required": True,
                            "description": "Model ID",
                        },
                        "messages": {
                            "type": "array",
                            "required": True,
                            "description": "Chat messages",
                        },
                    },
                ),
                APIEndpoint(
                    provider="openai",
                    name="create_image",
                    method="POST",
                    path="/images/generations",
                    description="Generate an image from a prompt",
                    parameters={
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "Image description",
                        },
                        "n": {
                            "type": "int",
                            "required": False,
                            "description": "Number of images",
                        },
                        "size": {
                            "type": "string",
                            "required": False,
                            "description": "Image size",
                        },
                    },
                ),
                APIEndpoint(
                    provider="openai",
                    name="create_embedding",
                    method="POST",
                    path="/embeddings",
                    description="Create text embeddings",
                    parameters={
                        "model": {
                            "type": "string",
                            "required": True,
                            "description": "Model ID",
                        },
                        "input": {
                            "type": "string",
                            "required": True,
                            "description": "Text to embed",
                        },
                    },
                ),
                APIEndpoint(
                    provider="openai",
                    name="transcribe",
                    method="POST",
                    path="/audio/transcriptions",
                    description="Transcribe audio to text",
                    parameters={
                        "file": {
                            "type": "file",
                            "required": True,
                            "description": "Audio file",
                        },
                        "model": {
                            "type": "string",
                            "required": True,
                            "description": "Model ID",
                        },
                    },
                ),
            ],
        )
        self.providers["openai"] = openai_provider

    def register_provider(self, config: ProviderConfig):
        """Register a provider configuration."""
        self.providers[config.name] = config

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get provider by name."""
        return self.providers.get(name)

    def configure_credentials(self, provider: str, **credentials):
        """Set credentials for a provider."""
        if provider in self.providers:
            self.providers[provider].credentials = credentials

    def find_endpoint(self, provider: str, action: str) -> APIEndpoint | None:
        """Find the best matching endpoint for an action."""
        if provider not in self.providers:
            return None

        endpoints = self.providers[provider].endpoints

        for ep in endpoints:
            if ep.name == action:
                return ep

        best_match = None
        best_score = 0.0
        for ep in endpoints:
            score = ep.matches_action(action)
            if score > best_score:
                best_score = score
                best_match = ep

        return best_match if best_score > 0.3 else None

    def list_capabilities(self, provider: str | None = None) -> list[dict]:
        """List available capabilities."""
        caps = []
        providers = (
            [self.providers[provider]] if provider else self.providers.values()
        )

        for p in providers:
            for ep in p.endpoints:
                caps.append(
                    {
                        "provider": p.name,
                        "action": ep.name,
                        "description": ep.description,
                        "method": ep.method,
                    }
                )
        return caps


# =============================================================================
# Schema Transformer
# =============================================================================


class SchemaTransformer:
    """Transforms provider-specific responses into normalized semantic format."""

    TRANSFORMS = {
        "stripe": {
            "payment": {
                "id": lambda r: r.get("id"),
                "amount_cents": lambda r: r.get("amount"),
                "amount_display": lambda r: f"${r.get('amount', 0) / 100:.2f}",
                "currency": lambda r: r.get("currency", "usd").upper(),
                "status": lambda r: r.get("status"),
                "created_at": lambda r: datetime.fromtimestamp(
                    r.get("created", 0), tz=timezone.utc
                ).isoformat(),
                "customer_id": lambda r: r.get("customer"),
                "description": lambda r: r.get("description"),
            },
            "customer": {
                "id": lambda r: r.get("id"),
                "email": lambda r: r.get("email"),
                "name": lambda r: r.get("name"),
                "created_at": lambda r: datetime.fromtimestamp(
                    r.get("created", 0), tz=timezone.utc
                ).isoformat(),
            },
        },
        "twilio": {
            "message": {
                "id": lambda r: r.get("sid"),
                "from": lambda r: r.get("from"),
                "to": lambda r: r.get("to"),
                "body": lambda r: r.get("body"),
                "status": lambda r: r.get("status"),
                "direction": lambda r: r.get("direction"),
                "created_at": lambda r: r.get("date_created"),
            },
        },
        "openai": {
            "completion": {
                "id": lambda r: r.get("id"),
                "content": lambda r: r.get("choices", [{}])[0]
                .get("message", {})
                .get("content"),
                "model": lambda r: r.get("model"),
                "usage": lambda r: r.get("usage"),
            },
            "image": {
                "url": lambda r: r.get("data", [{}])[0].get("url"),
                "revised_prompt": lambda r: r.get("data", [{}])[0].get(
                    "revised_prompt"
                ),
            },
        },
    }

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    def transform(
        self, raw_response: dict | list, provider: str, action: str
    ) -> dict | list:
        """Transform raw API response to normalized format."""
        entity_type = self._infer_entity_type(action)
        rules = self.TRANSFORMS.get(provider, {}).get(entity_type, {})

        if not rules:
            return self._basic_transform(raw_response, provider)

        if isinstance(raw_response, list):
            return [self._apply_rules(item, rules) for item in raw_response]

        if isinstance(raw_response, dict) and "data" in raw_response:
            transformed_data = [
                self._apply_rules(item, rules)
                for item in raw_response.get("data", [])
            ]
            return {
                "data": transformed_data,
                "has_more": raw_response.get("has_more", False),
                "total_count": len(transformed_data),
            }

        return self._apply_rules(raw_response, rules)

    def _infer_entity_type(self, action: str) -> str:
        if "payment" in action or "charge" in action:
            return "payment"
        elif "customer" in action:
            return "customer"
        elif "message" in action or "sms" in action:
            return "message"
        elif "completion" in action:
            return "completion"
        elif "image" in action:
            return "image"
        return "generic"

    def _apply_rules(self, data: dict, rules: dict) -> dict:
        result = {"_provider_data": data}
        for field_name, transformer in rules.items():
            try:
                result[field_name] = transformer(data)
            except Exception:
                result[field_name] = None
        return result

    def _basic_transform(
        self, data: dict | list, provider: str
    ) -> dict | list:
        if isinstance(data, list):
            return [{"_provider": provider, **item} for item in data]
        return {"_provider": provider, **data}


# =============================================================================
# Execution Engine
# =============================================================================


class ExecutionEngine:
    """Executes API calls with retry logic, rate limiting, and caching."""

    def __init__(
        self,
        registry: APIRegistry,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.registry = registry
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.Client | None = None
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = 60

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def execute(
        self,
        provider: str,
        endpoint: APIEndpoint,
        params: dict,
    ) -> dict:
        """Execute an API call."""
        config = self.registry.get_provider(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")

        if not config.credentials:
            raise ValueError(f"No credentials configured for {provider}")

        url = self._build_url(config, endpoint, params)
        headers = config.get_auth_header()
        headers["Content-Type"] = "application/json"

        if endpoint.method == "GET":
            cache_key = self._cache_key(url, params)
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._make_request(
                    endpoint.method, url, headers, params, config
                )
                if endpoint.method == "GET":
                    self._set_cached(cache_key, response)
                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code < 500:
                    raise
                import time

                time.sleep(2**attempt)
            except httpx.RequestError as e:
                last_error = e
                import time

                time.sleep(2**attempt)

        raise last_error or Exception("Request failed after retries")

    def _build_url(
        self, config: ProviderConfig, endpoint: APIEndpoint, params: dict
    ) -> str:
        path = endpoint.path
        if "{account_sid}" in path:
            account_sid = config.credentials.get("account_sid", "")
            path = path.replace("{account_sid}", account_sid)
        return urljoin(config.base_url + "/", path.lstrip("/"))

    def _make_request(
        self,
        method: str,
        url: str,
        headers: dict,
        params: dict,
        config: ProviderConfig,
    ) -> dict:
        if config.name == "twilio" and method == "POST":
            response = self.client.request(
                method, url, headers=headers, data=params
            )
        elif method == "GET":
            response = self.client.request(
                method, url, headers=headers, params=params
            )
        else:
            response = self.client.request(
                method, url, headers=headers, json=params
            )
        response.raise_for_status()
        return response.json()

    def _cache_key(self, url: str, params: dict) -> str:
        data = f"{url}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            timestamp, data = self._cache[key]
            if (datetime.now(timezone.utc) - timestamp).seconds < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: Any):
        self._cache[key] = (datetime.now(timezone.utc), data)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


# =============================================================================
# Main SDK
# =============================================================================


class SemanticAPI:
    """
    Natural language interface to any API.

    Usage:
        api = SemanticAPI()
        api.add_provider("stripe", api_key="sk_live_...")

        payments = api.fetch("get my last 5 stripe payments")
        api.execute("send SMS to +1234567890 saying 'Hello!'")
    """

    def __init__(
        self,
        api_key: str | None = None,
        llm_client: Optional[Callable] = None,
        debug: bool = False,
    ):
        self.api_key = api_key
        self.debug = debug

        self.registry = APIRegistry()
        self.parser = IntentParser(llm_client=llm_client)
        self.transformer = SchemaTransformer(llm_client=llm_client)
        self.engine = ExecutionEngine(self.registry)

    def add_provider(self, name: str, **credentials):
        """
        Add and configure an API provider.

        Examples:
            api.add_provider("stripe", api_key="sk_live_...")
            api.add_provider("twilio", account_sid="...", auth_token="...")
        """
        if name not in self.registry.providers:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {list(self.registry.providers.keys())}"
            )
        self.registry.configure_credentials(name, **credentials)

    def fetch(self, query: str) -> SemanticResponse:
        """Fetch data using natural language."""
        return self._process(query, IntentType.FETCH)

    def execute(self, command: str) -> SemanticResponse:
        """Execute an action using natural language."""
        return self._process(command, IntentType.EXECUTE)

    def _process(self, query: str, intent_type: IntentType) -> SemanticResponse:
        timestamp = datetime.now(timezone.utc)

        try:
            intent = self.parser.parse(query, intent_type)

            if not intent.provider:
                return SemanticResponse(
                    success=False,
                    data=None,
                    provider="unknown",
                    provider_id=None,
                    operation=intent.action,
                    timestamp=timestamp,
                    error="Could not determine provider from query.",
                )

            endpoint = self.registry.find_endpoint(intent.provider, intent.action)

            if not endpoint:
                caps = self.registry.list_capabilities(intent.provider)
                available = [c["action"] for c in caps]
                return SemanticResponse(
                    success=False,
                    data=None,
                    provider=intent.provider,
                    provider_id=None,
                    operation=intent.action,
                    timestamp=timestamp,
                    error=f"Unknown action '{intent.action}' for {intent.provider}. "
                    f"Available: {available}",
                )

            params = self._build_params(intent, endpoint)
            raw_response = self.engine.execute(intent.provider, endpoint, params)
            normalized = self.transformer.transform(
                raw_response, intent.provider, intent.action
            )

            provider_id = None
            if isinstance(normalized, dict):
                provider_id = normalized.get("id")

            return SemanticResponse(
                success=True,
                data=normalized,
                provider=intent.provider,
                provider_id=provider_id,
                operation=intent.action,
                timestamp=timestamp,
                raw_response=raw_response if self.debug else None,
            )

        except ValueError as e:
            return SemanticResponse(
                success=False,
                data=None,
                provider=intent.provider if "intent" in locals() else "unknown",
                provider_id=None,
                operation=intent.action if "intent" in locals() else "unknown",
                timestamp=timestamp,
                error=str(e),
                needs_auth="credentials" in str(e).lower(),
            )
        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.json()
            except Exception:
                error_body = e.response.text

            return SemanticResponse(
                success=False,
                data=None,
                provider=intent.provider if "intent" in locals() else "unknown",
                provider_id=None,
                operation=intent.action if "intent" in locals() else "unknown",
                timestamp=timestamp,
                error=f"API error ({e.response.status_code}): {error_body}",
            )
        except Exception as e:
            return SemanticResponse(
                success=False,
                data=None,
                provider=intent.provider if "intent" in locals() else "unknown",
                provider_id=None,
                operation=intent.action if "intent" in locals() else "unknown",
                timestamp=timestamp,
                error=f"Unexpected error: {str(e)}",
            )

    def _build_params(self, intent: ParsedIntent, endpoint: APIEndpoint) -> dict:
        params = {}

        entity_to_param = {
            "amount": "amount",
            "customer_id": "customer",
            "payment_id": "charge",
            "to": "To",
            "body": "Body",
            "from": "From",
            "prompt": "prompt",
            "text": "input",
        }

        for entity_name, value in intent.entities.items():
            param_name = entity_to_param.get(entity_name, entity_name)
            if param_name in endpoint.parameters:
                params[param_name] = value
            else:
                params[entity_name] = value

        for filter_name, value in intent.filters.items():
            params[filter_name] = value

        # Defaults for specific providers
        if endpoint.provider == "openai":
            if endpoint.name == "create_image":
                params.setdefault("n", 1)
                params.setdefault("size", "1024x1024")
            elif endpoint.name == "create_embedding":
                params.setdefault("model", "text-embedding-ada-002")
            elif endpoint.name == "create_completion":
                params.setdefault("model", "gpt-4o-mini")
                prompt = params.pop("prompt", None) or intent.raw_query
                if "messages" not in params:
                    params["messages"] = [{"role": "user", "content": prompt}]

        if endpoint.provider == "stripe":
            if endpoint.name == "create_payment":
                params.setdefault("currency", "usd")
                if "amount" in params and params["amount"] < 100:
                    params["amount"] = int(params["amount"] * 100)

        if endpoint.provider == "twilio":
            config = self.registry.get_provider("twilio")
            if config and "from_number" in config.credentials:
                params.setdefault("From", config.credentials["from_number"])

        return params

    def capabilities(self, provider: str | None = None) -> list[dict]:
        """List available capabilities."""
        return self.registry.list_capabilities(provider)

    def providers(self) -> list[str]:
        """List available providers."""
        return list(self.registry.providers.keys())

    def is_configured(self, provider: str) -> bool:
        """Check if a provider has credentials configured."""
        config = self.registry.get_provider(provider)
        return bool(config and config.credentials)

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_api(**provider_credentials) -> SemanticAPI:
    """
    Quick factory for creating a configured SemanticAPI.

    Example:
        api = create_api(
            stripe={"api_key": "sk_live_..."},
            twilio={"account_sid": "...", "auth_token": "..."},
        )
    """
    api = SemanticAPI()
    for provider, creds in provider_credentials.items():
        if isinstance(creds, dict):
            api.add_provider(provider, **creds)
    return api
