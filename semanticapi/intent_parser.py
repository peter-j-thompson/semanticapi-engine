"""
Semantic API Intent Parser
===========================

Parses natural language API requests into structured specifications
using an LLM (Claude by default).

This module provides LLM-powered intent parsing with multi-turn
clarification support. For simple pattern-based parsing without
an LLM dependency, see `engine.IntentParser`.

Usage:
    from semanticapi.intent_parser import IntentParser

    parser = IntentParser()
    result = parser.parse("Get my last 10 Stripe payments over $100")
    print(result.provider)  # "stripe"
    print(result.action)    # "list"
    print(result.filters)   # {"min_amount": 10000, "limit": 10}
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class Ambiguity(Enum):
    """Types of ambiguity that may require clarification."""

    PROVIDER = "provider"
    ACTION = "action"
    ENTITY = "entity"
    FILTER = "filter"
    PARAMETER = "parameter"


@dataclass
class ClarificationNeeded:
    """Represents a clarification request for ambiguous input."""

    ambiguity_type: Ambiguity
    question: str
    options: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a natural language API request."""

    provider: Optional[str] = None
    action: Optional[str] = None
    entity: Optional[str] = None
    filters: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_input: str = ""
    clarification_needed: Optional[ClarificationNeeded] = None
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "provider": self.provider,
            "action": self.action,
            "entity": self.entity,
            "filters": self.filters,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "raw_input": self.raw_input,
            "reasoning": self.reasoning,
        }
        if self.clarification_needed:
            result["clarification_needed"] = {
                "type": self.clarification_needed.ambiguity_type.value,
                "question": self.clarification_needed.question,
                "options": self.clarification_needed.options,
                "context": self.clarification_needed.context,
            }
        return result

    @property
    def is_complete(self) -> bool:
        """Check if parsing is complete (no clarification needed)."""
        return self.clarification_needed is None and self.confidence > 0.5


# Known providers and their capabilities (for prompt context)
PROVIDER_REGISTRY = {
    "stripe": {
        "entities": ["payment", "customer", "subscription", "invoice", "charge", "refund"],
        "actions": ["list", "get", "create", "update", "delete"],
        "aliases": ["stripe", "payment processor"],
    },
    "github": {
        "entities": ["repository", "issue", "pull_request", "commit", "branch", "user"],
        "actions": ["list", "get", "create", "update", "delete", "search", "fork", "star"],
        "aliases": ["github", "gh", "git"],
    },
    "slack": {
        "entities": ["message", "channel", "user", "file", "reaction"],
        "actions": ["send", "list", "get", "search", "create", "delete"],
        "aliases": ["slack"],
    },
    "notion": {
        "entities": ["page", "database", "block", "user"],
        "actions": ["list", "get", "create", "update", "delete", "search", "query"],
        "aliases": ["notion"],
    },
    "openai": {
        "entities": ["completion", "embedding", "image", "model"],
        "actions": ["create", "list", "get"],
        "aliases": ["openai", "gpt", "chatgpt"],
    },
    "twilio": {
        "entities": ["message", "call", "phone_number"],
        "actions": ["send", "list", "get", "create"],
        "aliases": ["twilio", "sms"],
    },
    "shopify": {
        "entities": ["product", "order", "customer", "inventory"],
        "actions": ["list", "get", "create", "update"],
        "aliases": ["shopify", "store"],
    },
    "gmail": {
        "entities": ["email", "message", "thread", "label"],
        "actions": ["send", "list", "get", "search"],
        "aliases": ["gmail", "email", "google mail"],
    },
}


class IntentParser:
    """
    LLM-powered intent parser with multi-turn clarification support.

    Uses Claude to intelligently extract:
    - Provider (which API/service)
    - Action (what operation)
    - Entity (what resource)
    - Filters (query constraints)
    - Parameters (additional data)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[dict[str, str]] = []
        self.pending_result: Optional[ParseResult] = None

    def _build_system_prompt(self) -> str:
        providers_info = json.dumps(PROVIDER_REGISTRY, indent=2)
        return f"""You are an API intent parser. Extract structured API call specifications from natural language.

## Known Providers
{providers_info}

## Output Format
JSON object:
{{
    "provider": "string or null",
    "action": "string or null",
    "entity": "string or null",
    "filters": {{}},
    "parameters": {{}},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "clarification_needed": null or {{
        "type": "provider|action|entity|filter|parameter",
        "question": "What to ask",
        "options": ["opt1", "opt2"]
    }}
}}

## Guidelines
- Convert amounts to cents for Stripe (e.g., $100 → 10000)
- Use snake_case for actions
- Confidence: 0.9+ very clear, 0.7-0.9 reasonable, 0.5-0.7 some ambiguity, <0.5 needs clarification"""

    def parse(
        self, request: str, context: Optional[dict[str, Any]] = None
    ) -> ParseResult:
        """
        Parse a natural language API request.

        Args:
            request: Natural language API request
            context: Optional context from prior conversation

        Returns:
            ParseResult with extracted intent
        """
        user_message = request
        if context:
            user_message = f"Context: {json.dumps(context)}\n\nRequest: {request}"

        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_prompt(),
            messages=self.conversation_history,
        )

        assistant_message = response.content[0].text
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        try:
            json_str = assistant_message
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            clarification = None
            if data.get("clarification_needed"):
                clar_data = data["clarification_needed"]
                clarification = ClarificationNeeded(
                    ambiguity_type=Ambiguity(clar_data["type"]),
                    question=clar_data["question"],
                    options=clar_data.get("options", []),
                    context=clar_data.get("context", {}),
                )

            result = ParseResult(
                provider=data.get("provider"),
                action=data.get("action"),
                entity=data.get("entity"),
                filters=data.get("filters", {}),
                parameters=data.get("parameters", {}),
                confidence=data.get("confidence", 0.0),
                raw_input=request,
                clarification_needed=clarification,
                reasoning=data.get("reasoning", ""),
            )

            self.pending_result = result
            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return ParseResult(
                raw_input=request,
                confidence=0.0,
                reasoning=f"Failed to parse response: {e}",
                clarification_needed=ClarificationNeeded(
                    ambiguity_type=Ambiguity.ACTION,
                    question="I couldn't understand that request. Could you rephrase?",
                ),
            )

    def clarify(self, response: str) -> ParseResult:
        """Continue a multi-turn conversation to resolve ambiguity."""
        if not self.pending_result or not self.pending_result.clarification_needed:
            return self.parse(response)

        clarification_context = {
            "previous_interpretation": self.pending_result.to_dict(),
            "clarification_type": self.pending_result.clarification_needed.ambiguity_type.value,
        }
        return self.parse(response, context=clarification_context)

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.pending_result = None


def parse_intent(request: str, model: str = "claude-sonnet-4-20250514") -> ParseResult:
    """
    Quick parse of a single API request.

    Args:
        request: Natural language API request
        model: Claude model to use

    Returns:
        ParseResult with extracted intent
    """
    parser = IntentParser(model=model)
    return parser.parse(request)
