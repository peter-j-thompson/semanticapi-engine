"""
Schema Transformer
===================

Normalizes API responses from different providers into a consistent
semantic format. Handles money amounts, timestamps, phone numbers,
and other common data types.

Usage:
    from semanticapi.schema_transformer import SchemaTransformer

    transformer = SchemaTransformer()
    result = transformer.transform(
        raw_response={"amount": 2000, "currency": "usd", "created": 1234567890},
        provider="stripe",
        entity_type="payment",
    )
    print(result["amount"]["display"])  # "$20.00"
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SchemaTransformer:
    """
    Transforms raw API responses into normalized semantic schemas.

    Features:
    - Provider-specific field mappings
    - Automatic type coercion (timestamps, money, phone numbers)
    - Graceful handling of missing fields
    - Raw response preservation
    """

    # Field mappings: {provider: {entity_type: {raw_field: semantic_field}}}
    MAPPINGS = {
        "stripe": {
            "payment": {
                "id": "id",
                "amount": "amount",
                "currency": "currency",
                "status": "status",
                "created": "created_at",
                "description": "description",
                "receipt_email": "email",
                "customer": "customer_id",
            },
            "customer": {
                "id": "id",
                "email": "email",
                "name": "name",
                "created": "created_at",
                "phone": "phone",
            },
        },
        "twilio": {
            "message": {
                "sid": "id",
                "from": "from_address",
                "to": "to_address",
                "body": "body",
                "status": "status",
                "direction": "direction",
                "date_created": "created_at",
            },
        },
        "openai": {
            "completion": {
                "id": "id",
                "model": "model",
                "usage": "usage",
            },
            "image": {},
        },
        "github": {
            "repository": {
                "id": "id",
                "full_name": "full_name",
                "name": "name",
                "description": "description",
                "html_url": "url",
                "stargazers_count": "stars",
                "language": "language",
                "created_at": "created_at",
                "updated_at": "updated_at",
            },
            "issue": {
                "id": "id",
                "number": "number",
                "title": "title",
                "body": "body",
                "state": "status",
                "html_url": "url",
                "created_at": "created_at",
                "updated_at": "updated_at",
            },
        },
    }

    # Fields that contain timestamps
    TIMESTAMP_FIELDS = {
        "created_at", "updated_at", "sent_at", "delivered_at",
        "read_at", "captured_at", "last_login_at",
    }

    # Fields that contain monetary amounts (in cents)
    MONEY_FIELDS = {"amount", "refunded_amount"}

    def __init__(self):
        pass

    def transform(
        self,
        raw_response: Dict[str, Any],
        provider: str,
        entity_type: str,
    ) -> Dict[str, Any]:
        """
        Transform a raw API response to normalized format.

        Args:
            raw_response: Raw response from the provider API
            provider: Provider name (e.g., "stripe", "twilio")
            entity_type: Entity type (e.g., "payment", "message")

        Returns:
            Normalized dictionary with consistent field names and formats
        """
        mapping = self.MAPPINGS.get(provider, {}).get(entity_type, {})

        if not mapping:
            # No specific mapping — return with metadata
            return {
                "_raw": raw_response,
                "_provider": provider,
                "_entity_type": entity_type,
                **raw_response,
            }

        result = {}
        currency = raw_response.get("currency", "USD")

        for raw_field, semantic_field in mapping.items():
            value = raw_response.get(raw_field)
            if value is None:
                continue

            # Apply type transformations
            if semantic_field in self.TIMESTAMP_FIELDS:
                value = self._normalize_timestamp(value)
            elif semantic_field in self.MONEY_FIELDS:
                value = self._normalize_money(value, currency)
            elif semantic_field == "phone":
                value = self._normalize_phone(value)

            result[semantic_field] = value

        # Extract OpenAI completion content
        if provider == "openai" and entity_type == "completion":
            choices = raw_response.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                result["content"] = content

        # Extract OpenAI image URL
        if provider == "openai" and entity_type == "image":
            data = raw_response.get("data", [])
            if data:
                result["url"] = data[0].get("url")
                result["revised_prompt"] = data[0].get("revised_prompt")

        result["_raw"] = raw_response
        result["_provider"] = provider

        return result

    def transform_list(
        self,
        raw_responses: List[Dict[str, Any]],
        provider: str,
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """Transform a list of responses."""
        return [
            self.transform(r, provider=provider, entity_type=entity_type)
            for r in raw_responses
        ]

    def _normalize_timestamp(self, value: Any) -> Optional[str]:
        """Convert various timestamp formats to ISO 8601."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            # Detect milliseconds
            if value > 32503680000:
                value = value / 1000
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
            return dt.isoformat()

        if isinstance(value, str):
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.isoformat()
                except ValueError:
                    continue
            return value

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()

        return str(value)

    def _normalize_money(self, amount: Any, currency: str) -> Dict[str, Any]:
        """Normalize money amount to standard format."""
        if amount is None:
            return None

        if isinstance(amount, str):
            amount = int(float(amount) * 100)

        cents = int(amount)
        dollars = cents / 100.0
        currency = currency.upper()

        symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        symbol = symbols.get(currency, currency + " ")

        if currency == "JPY":
            display = f"{symbol}{cents}"
        else:
            display = f"{symbol}{dollars:,.2f}"

        return {
            "cents": cents,
            "value": dollars,
            "currency": currency,
            "display": display,
        }

    def _normalize_phone(self, value: Any) -> Optional[Dict[str, Any]]:
        """Normalize phone number."""
        if value is None:
            return None

        raw = str(value)
        digits = re.sub(r"[^\d+]", "", raw)

        e164 = None
        country_code = None
        display = raw

        if digits.startswith("+"):
            e164 = digits
        elif len(digits) == 10:
            e164 = f"+1{digits}"
            country_code = "US"
        elif len(digits) == 11 and digits.startswith("1"):
            e164 = f"+{digits}"
            country_code = "US"

        if e164 and e164.startswith("+1") and len(e164) == 12:
            d = e164[2:]
            display = f"({d[:3]}) {d[3:6]}-{d[6:]}"

        return {
            "raw": raw,
            "e164": e164,
            "display": display,
            "country_code": country_code,
        }
