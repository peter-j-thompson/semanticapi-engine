"""
Tests for the pattern-based intent parser in engine.py.

These tests don't require an LLM — they test the regex-based parsing.
"""

import pytest
from semanticapi.engine import IntentParser, IntentType, ParsedIntent


@pytest.fixture
def parser():
    return IntentParser()


class TestIntentParser:
    """Test the pattern-based intent parser."""

    def test_stripe_payments(self, parser):
        result = parser.parse("get my last 5 stripe payments")
        assert result.provider == "stripe"
        assert result.action == "list_payments"
        assert result.filters.get("limit") == 5
        assert result.confidence > 0.5

    def test_stripe_payments_no_limit(self, parser):
        result = parser.parse("show stripe payments")
        assert result.provider == "stripe"
        assert result.action == "list_payments"

    def test_stripe_balance(self, parser):
        result = parser.parse("what's my stripe balance")
        assert result.provider == "stripe"
        assert result.action == "get_balance"

    def test_stripe_customers(self, parser):
        result = parser.parse("list all stripe customers")
        assert result.provider == "stripe"
        assert result.action == "list_customers"

    def test_send_sms(self, parser):
        result = parser.parse("send SMS to +15551234567 saying 'Hello!'")
        assert result.provider == "twilio"
        assert result.action == "send_sms"
        assert "+15551234567" in result.entities.get("to", "")

    def test_send_text_message(self, parser):
        result = parser.parse("send a text message to +15551234567")
        assert result.provider == "twilio"
        assert result.action == "send_sms"

    def test_create_payment(self, parser):
        result = parser.parse("create a payment for $50")
        assert result.provider == "stripe"
        assert result.action == "create_payment"
        assert result.entities.get("amount") == 50.0

    def test_refund_payment(self, parser):
        result = parser.parse("refund payment ch_abc123")
        assert result.provider == "stripe"
        assert result.action == "create_refund"
        assert result.entities.get("payment_id") == "ch_abc123"

    def test_generate_image(self, parser):
        result = parser.parse("generate an image of a sunset over mountains")
        assert result.provider == "openai"
        assert result.action == "create_image"
        assert "sunset" in result.entities.get("prompt", "")

    def test_unknown_query(self, parser):
        result = parser.parse("do something random")
        assert result.confidence < 0.5

    def test_basic_parse_with_provider(self, parser):
        result = parser.parse("list github repos")
        assert result.provider == "github"

    def test_parsed_intent_to_dict(self):
        intent = ParsedIntent(
            intent_type=IntentType.FETCH,
            provider="stripe",
            action="list_payments",
            entities={},
            filters={"limit": 10},
            output_hints={},
            confidence=0.9,
            raw_query="get stripe payments",
        )
        d = intent.to_dict()
        assert d["provider"] == "stripe"
        assert d["intent_type"] == "fetch"
        assert d["filters"]["limit"] == 10


class TestIntentType:
    def test_fetch(self):
        assert IntentType.FETCH.value == "fetch"

    def test_execute(self):
        assert IntentType.EXECUTE.value == "execute"

    def test_stream(self):
        assert IntentType.STREAM.value == "stream"
