"""
Basic Usage Example
====================

Shows how to use the Semantic API Engine to interact with APIs
using natural language.
"""

from semanticapi import SemanticAPI, AgenticProcessor

# ─────────────────────────────────────────────────────────────
# Example 1: Simple SDK Usage (pattern-based, no LLM needed)
# ─────────────────────────────────────────────────────────────

def sdk_example():
    """Use the SemanticAPI SDK for simple queries."""
    api = SemanticAPI()

    # Configure providers
    api.add_provider("stripe", api_key="sk_live_your_key_here")
    api.add_provider("twilio",
        account_sid="ACxxxxxxxxxxxxxxxx",
        auth_token="your_auth_token",
        from_number="+15551234567",
    )

    # Fetch data with natural language
    payments = api.fetch("get my last 5 stripe payments")
    print(f"Success: {payments.success}")
    print(f"Data: {payments.data}")

    # Execute actions
    result = api.execute("send SMS to +15559876543 saying 'Hello from Semantic API!'")
    print(f"SMS sent: {result.success}")

    # List available capabilities
    caps = api.capabilities()
    for cap in caps:
        print(f"  {cap['provider']}.{cap['action']}: {cap['description']}")


# ─────────────────────────────────────────────────────────────
# Example 2: Agentic Processor (LLM-powered, multi-step)
# ─────────────────────────────────────────────────────────────

def agentic_example():
    """
    Use the AgenticProcessor for complex, multi-step queries.

    Requires an LLM API key (e.g., ANTHROPIC_API_KEY env var).
    """
    processor = AgenticProcessor(
        ai_provider="anthropic",  # or "openai", "groq", "ollama"
        debug=True,
    )

    # Add provider credentials
    processor.add_provider("stripe", api_key="sk_live_your_key_here")
    processor.add_provider("github", access_token="ghp_your_token_here")

    # Process a complex query — the AI figures out what to do
    result = processor.process(
        "What's my Stripe balance? Also list my last 3 payments."
    )

    print(f"Status: {result.status}")
    print(f"Response: {result.response}")

    # The AI might ask for clarification
    if result.status == "needs_input":
        print(f"Question: {result.question}")
        if result.options:
            print(f"Options: {result.options}")

    # Check execution steps
    for step in result.steps:
        print(f"  Step: {step['tool']} → {'✓' if 'error' not in step.get('result', {}) else '✗'}")


# ─────────────────────────────────────────────────────────────
# Example 3: Quick one-liner
# ─────────────────────────────────────────────────────────────

def quick_example():
    """Process a query in one line."""
    from semanticapi import process_query

    result = process_query(
        "get my stripe balance",
        stripe={"api_key": "sk_live_your_key_here"},
    )
    print(result.response)


if __name__ == "__main__":
    print("=" * 60)
    print("Semantic API Engine — Basic Usage Examples")
    print("=" * 60)
    print()
    print("Note: Replace API keys with real ones to run these examples.")
    print()

    # Uncomment to run:
    # sdk_example()
    # agentic_example()
    # quick_example()

    # Show available capabilities without API keys
    api = SemanticAPI()
    print("Available providers:", api.providers())
    print("\nCapabilities:")
    for cap in api.capabilities():
        print(f"  {cap['provider']}.{cap['action']}: {cap['description']}")
