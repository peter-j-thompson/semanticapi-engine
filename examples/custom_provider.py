"""
Custom Provider Example
========================

Shows how to add your own API provider to the Semantic API Engine.

Providers are defined as JSON files — no code needed!
"""

import json
import os
from pathlib import Path

from semanticapi import AgenticProcessor


def create_custom_provider():
    """
    Create a custom provider definition for any API.

    This example creates a provider for a weather API.
    """

    weather_provider = {
        "provider": "weatherapi",
        "name": "Weather API",
        "description": "Real-time weather data and forecasts",
        "base_url": "https://api.weatherapi.com/v1",
        "auth": {
            "type": "bearer",
            "header": "Authorization",
            "prefix": "Bearer",
        },
        "capabilities": [
            {
                "id": "current_weather",
                "name": "Get Current Weather",
                "description": "Get current weather conditions for a location",
                "semantic_tags": [
                    "weather",
                    "temperature",
                    "current",
                    "conditions",
                    "forecast",
                ],
                "endpoint": {
                    "method": "GET",
                    "path": "/current.json",
                    "params": {
                        "q": {
                            "type": "string",
                            "required": True,
                            "description": "Location (city name, zip code, coordinates)",
                        }
                    },
                },
            },
            {
                "id": "forecast",
                "name": "Get Weather Forecast",
                "description": "Get weather forecast for up to 14 days",
                "semantic_tags": [
                    "forecast",
                    "prediction",
                    "future weather",
                    "tomorrow",
                ],
                "endpoint": {
                    "method": "GET",
                    "path": "/forecast.json",
                    "params": {
                        "q": {
                            "type": "string",
                            "required": True,
                            "description": "Location",
                        },
                        "days": {
                            "type": "integer",
                            "required": False,
                            "description": "Number of forecast days (1-14)",
                        },
                    },
                },
            },
        ],
    }

    # Save to a custom providers directory
    custom_dir = Path("./custom_providers")
    custom_dir.mkdir(exist_ok=True)

    with open(custom_dir / "weatherapi.json", "w") as f:
        json.dump(weather_provider, f, indent=2)

    print(f"Created custom provider: {custom_dir / 'weatherapi.json'}")
    return custom_dir


def use_custom_provider(providers_dir: str):
    """Use the custom provider with the agentic processor."""

    processor = AgenticProcessor(
        providers_dir=providers_dir,
        ai_provider="anthropic",
        debug=True,
    )

    # Add credentials for the custom provider
    processor.add_provider("weatherapi", api_key="your_weather_api_key")

    # Query it with natural language
    result = processor.process("What's the weather in San Francisco?")
    print(f"\nResponse: {result.response}")


if __name__ == "__main__":
    print("=" * 60)
    print("Custom Provider Example")
    print("=" * 60)
    print()

    # Step 1: Create the custom provider JSON
    custom_dir = create_custom_provider()

    # Step 2: Use it (uncomment with real API key)
    # use_custom_provider(str(custom_dir))

    print()
    print("To use this provider:")
    print(f"  1. Add your API key")
    print(f"  2. Point the engine to your providers directory:")
    print(f"     PROVIDERS_DIR={custom_dir} python -m semanticapi.server")
    print()
    print("Or load both bundled + custom providers by placing your JSON")
    print("file in the providers/ directory alongside the built-in ones.")
