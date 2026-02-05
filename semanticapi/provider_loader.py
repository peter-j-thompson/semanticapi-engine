"""
Provider Loader
================

Loads provider definitions from JSON files on disk.

Providers define:
- API base URLs and authentication
- Available capabilities (endpoints)
- Semantic tags for natural language matching

Usage:
    from semanticapi.provider_loader import load_providers

    providers = load_providers("./providers")
    for p in providers:
        print(f"{p.name}: {len(p.capabilities)} capabilities")
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class LoadedProvider:
    """A provider loaded from a JSON definition file."""

    provider_id: str
    name: str
    description: str
    base_url: str
    auth: Dict[str, Any]
    capabilities: List[Dict[str, Any]]
    source: str  # "core", "custom"
    config: Dict[str, Any]  # Full original config


# Default providers directory (bundled with the package)
PROVIDERS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "providers")


def load_json_providers(directory: str, source: str = "core") -> List[LoadedProvider]:
    """
    Load provider JSON files from a directory.

    Args:
        directory: Path to directory containing .json provider files.
        source: Label for where these providers came from ("core", "custom").

    Returns:
        List of LoadedProvider instances.
    """
    providers = []

    if not os.path.exists(directory):
        return providers

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r") as f:
                config = json.load(f)

            provider = LoadedProvider(
                provider_id=config.get("provider", filename.replace(".json", "")),
                name=config.get("name", ""),
                description=config.get("description", ""),
                base_url=config.get("base_url", ""),
                auth=config.get("auth", {}),
                capabilities=config.get("capabilities", []),
                source=source,
                config=config,
            )
            providers.append(provider)
        except Exception as e:
            print(f"Error loading provider {filepath}: {e}")

    return providers


def load_providers(directory: str = None) -> List[LoadedProvider]:
    """
    Load all providers from a directory.

    This is the main entry point for loading providers. If no directory
    is specified, loads from the bundled providers/ directory.

    Args:
        directory: Path to providers directory. Defaults to bundled providers.

    Returns:
        List of LoadedProvider instances.

    Example:
        providers = load_providers()
        for p in providers:
            print(f"{p.name}: {len(p.capabilities)} capabilities")
    """
    if directory is None:
        directory = PROVIDERS_DIR
    return load_json_providers(directory, source="core")


@lru_cache(maxsize=1)
def get_static_providers() -> List[LoadedProvider]:
    """
    Load and cache the bundled providers.

    Results are cached since the bundled providers don't change at runtime.
    Call clear_provider_cache() if you add new provider files dynamically.
    """
    return load_providers(PROVIDERS_DIR)


def clear_provider_cache():
    """Clear the cached static providers (after adding new ones)."""
    get_static_providers.cache_clear()


def get_provider_context_for_llm(providers: List[LoadedProvider]) -> str:
    """
    Build a context string describing providers for use in LLM prompts.

    Args:
        providers: List of loaded providers.

    Returns:
        Formatted string describing all providers and their capabilities.
    """
    lines = ["Available API providers and their capabilities:\n"]

    for provider in providers:
        lines.append(f"\n## {provider.name} ({provider.provider_id})")
        lines.append(f"Description: {provider.description}")

        if provider.capabilities:
            lines.append("Capabilities:")
            for cap in provider.capabilities:
                cap_id = cap.get("id", "")
                cap_name = cap.get("name", cap_id)
                cap_desc = cap.get("description", "")
                tags = cap.get("semantic_tags", [])

                lines.append(f"  - {cap_id}: {cap_name}")
                if cap_desc:
                    lines.append(f"    {cap_desc}")
                if tags:
                    lines.append(f"    Keywords: {', '.join(tags[:5])}")

    return "\n".join(lines)


def find_provider_by_id(
    providers: List[LoadedProvider], provider_id: str
) -> Optional[LoadedProvider]:
    """Find a provider by its ID."""
    for p in providers:
        if p.provider_id == provider_id:
            return p
    return None


def find_capability(
    provider: LoadedProvider, capability_id: str
) -> Optional[Dict[str, Any]]:
    """Find a capability within a provider."""
    for cap in provider.capabilities:
        if cap.get("id") == capability_id:
            return cap
    return None
