"""
API Registry
=============

Registry for API providers and their capabilities with keyword-based
search. Provides a lightweight way to discover and match API capabilities
to natural language queries.

Usage:
    from semanticapi.api_registry import APIRegistry

    registry = APIRegistry(providers_dir="./providers")
    registry.load_all_providers()

    matches = registry.find("send a text message")
    # [{"provider": "twilio", "capability": "send_sms", "confidence": 0.85, ...}]
"""

import json
from pathlib import Path
from typing import Optional


class APIRegistry:
    """
    Registry for API capabilities with keyword-based search.

    Stores provider definitions and their capabilities, providing
    semantic matching through keyword overlap scoring.
    """

    def __init__(self, providers_dir: Optional[str] = None):
        """
        Initialize the API Registry.

        Args:
            providers_dir: Path to directory containing provider JSON files.
                          Defaults to the bundled providers/ directory.
        """
        if providers_dir:
            self.providers_dir = Path(providers_dir)
        else:
            self.providers_dir = Path(__file__).parent.parent / "providers"

        self._providers: dict = {}
        self._capabilities: list = []

    def load_provider(self, provider_id: str) -> bool:
        """
        Load a provider from a JSON file.

        Args:
            provider_id: Provider name (e.g., "stripe") — looks for {provider_id}.json

        Returns:
            True if loaded successfully.
        """
        json_path = self.providers_dir / f"{provider_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Provider not found: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        return self.register_provider(data)

    def register_provider(self, provider_def: dict) -> bool:
        """
        Register a provider and its capabilities.

        Args:
            provider_def: Full provider definition dict.

        Returns:
            True if registered successfully.
        """
        provider_id = provider_def.get("provider", "")
        if not provider_id:
            return False

        self._providers[provider_id] = {
            "id": provider_id,
            "name": provider_def.get("name", provider_id),
            "description": provider_def.get("description", ""),
            "base_url": provider_def.get("base_url", ""),
            "auth": provider_def.get("auth", {}),
        }

        # Remove old capabilities for this provider
        self._capabilities = [
            c for c in self._capabilities if c["provider_id"] != provider_id
        ]

        for cap in provider_def.get("capabilities", []):
            self._capabilities.append(
                {
                    "id": f"{provider_id}.{cap['id']}",
                    "provider_id": provider_id,
                    "capability_id": cap["id"],
                    "name": cap.get("name", ""),
                    "description": cap.get("description", ""),
                    "semantic_tags": cap.get("semantic_tags", []),
                    "endpoint": cap.get("endpoint", {}),
                }
            )

        return True

    def load_all_providers(self) -> int:
        """
        Load all provider JSON files from the providers directory.

        Returns:
            Number of providers loaded.
        """
        count = 0
        if not self.providers_dir.exists():
            return count

        for json_file in sorted(self.providers_dir.glob("*.json")):
            try:
                self.load_provider(json_file.stem)
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load {json_file.stem}: {e}")
        return count

    def find(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.3,
        providers: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Find API capabilities matching a natural language query.

        Uses keyword overlap scoring between the query and capability
        descriptions, names, and semantic tags.

        Args:
            query: Natural language description
            limit: Max results to return
            min_confidence: Minimum score threshold (0-1)
            providers: Optional provider filter

        Returns:
            List of matching capabilities with confidence scores.
        """
        query_words = set(query.lower().split())

        results = []
        for cap in self._capabilities:
            if providers and cap["provider_id"] not in providers:
                continue

            # Build searchable text from name, description, and tags
            text_parts = [
                cap["name"].lower(),
                cap["description"].lower(),
                " ".join(cap["semantic_tags"]).lower(),
            ]
            text = " ".join(text_parts)
            text_words = set(text.split())

            # Score by keyword overlap
            overlap = len(query_words & text_words)
            if overlap > 0:
                score = min(overlap / max(len(query_words), 1), 0.99)

                if score >= min_confidence:
                    provider_info = self._providers.get(cap["provider_id"], {})
                    results.append(
                        {
                            "provider": cap["provider_id"],
                            "provider_name": provider_info.get("name", cap["provider_id"]),
                            "capability": cap["capability_id"],
                            "name": cap["name"],
                            "description": cap["description"],
                            "confidence": round(score, 4),
                            "endpoint": cap["endpoint"],
                        }
                    )

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:limit]

    def get_capability(self, capability_id: str) -> Optional[dict]:
        """
        Get a specific capability by full ID (e.g., "stripe.list_payments").
        """
        for cap in self._capabilities:
            if cap["id"] == capability_id:
                provider_info = self._providers.get(cap["provider_id"], {})
                return {
                    "provider": cap["provider_id"],
                    "provider_name": provider_info.get("name", ""),
                    "capability": cap["capability_id"],
                    "name": cap["name"],
                    "description": cap["description"],
                    "semantic_tags": cap["semantic_tags"],
                    "endpoint": cap["endpoint"],
                }
        return None

    def list_providers(self) -> list[dict]:
        """List all registered providers."""
        results = []
        for pid, info in self._providers.items():
            cap_count = sum(
                1 for c in self._capabilities if c["provider_id"] == pid
            )
            results.append(
                {
                    "id": pid,
                    "name": info["name"],
                    "description": info["description"],
                    "capabilities": cap_count,
                }
            )
        return results

    def list_capabilities(self, provider_id: Optional[str] = None) -> list[dict]:
        """List capabilities, optionally filtered by provider."""
        results = []
        for cap in self._capabilities:
            if provider_id and cap["provider_id"] != provider_id:
                continue
            results.append(
                {
                    "id": cap["id"],
                    "name": cap["name"],
                    "description": cap["description"],
                }
            )
        return results

    def stats(self) -> dict:
        """Get registry statistics."""
        return {
            "providers": len(self._providers),
            "capabilities": len(self._capabilities),
        }
