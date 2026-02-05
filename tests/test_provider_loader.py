"""
Tests for the provider loader.
"""

import json
import os
import tempfile
import pytest

from semanticapi.provider_loader import (
    LoadedProvider,
    load_json_providers,
    load_providers,
    find_provider_by_id,
    find_capability,
    get_provider_context_for_llm,
)


@pytest.fixture
def sample_provider_dir():
    """Create a temporary directory with sample provider JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample provider
        provider = {
            "provider": "testapi",
            "name": "Test API",
            "description": "A test API for unit tests",
            "base_url": "https://api.test.com/v1",
            "auth": {"type": "bearer"},
            "capabilities": [
                {
                    "id": "list_items",
                    "name": "List Items",
                    "description": "List all items",
                    "semantic_tags": ["items", "list"],
                    "endpoint": {
                        "method": "GET",
                        "path": "/items",
                        "params": {
                            "limit": {
                                "type": "integer",
                                "required": False,
                                "description": "Max results",
                            }
                        },
                    },
                },
                {
                    "id": "create_item",
                    "name": "Create Item",
                    "description": "Create a new item",
                    "semantic_tags": ["create", "item", "new"],
                    "endpoint": {
                        "method": "POST",
                        "path": "/items",
                        "params": {
                            "name": {
                                "type": "string",
                                "required": True,
                                "description": "Item name",
                            }
                        },
                    },
                },
            ],
        }

        with open(os.path.join(tmpdir, "testapi.json"), "w") as f:
            json.dump(provider, f)

        # Create a second provider
        provider2 = {
            "provider": "anotherapi",
            "name": "Another API",
            "description": "Another test API",
            "base_url": "https://api.another.com",
            "auth": {"type": "basic"},
            "capabilities": [],
        }

        with open(os.path.join(tmpdir, "anotherapi.json"), "w") as f:
            json.dump(provider2, f)

        yield tmpdir


class TestLoadJsonProviders:
    def test_loads_providers(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        assert len(providers) == 2

    def test_provider_fields(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        testapi = next(p for p in providers if p.provider_id == "testapi")

        assert testapi.name == "Test API"
        assert testapi.description == "A test API for unit tests"
        assert testapi.base_url == "https://api.test.com/v1"
        assert testapi.auth == {"type": "bearer"}
        assert len(testapi.capabilities) == 2

    def test_empty_directory(self, tmp_path):
        providers = load_json_providers(str(tmp_path))
        assert len(providers) == 0

    def test_nonexistent_directory(self):
        providers = load_json_providers("/nonexistent/path")
        assert len(providers) == 0

    def test_source_label(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir, source="custom")
        for p in providers:
            assert p.source == "custom"

    def test_ignores_non_json(self, sample_provider_dir):
        # Write a non-JSON file
        with open(os.path.join(sample_provider_dir, "readme.txt"), "w") as f:
            f.write("not a provider")

        providers = load_json_providers(sample_provider_dir)
        assert len(providers) == 2  # Still only the 2 JSON files


class TestLoadProviders:
    def test_loads_bundled_providers(self):
        """Verify the bundled providers load correctly."""
        providers = load_providers()
        assert len(providers) >= 8  # We ship 8 providers

        provider_ids = [p.provider_id for p in providers]
        assert "stripe" in provider_ids
        assert "twilio" in provider_ids
        assert "github" in provider_ids
        assert "openai" in provider_ids
        assert "slack" in provider_ids
        assert "notion" in provider_ids
        assert "gmail" in provider_ids
        assert "shopify" in provider_ids

    def test_providers_have_capabilities(self):
        providers = load_providers()
        for p in providers:
            if p.provider_id != "anotherapi":
                assert len(p.capabilities) > 0, f"{p.name} has no capabilities"


class TestFindProviderById:
    def test_find_existing(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        result = find_provider_by_id(providers, "testapi")
        assert result is not None
        assert result.name == "Test API"

    def test_find_nonexistent(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        result = find_provider_by_id(providers, "nonexistent")
        assert result is None


class TestFindCapability:
    def test_find_capability(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        testapi = find_provider_by_id(providers, "testapi")
        cap = find_capability(testapi, "list_items")
        assert cap is not None
        assert cap["name"] == "List Items"

    def test_find_nonexistent_capability(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        testapi = find_provider_by_id(providers, "testapi")
        cap = find_capability(testapi, "nonexistent")
        assert cap is None


class TestProviderContextForLLM:
    def test_generates_context(self, sample_provider_dir):
        providers = load_json_providers(sample_provider_dir)
        context = get_provider_context_for_llm(providers)

        assert "Test API" in context
        assert "testapi" in context
        assert "list_items" in context
        assert "create_item" in context
