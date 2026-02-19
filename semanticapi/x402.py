"""
x402 Payment Support
=====================

HTTP 402-based micropayment protocol using USDC on Base (L2 Ethereum).

This module provides x402 payment header generation and endpoint pricing
for self-hosted Semantic API instances that want to monetize their API.

See: https://www.x402.org

Usage:
    from semanticapi.x402 import get_price_for_endpoint, get_payment_required_header

    price = get_price_for_endpoint("/api/query")
    if price:
        headers = get_payment_required_header(price)
"""

import os
from typing import Dict, Optional

# Receiving wallet address (Base network)
WALLET_ADDRESS: str = os.getenv("X402_WALLET_ADDRESS", "")

# Network identifier (Base mainnet or Sepolia testnet)
NETWORK: str = os.getenv("X402_NETWORK", "eip155:84532")  # Default to testnet

# USDC contract addresses per network
USDC_ADDRESSES: Dict[str, str] = {
    "eip155:8453": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",   # Base mainnet
    "eip155:84532": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # Base Sepolia
}

# Whether x402 payment is required
REQUIRE_PAYMENT: bool = os.getenv("X402_REQUIRE_PAYMENT", "false").lower() == "true"

# Payment deadline in seconds
REQUIRED_DEADLINE_SECONDS: int = int(os.getenv("X402_DEADLINE_SECONDS", "86400"))

# x402 protocol version
X402_VERSION: int = 1

# Pricing per endpoint (in USDC)
ENDPOINT_PRICING: Dict[str, str] = {
    "/api/query": "0.01",
    "/api/query/batch": "0.05",
    "/api/query/agentic": "0.01",
    "/api/query/preflight": "0.01",
    "/api/discover/search": "0.05",
    "/api/discover/from-url": "0.10",
}

# Free endpoints (no payment required)
FREE_ENDPOINTS: list = [
    "/api/providers",
    "/api/capabilities",
    "/docs",
    "/openapi.json",
    "/health",
]


def get_price_for_endpoint(path: str) -> Optional[str]:
    """Get the USDC price for a given endpoint path, or None if free."""
    for prefix in FREE_ENDPOINTS:
        if path.startswith(prefix):
            return None

    if path in ENDPOINT_PRICING:
        return ENDPOINT_PRICING[path]

    for endpoint_prefix, price in ENDPOINT_PRICING.items():
        if path.startswith(endpoint_prefix):
            return price

    return None


def get_payment_required_header(price: str) -> dict:
    """Build the X-Payment-Required header value for HTTP 402 responses."""
    usdc_address = USDC_ADDRESSES.get(NETWORK, USDC_ADDRESSES["eip155:84532"])

    return {
        "x402Version": X402_VERSION,
        "accepts": [
            {
                "scheme": "exact",
                "network": NETWORK,
                "maxAmountRequired": price,
                "payToAddress": WALLET_ADDRESS,
                "requiredDeadlineSeconds": REQUIRED_DEADLINE_SECONDS,
                "usdcAddress": usdc_address,
            }
        ],
    }
