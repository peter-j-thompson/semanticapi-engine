"""
Agentic Query Processor
========================

AI-powered query processor that uses tool calling to interact with APIs.

The agent has access to API capabilities as tools and can:
1. Understand user intent from natural language
2. Call APIs as needed (multi-step reasoning)
3. Examine results and decide if more calls are needed
4. Synthesize a human-friendly response

Supports multiple LLM providers:
- Anthropic (Claude) — default
- OpenAI (GPT-4)
- Groq (Llama, Mixtral)
- Ollama (local models)
"""

import json
import base64
import re
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import httpx
import os


def apply_request_transform(
    params: dict,
    capability: dict,
    credentials: dict,
) -> dict:
    """
    Apply request transforms based on capability config.
    
    This replaces hardcoded provider-specific logic with config-driven transforms.
    The capability's request_transform field defines what transforms to apply.
    
    Supported transform types:
    - mime_encode: Encode fields into MIME format (for email APIs like Gmail)
    - default_from_creds: Set default param values from credentials
    
    Returns the transformed params dict.
    """
    transform = capability.get("request_transform")
    if not transform:
        return params
    
    # Handle legacy string format (deprecated)
    if isinstance(transform, str):
        return params
    
    transform_type = transform.get("type")
    
    if transform_type == "mime_encode":
        # MIME encoding for email APIs (e.g., Gmail)
        input_fields = transform.get("input_fields", {})
        output_field = transform.get("output_field", "raw")
        
        def get_field(aliases):
            for alias in aliases:
                if alias in params:
                    return params.pop(alias)
            return ""
        
        to = get_field(input_fields.get("to", ["to"]))
        subject = get_field(input_fields.get("subject", ["subject"]))
        body = get_field(input_fields.get("body", ["body"]))
        
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        params = {output_field: raw}
        
    elif transform_type == "default_from_creds":
        # Default param values from credentials (e.g., Twilio from_number)
        defaults = transform.get("defaults", {})
        for param_name, cred_key in defaults.items():
            if param_name not in params and credentials.get(cred_key):
                params[param_name] = credentials[cred_key]
    
    return params


@dataclass
class AgentResult:
    """Result of agentic query processing."""

    success: bool
    response: str
    steps: list = field(default_factory=list)
    error: Optional[str] = None
    raw_data: Optional[Any] = None
    status: str = "complete"  # "complete" | "needs_input" | "error"
    question: Optional[str] = None
    options: Optional[list] = None
    data: Optional[dict] = None


class AgenticProcessor:
    """
    Agentic AI processor that uses tool calling to interact with APIs.

    The AI has full autonomy to:
    - Decide which APIs to call
    - Make multiple calls if needed
    - Reason about results
    - Ask the user for clarification
    - Request provider connections when needed
    - Synthesize human-friendly responses

    Usage:
        processor = AgenticProcessor()
        processor.add_provider("stripe", api_key="sk_live_...")
        result = processor.process("get my stripe balance")
        print(result.response)
    """

    def __init__(
        self,
        providers_dir: str = None,
        model: str = None,
        max_steps: int = 10,
        debug: bool = False,
        ai_provider: str = "anthropic",
        api_key: str = None,
        base_url: str = None,
    ):
        """
        Initialize the agentic processor.

        Args:
            providers_dir: Path to provider JSON files directory.
                          Defaults to the bundled providers/ directory.
            model: LLM model to use. Defaults based on ai_provider.
            max_steps: Maximum tool calling iterations (safety limit).
            debug: Enable debug logging.
            ai_provider: LLM provider — "anthropic", "openai", "groq", or "ollama".
            api_key: API key for the LLM provider (or use env vars).
            base_url: Custom endpoint URL (for Ollama, proxies, etc.).
        """
        self.ai_provider = ai_provider
        self.max_steps = max_steps
        self.debug = debug

        self.model = model or self._get_default_model(ai_provider)

        # LLM clients
        self.anthropic_client = None
        self.openai_client = None
        self._init_llm_client(ai_provider, api_key, base_url)

        # Load provider capabilities from JSON files
        if providers_dir is None:
            providers_dir = Path(__file__).parent.parent / "providers"
        self.providers_dir = Path(providers_dir)
        self.capabilities = self._load_capabilities()

        # Configured provider credentials (set via add_provider)
        self.credentials = {}

        # User context per provider (username, repos, etc.)
        self.user_context = {}

    def _get_default_model(self, provider: str) -> str:
        """Get the default model for a provider."""
        defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
            "groq": "llama-3.3-70b-versatile",
            "ollama": "llama3.2",
        }
        return defaults.get(provider, "gpt-4o-mini")

    def _init_llm_client(
        self, provider: str, api_key: str = None, base_url: str = None
    ):
        """Initialize the LLM client for the specified provider."""
        if provider == "anthropic":
            import anthropic

            if api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            else:
                self.anthropic_client = anthropic.Anthropic()

        elif provider == "openai":
            import openai

            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self.openai_client = (
                openai.OpenAI(**kwargs) if kwargs else openai.OpenAI()
            )

        elif provider == "groq":
            import openai

            self.openai_client = openai.OpenAI(
                api_key=api_key or os.environ.get("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )

        elif provider == "ollama":
            import openai

            ollama_url = (
                base_url
                or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            )
            self.openai_client = openai.OpenAI(
                api_key="ollama",
                base_url=ollama_url,
            )

        if self.debug:
            print(
                f"[AgenticProcessor] Initialized {provider} client, model: {self.model}"
            )

    def _load_capabilities(self) -> dict:
        """Load all provider capability definitions from JSON files."""
        capabilities = {}

        if not self.providers_dir.exists():
            return capabilities

        for json_file in self.providers_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    provider_def = json.load(f)
                    provider_name = provider_def.get("provider")
                    if provider_name:
                        capabilities[provider_name] = provider_def
                        if self.debug:
                            cap_count = len(
                                provider_def.get("capabilities", [])
                            )
                            print(
                                f"[AgenticProcessor] Loaded {provider_name}: {cap_count} capabilities"
                            )
            except Exception as e:
                print(f"[AgenticProcessor] Failed to load {json_file}: {e}")

        return capabilities

    def add_provider(self, provider: str, **credentials):
        """Add credentials for a provider."""
        self.credentials[provider] = credentials
        if self.debug:
            print(f"[AgenticProcessor] Added credentials for {provider}")

    def set_user_context(self, provider: str, context: dict):
        """Set user context for a provider (username, repos, etc.)."""
        self.user_context[provider] = context

    @staticmethod
    def _sanitize_param_name(name: str) -> str:
        """Sanitize parameter names to match ^[a-zA-Z0-9_.-]{1,64}$."""
        import re
        sanitized = re.sub(r'[\[\]${}]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)  # collapse multiple underscores
        sanitized = sanitized.strip('_')
        return sanitized[:64]

    def _build_tools(self) -> list:
        """Convert capabilities to OpenAI tool format."""
        tools = []

        # ask_user — for clarifications
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": (
                        "Ask the user for clarification when you need more information. "
                        "Use when: (1) multiple valid options exist, (2) required info is missing, "
                        "(3) you're uncertain how to proceed, (4) an API call failed. "
                        "ALWAYS prefer asking over guessing."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Clear question for the user",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional list of choices",
                            },
                            "context": {
                                "type": "string",
                                "description": "What you've tried so far",
                            },
                        },
                        "required": ["question"],
                    },
                },
            }
        )

        # request_provider_connection — for auth
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "request_provider_connection",
                    "description": (
                        "Request the user to connect a provider that is not yet configured. "
                        "Use when you need a provider's capabilities but it's not connected."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name (e.g., 'stripe', 'gmail')",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this provider is needed",
                            },
                        },
                        "required": ["provider", "reason"],
                    },
                },
            }
        )

        # Provider capability tools
        for provider_name, provider_def in self.capabilities.items():
            is_connected = provider_name in self.credentials
            connection_status = (
                ""
                if is_connected
                else " [NOT CONNECTED - call request_provider_connection first]"
            )

            for cap in provider_def.get("capabilities", []):
                params_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

                endpoint = cap.get("endpoint", {})
                for param_name, param_def in endpoint.get("params", {}).items():
                    param_type = param_def.get("type", "string")
                    json_type = {
                        "string": "string",
                        "integer": "integer",
                        "number": "number",
                        "boolean": "boolean",
                    }.get(param_type, "string")

                    safe_name = self._sanitize_param_name(param_name)
                    params_schema["properties"][safe_name] = {
                        "type": json_type,
                        "description": param_def.get("description", ""),
                    }

                    if param_def.get("required"):
                        params_schema["required"].append(safe_name)

                tool = {
                    "type": "function",
                    "function": {
                        "name": f"{provider_name}__{cap['id']}",
                        "description": (
                            f"[{provider_def.get('name', provider_name)}]"
                            f"{connection_status} {cap.get('description', '')}"
                        ),
                        "parameters": params_schema,
                    },
                }
                tools.append(tool)

        return tools

    def _build_claude_tools(self) -> list:
        """Convert capabilities to Claude/Anthropic tool format."""
        tools = []

        # ask_user
        tools.append(
            {
                "name": "ask_user",
                "description": (
                    "Ask the user for clarification when you need more information. "
                    "Use when: (1) multiple valid options exist, (2) required info is missing, "
                    "(3) you're uncertain how to proceed, (4) an API call failed. "
                    "ALWAYS prefer asking over guessing."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Clear question for the user",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of choices",
                        },
                        "context": {
                            "type": "string",
                            "description": "What you've tried so far",
                        },
                    },
                    "required": ["question"],
                },
            }
        )

        # request_provider_connection
        tools.append(
            {
                "name": "request_provider_connection",
                "description": (
                    "Request the user to connect a provider that is not yet configured. "
                    "Use when you need a provider's capabilities but it's not connected."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "description": "Provider name (e.g., 'stripe', 'gmail')",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this provider is needed",
                        },
                    },
                    "required": ["provider", "reason"],
                },
            }
        )

        # Provider capability tools
        for provider_name, provider_def in self.capabilities.items():
            is_connected = provider_name in self.credentials
            connection_status = (
                ""
                if is_connected
                else " [NOT CONNECTED - call request_provider_connection first]"
            )

            for cap in provider_def.get("capabilities", []):
                params_schema = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }

                endpoint = cap.get("endpoint", {})
                for param_name, param_def in endpoint.get("params", {}).items():
                    param_type = param_def.get("type", "string")
                    json_type = {
                        "string": "string",
                        "integer": "integer",
                        "number": "number",
                        "boolean": "boolean",
                    }.get(param_type, "string")

                    safe_name = self._sanitize_param_name(param_name)
                    params_schema["properties"][safe_name] = {
                        "type": json_type,
                        "description": param_def.get("description", ""),
                    }

                    if param_def.get("required"):
                        params_schema["required"].append(safe_name)

                tools.append(
                    {
                        "name": f"{provider_name}__{cap['id']}",
                        "description": (
                            f"[{provider_def.get('name', provider_name)}]"
                            f"{connection_status} {cap.get('description', '')}"
                        ),
                        "input_schema": params_schema,
                    }
                )

        return tools

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        connected_providers = []
        unconnected_providers = []

        for provider_name, provider_def in self.capabilities.items():
            caps = [
                f"  - {c['id']}: {c['description']}"
                for c in provider_def.get("capabilities", [])
            ]
            provider_block = (
                f"{provider_def.get('name', provider_name)}:\n" + "\n".join(caps)
            )

            if provider_name in self.credentials:
                connected_providers.append(provider_block)
            else:
                unconnected_providers.append(
                    f"{provider_def.get('name', provider_name)} (NOT CONNECTED)"
                )

        # Build user context section
        user_context_section = ""
        user_context_info = []
        for provider_name, context in self.user_context.items():
            if context:
                ctx_lines = [f"{provider_name.upper()} USER CONTEXT:"]
                for key, value in context.items():
                    ctx_lines.append(f"  - {key}: {value}")
                user_context_info.append("\n".join(ctx_lines))

        if user_context_info:
            user_context_section = f"""

USER IDENTITY & CONTEXT:
{chr(10).join(user_context_info)}

Use this context to make smart defaults (e.g., infer the user's own repos/resources).
"""

        # Connected section
        if connected_providers:
            connected_section = f"""
CONNECTED APIs (ready to use):

{chr(10).join(connected_providers)}"""
        else:
            connected_section = """
NO PROVIDERS CONNECTED YET.

The user hasn't connected any API providers. Help them by:
1. Understanding what they want to do
2. Using request_provider_connection to ask them to connect the needed provider"""

        # Unconnected section
        unconnected_section = ""
        if unconnected_providers:
            unconnected_section = f"""

AVAILABLE BUT NOT CONNECTED (user needs to add credentials):
{chr(10).join(unconnected_providers)}

To use these providers, call request_provider_connection."""

        return f"""You are an AI assistant that helps users interact with various APIs using natural language.{user_context_section}
{connected_section}
{unconnected_section}

INSTRUCTIONS:

1. UNDERSTAND THE USER'S INTENT: What do they actually want to accomplish?

2. CALL THE RIGHT TOOLS: Use the available tools to fulfill the request. You may need multiple calls.

3. HANDLE API QUIRKS:
   - Gmail list_messages returns only IDs — call get_message for each to get content.
   - Gmail send_email: provide 'to', 'subject', and 'body' parameters directly.
   - Twilio SMS: provide 'To' (recipient) and 'Body' (message text).

4. REASON ABOUT RESULTS: After each API call, examine the result.
   - Did you get what the user needs?
   - Do you need more calls?
   - Is there an error to handle?

5. BATCH AUTH REQUESTS:
   - Before executing API calls, identify ALL providers needed.
   - If multiple providers are not connected, request them ALL at once.

6. ASK FOR CLARIFICATION (use ask_user liberally):
   - If you don't know which target to use (channel, repo, user), ASK.
   - If an API call fails and there are alternatives, ASK.
   - NEVER retry the same failing call — ask the user instead.
   - NEVER guess when you could ask.

7. AVOID RUNAWAY CALLS:
   - If a tool requires parameters you don't have, ASK.
   - Maximum 3 similar calls before asking for help.

8. SYNTHESIZE A RESPONSE: Provide a clear, human-friendly summary.
   Don't dump raw JSON — interpret the results.

Be helpful, concise, and proactive. ALWAYS prefer asking over guessing."""

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute an API tool call."""

        # Handle request_provider_connection
        if tool_name == "request_provider_connection":
            provider = arguments.get("provider", "unknown")
            reason = arguments.get("reason", "This provider is needed.")

            # Normalize provider name
            if provider not in self.capabilities and provider not in self.credentials:
                provider_lower = provider.lower().strip()
                for pid, pdef in self.capabilities.items():
                    if (
                        pid == provider_lower
                        or pdef.get("name", "").lower().strip() == provider_lower
                    ):
                        provider = pid
                        break

            # Already connected?
            if provider in self.credentials:
                provider_def = self.capabilities.get(provider, {})
                provider_display = provider_def.get("name", provider)
                return {
                    "status": "already_connected",
                    "provider": provider,
                    "provider_name": provider_display,
                    "message": f"{provider_display} is already connected!",
                    "action": "proceed",
                }

            # Need connection
            provider_def = self.capabilities.get(provider)
            if provider_def:
                provider_display = provider_def.get("name", provider)
                capabilities = [
                    c["name"]
                    for c in provider_def.get("capabilities", [])[:5]
                ]
                cap_list = ", ".join(capabilities)
                auth_config = provider_def.get("auth", {})
                auth_type = auth_config.get("type", "api_key")
            else:
                provider_display = provider
                cap_list = "various capabilities"
                auth_type = "api_key"

            return {
                "status": "connection_required",
                "provider": provider,
                "provider_name": provider_display,
                "auth_type": auth_type,
                "reason": reason,
                "capabilities": cap_list,
                "message": f"Please connect {provider_display} to continue. {reason}",
                "action": "connect_provider",
            }

        # Parse tool name: provider__capability
        parts = tool_name.split("__")
        if len(parts) != 2:
            return {"error": f"Invalid tool name format: {tool_name}"}

        provider_name, capability_id = parts

        # Get provider definition
        provider_def = self.capabilities.get(provider_name)
        if not provider_def:
            return {"error": f"Unknown provider: {provider_name}"}

        # Find the capability
        capability = None
        for cap in provider_def.get("capabilities", []):
            if cap["id"] == capability_id:
                capability = cap
                break

        if not capability:
            return {"error": f"Unknown capability: {capability_id}"}

        # Check credentials
        creds = self.credentials.get(provider_name, {})
        if not creds:
            provider_display = provider_def.get("name", provider_name)
            return {
                "error": f"{provider_display} is not connected",
                "status": "not_connected",
                "provider": provider_name,
                "hint": f"Use request_provider_connection to ask the user to connect {provider_display}.",
                "action": "request_connection",
            }

        # Build the API request
        endpoint = capability.get("endpoint", {})
        method = endpoint.get("method", "GET")
        path = endpoint.get("path", "")
        base_url = provider_def.get("base_url", "")

        url = base_url + path

        # Handle path parameters
        path_params = []
        for key, value in arguments.items():
            placeholder = "{" + key + "}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))
                path_params.append(key)

        for key in path_params:
            del arguments[key]

        # Handle special path params from credentials
        if "{account_sid}" in url:
            url = url.replace(
                "{account_sid}", creds.get("account_sid", "")
            )

        # Check for unresolved path parameters
        missing_params = re.findall(r"\{(\w+)\}", url)
        if missing_params:
            return {
                "error": f"Missing required parameters: {', '.join(missing_params)}. Ask the user!",
                "missing_params": missing_params,
            }

        # Build headers based on auth type
        auth_config = provider_def.get("auth", {})
        headers = {}
        auth = None

        auth_type = auth_config.get("type", "bearer")

        if auth_type == "oauth2":
            access_token = creds.get("access_token", "")
            if not access_token:
                return {
                    "error": f"No access token for {provider_name}. Please connect via OAuth."
                }
            headers["Authorization"] = f"Bearer {access_token}"
        elif auth_type == "bearer":
            api_key = creds.get("api_key", "")
            prefix = auth_config.get("prefix", "Bearer")
            headers["Authorization"] = f"{prefix} {api_key}"
        elif auth_type == "basic":
            auth = (
                creds.get("account_sid", ""),
                creds.get("auth_token", ""),
            )

        # Add extra headers (e.g., Notion-Version)
        extra_headers = auth_config.get("extra_headers", {})
        headers.update(extra_headers)

        # Copy arguments to avoid mutation
        params = dict(arguments)

        # Apply config-driven request transforms (no hardcoded provider logic!)
        # Transforms are defined in the capability's request_transform field
        params = apply_request_transform(params, capability, creds)

        if self.debug:
            print(f"[AgenticProcessor] {method} {url}")
            print(f"[AgenticProcessor] Params: {params}")

        # Determine content type from provider config (default to JSON)
        content_type = provider_def.get("request_content_type", "json")
        use_form_data = content_type == "form"

        # Make the HTTP request
        try:
            with httpx.Client(timeout=30) as client:
                if method == "GET":
                    response = client.get(
                        url, headers=headers, params=params, auth=auth
                    )
                elif method == "POST":
                    if use_form_data:
                        # Form-urlencoded (Stripe, Twilio, etc.)
                        response = client.post(
                            url, headers=headers, data=params, auth=auth
                        )
                    else:
                        headers["Content-Type"] = "application/json"
                        response = client.post(
                            url, headers=headers, json=params, auth=auth
                        )
                elif method == "PUT":
                    headers["Content-Type"] = "application/json"
                    response = client.put(
                        url, headers=headers, json=params, auth=auth
                    )
                elif method == "PATCH":
                    headers["Content-Type"] = "application/json"
                    response = client.patch(
                        url, headers=headers, json=params, auth=auth
                    )
                elif method == "DELETE":
                    response = client.delete(
                        url, headers=headers, params=params, auth=auth
                    )
                else:
                    return {"error": f"Unsupported method: {method}"}

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
            except Exception:
                error_body = e.response.text
            return {
                "error": f"API error ({e.response.status_code})",
                "details": error_body,
            }
        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # Main processing methods
    # =========================================================================

    def process(self, query: str) -> AgentResult:
        """
        Process a natural language query using agentic tool calling.

        The AI will understand the query, call tools as needed, reason about
        results, and synthesize a human-friendly response.

        Args:
            query: Natural language query (e.g., "get my stripe balance")

        Returns:
            AgentResult with the response and execution steps.
        """
        if self.ai_provider == "anthropic":
            return self._process_with_claude(query)
        else:
            # OpenAI, Groq, Ollama all use OpenAI-compatible API
            return self._process_with_openai(query)

    def _process_with_claude(self, query: str) -> AgentResult:
        """Process query using Claude (Anthropic)."""
        tools = self._build_claude_tools()

        if not tools or len(tools) <= 2:
            return AgentResult(
                success=False,
                response="No API providers are available. Please add provider JSON files to the providers/ directory.",
                error="No provider definitions loaded",
                status="error",
            )

        system_prompt = self._build_system_prompt()
        messages = [{"role": "user", "content": query}]

        steps = []
        all_tool_results = []
        consecutive_failures = 0
        failed_tools = set()
        pending_auth = []

        for step in range(self.max_steps):
            if consecutive_failures >= 3:
                return AgentResult(
                    success=True,
                    response=(
                        f"I've tried several approaches but keep running into issues.\n\n"
                        f"Failed operations: {', '.join(failed_tools)}\n\n"
                        f"Please provide more specific details."
                    ),
                    steps=steps,
                    status="needs_input",
                    question="Could you provide more details?",
                )

            if self.debug:
                print(
                    f"[AgenticProcessor/Claude] Step {step + 1}/{self.max_steps}"
                )

            try:
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:
                return AgentResult(
                    success=False,
                    response=f"AI error: {str(e)}",
                    error=str(e),
                    steps=steps,
                    status="error",
                )

            # Done — no more tool calls
            if response.stop_reason == "end_turn":
                text_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text

                return AgentResult(
                    success=True,
                    response=text_content or "Done.",
                    steps=steps,
                    raw_data=all_tool_results[-1]
                    if all_tool_results
                    else None,
                )

            elif response.stop_reason == "tool_use":
                # Serialize assistant content blocks
                assistant_content = []
                for block in response.content:
                    if block.type == "tool_use":
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input,
                            }
                        )
                    elif hasattr(block, "text"):
                        assistant_content.append(
                            {"type": "text", "text": block.text}
                        )
                messages.append(
                    {"role": "assistant", "content": assistant_content}
                )

                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_id = block.id
                        arguments = block.input or {}

                        if self.debug:
                            print(
                                f"[AgenticProcessor/Claude] Calling: {tool_name}"
                            )

                        # Handle ask_user
                        if tool_name == "ask_user":
                            question = arguments.get(
                                "question", "I need more information."
                            )
                            options = arguments.get("options")
                            context = arguments.get("context")

                            steps.append(
                                {
                                    "tool": "ask_user",
                                    "arguments": arguments,
                                    "result": {"status": "needs_input"},
                                }
                            )

                            response_text = question
                            if context:
                                response_text = f"{context}\n\n{question}"

                            return AgentResult(
                                success=True,
                                response=response_text,
                                steps=steps,
                                status="needs_input",
                                question=question,
                                options=options,
                            )

                        # Handle request_provider_connection — collect, don't return yet
                        if tool_name == "request_provider_connection":
                            result = self._execute_tool(tool_name, arguments)
                            provider = arguments.get("provider", "unknown")
                            reason = arguments.get("reason", "")
                            auth_type = result.get("auth_type", "api_key")
                            provider_display = result.get(
                                "provider_name", provider
                            )

                            steps.append(
                                {
                                    "tool": "request_provider_connection",
                                    "arguments": arguments,
                                    "result": result,
                                }
                            )

                            if result.get("status") == "already_connected":
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": json.dumps(result),
                                    }
                                )
                                continue

                            pending_auth.append(
                                {
                                    "provider": provider,
                                    "provider_name": provider_display,
                                    "auth_type": auth_type,
                                    "reason": reason,
                                }
                            )

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": json.dumps(
                                        {
                                            "status": "auth_required",
                                            "message": f"User will be prompted to connect {provider_display}.",
                                        }
                                    ),
                                }
                            )
                            continue

                        # Execute the API tool
                        result = self._execute_tool(tool_name, arguments)

                        if self.debug:
                            print(
                                f"[AgenticProcessor/Claude] Result: {json.dumps(result, indent=2)[:500]}..."
                            )

                        if "error" in result:
                            consecutive_failures += 1
                            failed_tools.add(
                                tool_name.replace("__", " → ")
                            )
                        else:
                            consecutive_failures = 0

                        steps.append(
                            {
                                "tool": tool_name,
                                "arguments": arguments,
                                "result": result,
                            }
                        )
                        all_tool_results.append(result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(result),
                            }
                        )

                # Return batched auth requests
                if pending_auth:
                    providers_needed = [
                        p["provider_name"] for p in pending_auth
                    ]
                    providers_list = ", ".join(providers_needed)

                    return AgentResult(
                        success=True,
                        response=(
                            f"To complete your request, I need access to: {providers_list}.\n\n"
                            f"Please connect them and try again."
                        ),
                        steps=steps,
                        status="needs_input",
                        question=f"Please connect: {providers_list}",
                        options=[f"Connect {p}" for p in providers_needed],
                        data={
                            "needs_auth": True,
                            "providers": pending_auth,
                        },
                    )

                messages.append({"role": "user", "content": tool_results})

            else:
                return AgentResult(
                    success=False,
                    response=f"Unexpected stop reason: {response.stop_reason}",
                    error=f"Unexpected stop reason: {response.stop_reason}",
                    steps=steps,
                    status="error",
                )

        return AgentResult(
            success=False,
            response="Max steps reached. Please try a simpler query.",
            error="Max steps reached",
            steps=steps,
            status="error",
        )

    def _process_with_openai(self, query: str) -> AgentResult:
        """Process query using OpenAI-compatible API (OpenAI, Groq, Ollama)."""
        tools = self._build_tools()

        if not tools or len(tools) <= 2:
            return AgentResult(
                success=False,
                response="No API providers are available. Please add provider JSON files to the providers/ directory.",
                error="No provider definitions loaded",
                status="error",
            )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query},
        ]

        steps = []
        all_tool_results = []
        consecutive_failures = 0
        failed_tools = set()
        pending_auth = []

        for step in range(self.max_steps):
            if consecutive_failures >= 3:
                return AgentResult(
                    success=True,
                    response=(
                        f"I've tried several approaches but keep running into issues.\n\n"
                        f"Failed operations: {', '.join(failed_tools)}\n\n"
                        f"Please provide more specific details."
                    ),
                    steps=steps,
                    status="needs_input",
                    question="Could you provide more details?",
                )

            if self.debug:
                print(
                    f"[AgenticProcessor/OpenAI] Step {step + 1}/{self.max_steps}"
                )

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as e:
                return AgentResult(
                    success=False,
                    response=f"AI error: {str(e)}",
                    error=str(e),
                    steps=steps,
                    status="error",
                )

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    if self.debug:
                        print(
                            f"[AgenticProcessor/OpenAI] Calling: {tool_name}"
                        )

                    if tool_name == "ask_user":
                        question = arguments.get(
                            "question", "I need more information."
                        )
                        options = arguments.get("options")
                        context = arguments.get("context")

                        steps.append(
                            {
                                "tool": "ask_user",
                                "arguments": arguments,
                                "result": {"status": "needs_input"},
                            }
                        )

                        response_text = question
                        if context:
                            response_text = f"{context}\n\n{question}"

                        return AgentResult(
                            success=True,
                            response=response_text,
                            steps=steps,
                            status="needs_input",
                            question=question,
                            options=options,
                        )

                    # Handle request_provider_connection — collect
                    if tool_name == "request_provider_connection":
                        result = self._execute_tool(tool_name, arguments)
                        provider = arguments.get("provider", "unknown")
                        reason = arguments.get("reason", "")
                        auth_type = result.get("auth_type", "api_key")
                        provider_display = result.get(
                            "provider_name", provider
                        )

                        steps.append(
                            {
                                "tool": "request_provider_connection",
                                "arguments": arguments,
                                "result": result,
                            }
                        )

                        if result.get("status") == "already_connected":
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(result),
                                }
                            )
                            continue

                        pending_auth.append(
                            {
                                "provider": provider,
                                "provider_name": provider_display,
                                "auth_type": auth_type,
                                "reason": reason,
                            }
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(
                                    {
                                        "status": "auth_required",
                                        "message": f"User will be prompted to connect {provider_display}.",
                                    }
                                ),
                            }
                        )
                        continue

                    # Execute the API tool
                    result = self._execute_tool(tool_name, arguments)

                    if self.debug:
                        print(
                            f"[AgenticProcessor/OpenAI] Result: {json.dumps(result, indent=2)[:500]}..."
                        )

                    if "error" in result:
                        consecutive_failures += 1
                        failed_tools.add(
                            tool_name.replace("__", " → ")
                        )
                    else:
                        consecutive_failures = 0

                    steps.append(
                        {
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    all_tool_results.append(result)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

                # Return batched auth requests
                if pending_auth:
                    providers_needed = [
                        p["provider_name"] for p in pending_auth
                    ]
                    providers_list = ", ".join(providers_needed)

                    return AgentResult(
                        success=True,
                        response=(
                            f"To complete your request, I need access to: {providers_list}.\n\n"
                            f"Please connect them and try again."
                        ),
                        steps=steps,
                        status="needs_input",
                        question=f"Please connect: {providers_list}",
                        options=[
                            f"Connect {p}" for p in providers_needed
                        ],
                        data={
                            "needs_auth": True,
                            "providers": pending_auth,
                        },
                    )
            else:
                return AgentResult(
                    success=True,
                    response=message.content or "Done.",
                    steps=steps,
                    raw_data=all_tool_results[-1]
                    if all_tool_results
                    else None,
                )

        return AgentResult(
            success=False,
            response="Max steps reached. Please try a simpler query.",
            error="Max steps reached",
            steps=steps,
            status="error",
        )


# =============================================================================
# Convenience function
# =============================================================================


def process_query(query: str, **provider_credentials) -> AgentResult:
    """
    Quick way to process a query with the agentic processor.

    Example:
        result = process_query(
            "get my stripe balance",
            stripe={"api_key": "sk_live_..."}
        )
        print(result.response)
    """
    processor = AgenticProcessor()
    for provider, creds in provider_credentials.items():
        processor.add_provider(provider, **creds)
    return processor.process(query)
