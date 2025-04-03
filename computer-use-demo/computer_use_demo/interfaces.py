"""
Shared interfaces and types for the Computer Use Demo.

This module contains the base interfaces and types used by both agents and tools,
avoiding circular dependencies between these modules.
"""

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Protocol, TypeAlias

# Define ToolVersion as a TypeAlias for compatibility with literal strings
ToolVersion: TypeAlias = Literal[
    "computer_use_20250124",
    "computer_use_20241022",
    "manager_only_20250124",
    "specialist_only_20250124",
]


class APIProvider(StrEnum):
    """API providers for Claude."""

    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


class AgentProtocol(Protocol):
    """Protocol defining the interface for an agent.

    This protocol allows tools to reference agents without importing the actual Agent class.
    """

    agent_id: str
    tool_version: str

    class HistoryProtocol(Protocol):
        """Protocol for agent history."""

        messages: List[Dict[str, Any]]

    history: HistoryProtocol

    async def run(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        provider: APIProvider,
        system_prompt_suffix: str,
        output_callback: Optional[Any] = None,
        tool_output_callback: Optional[Any] = None,
        api_response_callback: Optional[Any] = None,
        api_key: str,
        only_n_most_recent_images: Optional[int] = None,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        token_efficient_tools_beta: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run the agent."""
        ...
