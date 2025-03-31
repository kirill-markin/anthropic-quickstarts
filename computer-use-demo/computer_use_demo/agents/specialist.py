"""
Specialist agent for multi-agent system.
"""

from typing import Dict, Optional

from computer_use_demo.agents.agent import SYSTEM_PROMPT, Agent
from computer_use_demo.tools import ToolVersion

from .logging import get_logger
from .specialist_types import SPECIALIST_TYPES, get_full_prompt

# Get logger for this module
logger = get_logger("computer_use_demo.agents.specialist")


class SpecialistAgent(Agent):
    """Specialist agent that executes specific tasks."""

    def __init__(
        self,
        agent_id: str,
        specialist_type: str = "general",
        system_prompt: str = SYSTEM_PROMPT,
        tool_version: ToolVersion = "computer_use_20250124",
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize SpecialistAgent.

        Args:
            agent_id: Unique identifier for this agent
            specialist_type: Type of specialist (web_auth, lovable_bot, general)
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use
            metadata: Optional metadata for this specialist
        """
        if specialist_type not in SPECIALIST_TYPES:
            logger.warning(
                f"Unknown specialist type '{specialist_type}', defaulting to 'general'"
            )
            specialist_type = "general"

        # Add specialist-specific prompt to system prompt
        specialist_prompt = get_full_prompt(specialist_type)
        enhanced_prompt = f"{system_prompt}{specialist_prompt}"

        super().__init__(agent_id, enhanced_prompt, tool_version)

        self.specialist_type = specialist_type
        self.metadata = metadata or {}

        logger.debug(
            f"SpecialistAgent '{agent_id}' initialized with type '{specialist_type}'"
        )
