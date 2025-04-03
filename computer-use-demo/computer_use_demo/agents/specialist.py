"""
Specialist agent for multi-agent system.
"""

from typing import Dict, Optional

from computer_use_demo.agents.agent import SYSTEM_PROMPT, Agent
from computer_use_demo.agents.logging import get_logger
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.interfaces import ToolVersion

from .specialist_types import SPECIALIST_TYPES, get_full_prompt

# Get logger for this module
logger = get_logger("computer_use_demo.agents.specialist")

# Additional prompt instructions for specialists about returning control to manager
RETURN_TO_MANAGER_INSTRUCTIONS = """
<RETURN_TO_MANAGER>
You have access to a special tool called "return_to_manager" that allows you to return control to the Manager Agent.
Use this tool in the following situations:

1. When you have COMPLETED your assigned task successfully
2. When a user EXPLICITLY asks to speak with or return to the manager
3. When you encounter a situation outside your specialization that requires the manager's attention

When using this tool, provide:
- A clear summary of what you've accomplished so far
- Whether the task was completed successfully or not
- Any relevant context the manager should know

Example usage:
If the user asks "can you call the manager?", you should use the return_to_manager tool with appropriate parameters.

IMPORTANT: Always return to the manager when your task is complete or when the user asks to speak with the manager.
</RETURN_TO_MANAGER>
"""


class SpecialistAgent(Agent):
    """Specialist agent that executes specific tasks."""

    def __init__(
        self,
        history_tree: HistoryTree,
        agent_id: str,
        manager_agent: "Agent",
        specialist_type: str = "general",
        system_prompt: str = SYSTEM_PROMPT,
        tool_version: ToolVersion = "specialist_only_20250124",  # Changed default to specialist tools
        metadata: Optional[Dict[str, str]] = None,
        agent_tool_use_id: str = "",  # ID инструмента agent, который создал этого специалиста
    ) -> None:
        """Initialize SpecialistAgent.

        Args:
            history_tree: History tree for tracking all interactions (required)
            agent_id: Unique identifier for this agent
            manager_agent: Reference to the manager agent (required)
            specialist_type: Type of specialist (web_auth, lovable_bot, general)
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use
            metadata: Optional metadata for this specialist
            agent_tool_use_id: ID of the agent tool use that created this specialist
        """
        # Проверяем, что agent_tool_use_id не пустой
        if not agent_tool_use_id:
            error_msg = (
                f"Cannot create specialist '{agent_id}' without valid agent_tool_use_id"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if specialist_type not in SPECIALIST_TYPES:
            logger.warning(
                f"Unknown specialist type '{specialist_type}', defaulting to 'general'"
            )
            specialist_type = "general"

        # Add specialist-specific prompt to system prompt with return_to_manager instructions
        specialist_prompt = get_full_prompt(specialist_type)
        enhanced_prompt = (
            f"{system_prompt}{specialist_prompt}{RETURN_TO_MANAGER_INSTRUCTIONS}"
        )

        super().__init__(agent_id, history_tree, enhanced_prompt, tool_version)

        self.specialist_type = specialist_type
        self.metadata = metadata or {}

        # Сохраняем ссылку на manager_agent
        self.manager_agent = manager_agent

        # Сохраняем ID инструмента agent, который создал этого специалиста
        self.agent_tool_use_id = agent_tool_use_id
        logger.debug(
            f"SpecialistAgent '{agent_id}' initialized with agent_tool_use_id: {agent_tool_use_id}"
        )

        logger.debug(
            f"SpecialistAgent '{agent_id}' initialized with type '{specialist_type}'"
        )
