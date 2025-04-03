"""
Manager agent for multi-agent system that coordinates specialists.
"""

from typing import Any, Dict, Optional

from computer_use_demo.agents.agent import SYSTEM_PROMPT, Agent
from computer_use_demo.agents.logging import get_logger
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.interfaces import ToolVersion

# Get logger for this module
logger = get_logger("computer_use_demo.agents.manager")

# Additional instructions specifically for the manager agent
MANAGER_PROMPT_SUFFIX = """
<MANAGER_ROLE>
You are the Manager Agent in a multi-agent system. Your role is to:
1. Understand and decompose complex tasks
2. Delegate specific subtasks to specialist agents
3. Coordinate between multiple specialists
4. Synthesize the results of specialists' work
5. Maintain context and progress across the entire session

You have access to specialist agents with different capabilities through the 'agent' tool.
Each specialist has tools and knowledge for specific domains like web authentication, bot development, etc.

IMPORTANT: You should ALWAYS delegate work to specialist agents whenever possible and minimize direct computer interactions. You should only perform tasks directly when:
1. You need to understand a situation better to determine which specialist to delegate to
2. You need to verify or check results from specialists
3. The task is purely about coordinating between specialists

When delegating tasks:
- Be clear and specific about what each specialist should do
- Break down complex tasks into smaller subtasks that can be delegated
- Choose the most appropriate specialist based on their capabilities
- Provide relevant context from previous interactions
- Let specialists handle all direct computer interactions and tool usage
- Focus on coordination and oversight rather than execution

Remember: Your primary role is strategic oversight and delegation. Avoid direct computer interaction whenever possible - use specialists for all hands-on work with tools and systems.
</MANAGER_ROLE>

You are manager of this project. Please start.
"""

# Default settings for the Manager Agent
DEFAULT_MANAGER_SETTINGS = {
    "only_n_most_recent_images": 3,
    "hide_images": False,
    "token_efficient_tools_beta": True,
    "output_tokens": 8192,  # Manager needs more tokens to handle complex delegation
    "thinking_enabled": True,  # Enable thinking for complex decision making
    "thinking_budget": 4096,
    "model": "",  # Will be filled at runtime
    "provider": "anthropic",  # Default provider
    "api_key": "",  # Will be filled at runtime
}


class ManagerAgent(Agent):
    """Manager agent for multi-agent system."""

    def __init__(
        self,
        history_tree: HistoryTree,
        agent_id: str = "manager",
        system_prompt: str = SYSTEM_PROMPT + MANAGER_PROMPT_SUFFIX,
        tool_version: ToolVersion = "manager_only_20250124",
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the manager agent.

        Args:
            history_tree: History tree for tracking all interactions (required)
            agent_id: Unique identifier for this agent (default: "manager")
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use (should be manager-only version)
            settings: Optional settings dictionary
        """
        super().__init__(agent_id, history_tree, system_prompt, tool_version)
        self.settings = settings or {}
        self.specialists: Dict[str, Agent] = {}
        self.active_agent_id: str = agent_id
        logger.debug(f"ManagerAgent initialized with {len(self.settings)} settings")

    def get_active_agent(self) -> Agent:
        """Get the currently active agent.

        Returns:
            The active agent (self or a specialist)
        """
        if self.active_agent_id == self.agent_id:
            return self
        return self.specialists[self.active_agent_id]

    def set_active_agent(self, agent_id: str) -> None:
        """Set the active agent by ID.

        Args:
            agent_id: ID of the agent to set as active

        Raises:
            ValueError: If agent_id is not valid
        """
        if agent_id == self.agent_id:
            self.active_agent_id = agent_id
            logger.debug(f"Set active agent to manager: {agent_id}")
            return

        if agent_id not in self.specialists:
            raise ValueError(f"Agent {agent_id} not found in specialists")
        self.active_agent_id = agent_id
        logger.debug(f"Set active agent to specialist: {agent_id}")

    async def handle_user_message(self, message: str) -> None:
        """Handle a user message by forwarding it to the active agent.

        Args:
            message: The message from the user
        """
        logger.debug(
            f"Handling user message through active agent: {self.active_agent_id}"
        )

        # Get the active agent
        active_agent_id = self.active_agent_id
        active_agent = self.get_active_agent()

        # If we're the active agent, handle it directly
        if active_agent_id == self.agent_id:
            await super().handle_user_message(message)
        else:
            # Otherwise, forward to the specialist agent
            await active_agent.handle_user_message(message)

        logger.debug(
            f"Completed handling user message through agent: {active_agent_id}"
        )

    def register_specialist(self, specialist_id: str, specialist: Agent) -> None:
        """Register a specialist agent with this manager.

        Args:
            specialist_id: Unique identifier for the specialist
            specialist: The specialist Agent instance
        """
        self.specialists[specialist_id] = specialist
        # Share history tree with the specialist if we have one
        if self.history_tree and not specialist.history_tree:
            specialist.history_tree = self.history_tree

        # Устанавливаем ссылку на менеджера для специалиста
        specialist.manager_agent = self
        logger.debug(f"Set manager_agent reference for specialist '{specialist_id}'")

        # Set this specialist as the active agent
        self.set_active_agent(specialist_id)

        logger.debug(
            f"Registered specialist '{specialist_id}' with manager and set as active"
        )
