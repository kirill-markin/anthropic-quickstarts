"""
Manager agent for multi-agent system.
"""

from typing import Any, Dict, List, Optional, cast

from anthropic.types.beta import BetaMessageParam

from computer_use_demo.agents.agent import SYSTEM_PROMPT, Agent
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.interfaces import APIProvider, ToolVersion

from .logging import get_logger
from .specialist_types import SPECIALIST_TYPES

# Get logger for this module
logger = get_logger("computer_use_demo.agents.manager")


# Build manager prompt from specialist types
def _build_manager_prompt() -> str:
    """Build the manager prompt using information from the specialist types."""
    specialist_descriptions: List[str] = []

    for type_id, specialist_type in SPECIALIST_TYPES.items():
        specialist_descriptions.append(f"""
{specialist_type.id}. {specialist_type.name} ("{type_id}")
   - {specialist_type.description}
""")

    return f"""
<AGENT_ROLE>
You are the Manager Agent responsible for coordinating task execution in a multi-agent system.
Your primary responsibilities are:
1. Understand user requests and break them down into subtasks
2. Delegate subtasks to specialized agents with appropriate expertise
3. Synthesize results from specialized agents and provide cohesive responses
4. Maintain a high-level view of the task progress

IMPORTANT: You can ONLY take screenshots of the screen, but CANNOT directly interact with the computer.
For any mouse clicks, keyboard input, or other computer interactions, you MUST delegate to a specialist agent.

You have access to the following specialized agents:

{"".join(specialist_descriptions)}

When receiving a user request:
1. Analyze what needs to be done
2. Choose the appropriate specialized agent for the task based on their expertise
3. Provide the specialist with clear instructions and necessary context
4. Present the final results to the user in a coherent manner

Always delegate to the most appropriate specialist. For complex tasks that span multiple specialties,
break them down into subtasks and delegate each to the most suitable specialist in sequence.

Use the "agent" tool to delegate tasks to specialists. The output will be returned to you once
the specialist completes their assigned task.
</AGENT_ROLE>
"""


# Manager-specific system prompt addition
MANAGER_PROMPT_SUFFIX = _build_manager_prompt()

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
    """Manager agent that coordinates specialized agents."""

    def __init__(
        self,
        agent_id: str = "manager",
        system_prompt: str = SYSTEM_PROMPT,
        tool_version: ToolVersion = "manager_only_20250124",
        specialists: Optional[Dict[str, Agent]] = None,
        settings: Optional[Dict[str, Any]] = None,
        history_tree: Optional[HistoryTree] = None,
    ) -> None:
        """Initialize ManagerAgent with specialized agents.

        Args:
            agent_id: Unique identifier for this agent
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use
            specialists: Dictionary mapping specialist IDs to Agent instances
            settings: Optional custom settings to override defaults
            history_tree: Optional global history tree to track interactions
        """
        # Add manager-specific suffix to system prompt
        enhanced_prompt = f"{system_prompt}{MANAGER_PROMPT_SUFFIX}"
        super().__init__(agent_id, enhanced_prompt, tool_version, history_tree)

        # Initialize specialists dictionary
        self.specialists = specialists or {}

        # Apply settings with defaults
        self.settings = DEFAULT_MANAGER_SETTINGS.copy()
        if settings:
            self.settings.update(settings)

        logger.debug(
            f"ManagerAgent initialized with {len(self.specialists)} specialists"
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

        logger.debug(f"Registered specialist '{specialist_id}' with manager")

    async def delegate_task(
        self,
        specialist_id: str,
        task: str,
        context: Optional[str] = None,
        context_messages: Optional[List[BetaMessageParam]] = None,
    ) -> Dict[str, Any]:
        """Delegate a task to a specialist agent.

        Args:
            specialist_id: ID of the specialist to delegate to
            task: The task description to send to the specialist
            context: Optional context string to provide to the specialist
            context_messages: Optional context messages to provide to the specialist

        Returns:
            A dictionary with the session result information

        Raises:
            ValueError: If specialist_id is not registered
        """
        if specialist_id not in self.specialists:
            error_msg = f"Specialist '{specialist_id}' not registered with manager"
            logger.error(error_msg)
            raise ValueError(error_msg)

        specialist = self.specialists[specialist_id]
        logger.debug(f"Delegating task to specialist '{specialist_id}': {task[:50]}...")

        # Start specialist session in the history tree if available
        session_id = None
        if self.history_tree:
            session_id = self.history_tree.start_specialist_session(
                manager_id=self.agent_id,
                specialist_id=specialist_id,
                task=task,
                context=context,
            )
            logger.debug(
                f"Started specialist session in history tree, session_id={session_id}"
            )

        # Prepare the initial messages for the specialist with the task
        initial_messages: List[BetaMessageParam] = []
        if context_messages:
            initial_messages.extend(context_messages)

        # Add the task message
        task_content = task
        if context:
            task_content = f"Context:\n{context}\n\nTask:\n{task}"

        initial_messages.append(
            {"role": "user", "content": [{"type": "text", "text": task_content}]}
        )

        # Reset specialist's history before delegation
        specialist.history.messages = initial_messages

        # Execute the specialist agent with appropriate settings
        try:
            # Run the specialist with inherited settings from the manager
            # Use proper type casting to avoid type errors
            result_messages = await specialist.run(
                messages=specialist.history.messages,
                model=str(self.settings.get("model", "")),
                provider=cast(
                    APIProvider, self.settings.get("provider", APIProvider.ANTHROPIC)
                ),
                system_prompt_suffix="",  # Specialist has its own prompt
                api_key=str(self.settings.get("api_key", "")),
                only_n_most_recent_images=int(
                    self.settings.get("only_n_most_recent_images", 3)
                ),
                max_tokens=int(self.settings.get("output_tokens", 4096)),
                thinking_budget=int(self.settings.get("thinking_budget", 2048))
                if self.settings.get("thinking_enabled", False)
                else None,
                token_efficient_tools_beta=bool(
                    self.settings.get("token_efficient_tools_beta", True)
                ),
            )

            # Extract the final response
            final_response = None
            for msg in reversed(result_messages):
                if msg["role"] == "assistant":
                    final_response = msg
                    break

            success = True
            result_data = {
                "specialist_id": specialist_id,
                "task": task,
                "response": final_response,
                "success": success,
            }

            # End the specialist session in the history tree if available
            if self.history_tree and session_id:
                self.history_tree.end_specialist_session(
                    session_id=session_id, result=result_data, success=success
                )
                logger.debug(
                    f"Ended specialist session in history tree, session_id={session_id}"
                )

            return result_data

        except Exception as e:
            logger.error(f"Error during specialist task execution: {str(e)}")

            # End the specialist session with error in the history tree if available
            if self.history_tree and session_id:
                error_data = {
                    "specialist_id": specialist_id,
                    "task": task,
                    "error": str(e),
                    "success": False,
                }

                self.history_tree.end_specialist_session(
                    session_id=session_id, result=error_data, success=False
                )
                logger.debug(
                    f"Ended specialist session with error in history tree, session_id={session_id}"
                )

            # Re-raise the exception
            raise
