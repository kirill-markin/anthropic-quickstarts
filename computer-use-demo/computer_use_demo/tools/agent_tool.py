"""
Tool for delegating tasks to specialist agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from computer_use_demo.agents.logging import get_logger
from computer_use_demo.agents.specialist_types import get_all_specialist_descriptions
from computer_use_demo.interfaces import APIProvider, ToolVersion

from .base import BaseAnthropicTool, ToolResult

# Get logger for this module
logger = get_logger("computer_use_demo.tools.agent_tool")


class AgentTool20250124(BaseAnthropicTool):
    """Tool for delegating tasks to specialist agents."""

    name = "agent"
    description = "Tool for delegating tasks to specialist agents"

    # For running specialist agents
    model: str = ""
    provider: APIProvider = APIProvider.ANTHROPIC
    api_key: str = ""

    # Configuration parameters
    max_tokens: int = 4096
    only_n_most_recent_images: Optional[int] = 3
    thinking_budget: Optional[int] = None
    system_prompt_suffix: str = ""
    token_efficient_tools_beta: bool = False
    tool_version: ToolVersion = "computer_use_20250124"  # Default tool version

    # Reference to manager agent (to access history_tree)
    # Use Any type to avoid circular imports
    manager_agent: Optional[Any] = None

    def __init__(self, manager_agent: Optional[Any] = None):
        """Initialize AgentTool with optional manager agent reference.

        Args:
            manager_agent: Optional reference to the manager agent
        """
        super().__init__()
        self.manager_agent = manager_agent
        logger.debug("AgentTool20250124 initialized")

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool.

        Returns:
            The JSON schema
        """
        # Get all available specialist types
        specialist_descriptions = get_all_specialist_descriptions()
        specialist_enum = list(specialist_descriptions.keys())

        # Format descriptions
        descriptions = "\n".join(
            [f"- {k}: {v}" for k, v in specialist_descriptions.items()]
        )

        return {
            "name": self.name,
            "description": f"{self.description}\n\nAvailable specialists:\n{descriptions}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "specialist": {
                        "type": "string",
                        "enum": specialist_enum,
                        "description": "The specialist agent to delegate the task to",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task to delegate to the specialist agent. Be specific about what needs to be done.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context information for the specialist. Only provide if necessary for the task.",
                    },
                },
                "required": ["specialist", "task"],
            },
        }

    def ensure_config(self, streamlit_state: Optional[Any] = None) -> bool:
        """Ensure that the tool has all necessary configuration."""
        import streamlit as st

        # Use provided state or get from Streamlit
        state = streamlit_state or st.session_state

        # Set up basic configuration from state
        if hasattr(state, "model"):
            self.model = str(state.model)

        if hasattr(state, "api_key"):
            self.api_key = str(state.api_key)

        if hasattr(state, "provider"):
            from computer_use_demo.agents.agent import APIProvider

            self.provider = APIProvider(state.provider)

        # Get additional settings if available
        if hasattr(state, "output_tokens"):
            self.max_tokens = int(state.output_tokens)

        if hasattr(state, "thinking_budget") and hasattr(state, "thinking"):
            self.thinking_budget = (
                int(state.thinking_budget) if state.thinking else None
            )

        if hasattr(state, "only_n_most_recent_images"):
            self.only_n_most_recent_images = int(state.only_n_most_recent_images)

        if hasattr(state, "token_efficient_tools_beta"):
            self.token_efficient_tools_beta = bool(state.token_efficient_tools_beta)

        if hasattr(state, "custom_system_prompt"):
            self.system_prompt_suffix = str(state.custom_system_prompt)

        # Verify all required parameters are set
        is_valid = bool(self.model and self.api_key)
        logger.debug(f"Configuration valid: {is_valid}")
        return is_valid

    async def _call(
        self,
        specialist: str,
        task: str,
        context: Optional[str] = None,
    ) -> ToolResult:
        """Delegate a task to a specialist agent.

        Args:
            specialist: ID of the specialist to delegate to
            task: The task description to send to the specialist
            context: Optional context information for the specialist

        Returns:
            ToolResult containing the specialist's response
        """
        # Import here to avoid circular dependency
        from computer_use_demo.agents import SpecialistAgent

        try:
            logger.debug(
                f"Creating new specialist '{specialist}' for task: {task[:50]}..."
            )

            # Ensure the tool is configured properly
            if not self.ensure_config():
                logger.error("Failed to ensure proper configuration for AgentTool")
                return ToolResult(
                    error="Agent tool not properly configured. Could not automatically fix the configuration."
                )

            # Create a new specialist agent for this task
            # We'll use a timestamp in the ID to ensure uniqueness
            specialist_id = f"{specialist}_{datetime.now().timestamp()}"

            # Create the specialist
            specialist_agent = SpecialistAgent(
                agent_id=specialist_id,
                specialist_type=specialist,
                tool_version=self.tool_version,
            )

            # Connect to history tree if manager agent is available
            if (
                self.manager_agent
                and hasattr(self.manager_agent, "history_tree")
                and self.manager_agent.history_tree
            ):
                # Share the manager's history tree with the specialist
                specialist_agent.history_tree = self.manager_agent.history_tree
                logger.debug(
                    f"Shared history tree from manager with specialist '{specialist_id}'"
                )

            # Prepare initial message for the specialist
            task_content = task
            if context:
                task_content = f"Context:\n{context}\n\nTask:\n{task}"

            # Create initial message for the specialist
            specialist_message: Dict[str, Any] = {
                "role": "user",
                "content": [{"type": "text", "text": task_content}],
            }

            # Set specialist's history to just this message
            specialist_agent.history.messages = [specialist_message]

            # Run the specialist agent - with no callbacks
            result_messages = await specialist_agent.run(
                messages=specialist_agent.history.messages,
                model=self.model,
                provider=self.provider,
                system_prompt_suffix=self.system_prompt_suffix,
                output_callback=None,
                tool_output_callback=None,
                api_response_callback=None,
                api_key=self.api_key,
                only_n_most_recent_images=self.only_n_most_recent_images,
                max_tokens=self.max_tokens,
                thinking_budget=self.thinking_budget,
                token_efficient_tools_beta=self.token_efficient_tools_beta,
            )

            # Extract the final assistant response
            assistant_messages = [
                msg for msg in result_messages if msg["role"] == "assistant"
            ]
            if not assistant_messages:
                logger.error(f"Specialist '{specialist}' did not provide a response")
                return ToolResult(
                    error=f"Specialist '{specialist}' did not provide a response"
                )

            # Get the last assistant message
            last_message = assistant_messages[-1]

            # Format the response
            response_blocks: List[str] = []
            if isinstance(last_message["content"], list):
                # Extract text content from the message
                for block in last_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            response_blocks.append(text)

            response_text = "\n\n".join(response_blocks)

            # Create a summary of the task execution
            summary = (
                f"Task executed by specialist '{specialist}':\n"
                f"Task: {task[:100]}{'...' if len(task) > 100 else ''}\n\n"
                f"Result:\n{response_text[:500]}{'...' if len(response_text) > 500 else ''}"
            )

            logger.info(
                f"Specialist '{specialist}' task execution completed successfully"
            )
            return ToolResult(
                output=summary,
                system=f"Task delegated to specialist '{specialist}' successfully.",
            )
        except Exception as e:
            logger.error(
                f"Error delegating task to specialist '{specialist}': {str(e)}"
            )
            return ToolResult(
                error=f"Error delegating task to specialist '{specialist}': {str(e)}"
            )

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Implement the abstract __call__ method from BaseAnthropicTool."""
        # Ensure configuration is valid before each call
        self.ensure_config()

        return await self._call(
            specialist=kwargs.get("specialist", ""),
            task=kwargs.get("task", ""),
            context=kwargs.get("context", None),
        )

    def to_params(self) -> Dict[str, Any]:
        """Implement the abstract to_params method from BaseAnthropicTool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_schema()["input_schema"],
        }
