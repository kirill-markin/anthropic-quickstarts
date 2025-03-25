from typing import Literal, Any
import json
import logging

import streamlit as st
from anthropic.types.beta import BetaToolUnionParam

from .base import BaseAnthropicTool, ToolResult, ToolError

# Remove the circular import
# from computer_use_demo.streamlit import handle_agent_switch

# Настроить логгер
logger = logging.getLogger("computer_use_demo.tools.agent")


def get_current_agent() -> str:
    """Helper function to retrieve the current agent from session state."""
    return st.session_state.current_agent


class AgentTool(BaseAnthropicTool):
    """Tool to invoke specialized agents."""

    api_type: Literal["custom"] = "custom"
    name: Literal["agent"] = "agent"

    def to_params(self) -> BetaToolUnionParam:
        """Returns the tool parameters for the Anthropic API."""
        logger.debug("Creating agent tool parameters for API")
        return {
            "name": self.name,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["manager", "general", "login"],
                        "description": "The agent to switch to",
                    },
                    "task": {
                        "type": "string",
                        "description": "The task for the agent to perform",
                    },
                },
                "required": ["agent", "task"],
            },
            "description": "Switch between different specialized AI agents to handle different types of tasks",
        }

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Switch to a different agent with a specific task."""
        # Import handle_agent_switch inside the method to avoid circular imports
        from computer_use_demo.streamlit import handle_agent_switch

        agent: str = str(kwargs.get("agent", ""))
        task: str = str(kwargs.get("task", ""))

        if not agent or not task:
            raise ToolError("Both 'agent' and 'task' parameters are required")

        if agent not in ["manager", "general", "login"]:
            raise ToolError(
                f"Invalid agent: {agent}. Must be one of: manager, general, login"
            )

        agent_typed: Literal["manager", "general", "login"] = agent  # type: ignore

        current_agent = get_current_agent()
        logger.info(
            f"Agent switch requested from {current_agent} to {agent_typed} for task: {task}"
        )

        # Add transition to chat history
        handle_agent_switch(current_agent, agent_typed, task)

        # Ensure agent_switched flag is set to trigger a follow-up request
        st.session_state.agent_switched = True
        st.session_state.agent_switch_task = task

        # Use JSON for structured data instead of string splitting
        agent_data: dict[str, str] = {
            "action": "SWITCH_AGENT",
            "agent": agent,
            "task": task,
        }

        return ToolResult(
            output=f"Switching to {agent_typed} agent with task: {task}",
            system=json.dumps(agent_data),
        )
