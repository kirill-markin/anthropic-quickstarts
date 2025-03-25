from typing import Literal
import json
import logging

import streamlit as st
from anthropic.types.beta import BetaToolUnionParam

from .base import BaseAnthropicTool, ToolResult

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
            "description": "Switch between different specialized AI agents to handle different types of tasks"
        }

    async def __call__(self, agent: Literal["manager", "general", "login"], task: str) -> ToolResult:
        """Switch to a different agent with a specific task."""
        current_agent = get_current_agent()
        logger.info(f"Agent switch requested from {current_agent} to {agent} for task: {task}")
        
        # Import here to avoid circular import
        from computer_use_demo.streamlit import _handle_agent_switch
        
        # Add transition to chat history
        _handle_agent_switch(current_agent, agent, task)
        
        # Use JSON for structured data instead of string splitting
        agent_data = {
            "action": "SWITCH_AGENT",
            "agent": agent,
            "task": task
        }
        
        return ToolResult(
            output=f"Switching to {agent} agent with task: {task}",
            system=json.dumps(agent_data)
        ) 