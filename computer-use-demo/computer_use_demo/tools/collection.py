"""Collection classes for managing multiple tools."""

from typing import Any, Dict, List, Optional, Type

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(
        self,
        *tool_classes: Type[BaseAnthropicTool],
        manager_agent: Optional[Any] = None,
    ):
        """Initialize the tool collection.

        Args:
            *tool_classes: Tool classes to instantiate
            manager_agent: Optional reference to the manager agent, passed to tools that support it
        """
        self.tools: List[BaseAnthropicTool] = []
        self.tool_map: Dict[str, BaseAnthropicTool] = {}

        # Instantiate tools, passing manager_agent to those that support it
        for tool_cls in tool_classes:
            if tool_cls.__name__ == "AgentTool20250124" and manager_agent is not None:
                # Pass manager_agent to AgentTool
                tool = tool_cls(manager_agent=manager_agent)
            else:
                # Normal instantiation for other tools
                tool = tool_cls()

            self.tools.append(tool)
            # Use tool name attribute which is defined in BaseAnthropicTool
            self.tool_map[tool.name] = tool

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)
