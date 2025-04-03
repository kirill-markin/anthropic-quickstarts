"""
Collection of tools.
"""

from __future__ import annotations

from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from computer_use_demo.agents.logging import get_logger

from .base import BaseAnthropicTool, ToolFailure, ToolResult

# Get logger for this module
logger = get_logger("computer_use_demo.tools.collection")


class ToolCollection:
    """Collection of tools for an LLM agent."""

    def __init__(
        self,
        *tool_classes: Type[BaseAnthropicTool],
        tools: Optional[List[BaseAnthropicTool]] = None,
        manager_agent: Optional[Any] = None,
        agent_id: str,
        agent_tool_use_id: Optional[str] = None,
    ) -> None:
        """Initialize a collection of tools.

        Args:
            *tool_classes: Tool classes to instantiate
            tools: Optional precreated tool instances, preferred over tool_classes
            manager_agent: Reference to the agent (manager or specialist)
            agent_id: ID of the agent this collection is for (ОБЯЗАТЕЛЬНЫЙ параметр)
            agent_tool_use_id: Tool use ID for ReturnToManagerTool
        """
        # Обратная совместимость только для agent_tool_use_id
        if manager_agent is not None and agent_tool_use_id is None:
            # Пытаемся извлечь agent_tool_use_id из manager_agent
            if hasattr(manager_agent, "agent_tool_use_id"):
                agent_tool_use_id = getattr(manager_agent, "agent_tool_use_id", None)

        # Сохраняем ID агента
        self.agent_id = agent_id

        # Создаем пустой список инструментов
        self.tools = [] if tools is None else list(tools)

        # Создаем инструменты если not tools
        if not tools:
            for ToolClass in tool_classes:
                try:
                    # Специфическая обработка для ReturnToManagerTool20250124
                    if ToolClass.__name__ == "ReturnToManagerTool20250124":
                        # ReturnToManagerTool нужен только специалистам, пропускаем для менеджера
                        if agent_id == "manager":
                            logger.debug("Skipping ReturnToManagerTool for manager")
                            continue

                        # Для ReturnToManagerTool нужен agent_tool_use_id
                        if not agent_tool_use_id:
                            logger.debug(
                                "Skipping ReturnToManagerTool - missing agent_tool_use_id"
                            )
                            continue

                        # ReturnToManagerTool требует ссылку на реального менеджера
                        # Здесь мы просто передаем manager_agent, т.к. в цепочке вызовов
                        # для специалистов manager_agent - это и есть ссылка на менеджера
                        if not manager_agent:
                            logger.debug(
                                "Skipping ReturnToManagerTool - missing manager_agent"
                            )
                            continue

                        # Создаем инструмент
                        try:
                            tool = ToolClass(
                                manager_agent=manager_agent,
                                agent_tool_use_id=agent_tool_use_id,
                            )
                            self.tools.append(tool)
                            logger.debug(
                                f"Created ReturnToManagerTool with agent_tool_use_id: {agent_tool_use_id}"
                            )
                        except Exception as e:
                            logger.error(f"Failed to create ReturnToManagerTool: {e}")
                            continue

                    # Специфическая обработка для AgentTool20250124
                    elif ToolClass.__name__ == "AgentTool20250124":
                        # AgentTool нужен только менеджеру
                        if agent_id != "manager":
                            logger.debug("Skipping AgentTool - not a manager")
                            continue

                        # AgentTool требует ссылку на менеджера
                        if not manager_agent:
                            logger.debug("Skipping AgentTool - missing manager_agent")
                            continue

                        # Создаем инструмент
                        try:
                            tool = ToolClass(manager_agent=manager_agent)
                            self.tools.append(tool)
                            logger.debug("Created AgentTool for manager")
                        except Exception as e:
                            logger.error(f"Failed to create AgentTool: {e}")
                            continue

                    # Все остальные инструменты создаются без параметров
                    else:
                        tool = ToolClass()
                        self.tools.append(tool)

                except Exception as e:
                    logger.error(f"Failed to initialize tool {ToolClass.__name__}: {e}")

        # Создаем словарь инструментов по именам
        self.tool_by_name = {tool.name: tool for tool in self.tools}

        # Логируем результат
        logger.debug(
            f"Initialized tool collection with {len(self.tools)} tools: {', '.join(self.tool_by_name.keys())}"
        )

        # Проверяем наличие важных инструментов для отладки
        if "return_to_manager" in self.tool_by_name:
            logger.debug("ReturnToManagerTool is available")
        if "agent" in self.tool_by_name:
            logger.debug("AgentTool is available")

    def set_agent_id(self, agent_id: str) -> None:
        """Set the agent ID that's using this tool collection."""
        self.agent_id = agent_id
        logger.debug(f"Set agent ID for tool collection: {agent_id}")

    def to_params(self) -> List[dict]:
        """Convert all tools to API parameters.

        Returns:
            List of tool parameter dictionaries
        """
        return [tool.to_params() for tool in self.tools]

    async def run(
        self, name: str, tool_input: dict, calling_tool_use_id: str = "", **kwargs: Any
    ) -> Union[ToolResult, ToolFailure]:
        """Run a tool by name with the given input.

        Args:
            name: Name of the tool to run
            tool_input: Input for the tool
            calling_tool_use_id: Optional ID of the tool call that triggered this run
            **kwargs: Additional arguments to pass to the tool

        Returns:
            The result of running the tool
        """
        # Get tool by name
        if name not in self.tool_by_name:
            logger.error(f"Tool {name} not found in collection")
            return ToolFailure(error=f"Tool {name} not found in collection")

        tool = self.tool_by_name[name]
        tool_class_name = tool.__class__.__name__

        # Log tool execution
        logger.debug(f"Running tool {name} ({tool_class_name})")

        try:
            # Debugging: Print tool input
            logger.debug(f"Tool input: {tool_input}")

            # Transform input to match the tool's expected format
            args = {}

            # Если это инструмент agent и есть calling_tool_use_id, сохраним его
            if name == "agent" and calling_tool_use_id:
                logger.debug(
                    f"Adding calling_tool_use_id: {calling_tool_use_id} for agent tool"
                )

            # Copy all keys from tool_input to args
            for key, value in tool_input.items():
                args[key] = value

            # Special handling for ComputerTool20250124 which requires 'action' as a keyword argument
            if name == "computer" and "action" in tool_input and "action" not in args:
                logger.debug(
                    f"Adding 'action' from tool_input directly: {tool_input['action']}"
                )
                args["action"] = tool_input["action"]

            # Add any additional arguments from kwargs
            args.update(kwargs)

            # Handle complex nested tool_input values
            if name == "agent" and calling_tool_use_id:
                # Если это инструмент agent, передаем calling_tool_use_id как явный параметр
                result = await tool(calling_tool_use_id=calling_tool_use_id, **args)
            else:
                # Для других инструментов вызываем как обычно
                result = await tool(**args)

            # Log tool execution result
            if result.error:
                logger.error(f"Tool {name} failed: {result.error}")
            else:
                output_snippet = (result.output or "")[:100]
                logger.info(
                    f"Tool {name} completed successfully: {output_snippet}{'...' if len(output_snippet) >= 100 else ''}"
                )

            return result
        except Exception as e:
            logger.exception(f"Error running tool {name}: {e}")
            return ToolFailure(error=f"Error running tool {name}: {e}")

    def extract_tool_arguments(self, tool_input: dict, schema: dict) -> dict:
        """Extract and validate tool arguments from the input based on the schema.

        Args:
            tool_input: Input to the tool
            schema: JSON schema for the tool's input

        Returns:
            Validated and processed arguments to pass to the tool
        """
        try:
            # Extract properties from the schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # If there are no schema properties, just return the original input
            if not properties:
                return tool_input

            # Ensure required fields are present in our result
            result = {}
            for field in required:
                if field in tool_input:
                    result[field] = tool_input[field]

            # Add non-required fields if present
            for field in tool_input:
                if field not in result:
                    result[field] = tool_input[field]

            # Create a model for validation
            class InputModel(BaseModel):
                pass

            # Add fields dynamically based on schema
            for name, prop in properties.items():
                InputModel.__annotations__[name] = Any
                if name not in required:
                    # Make non-required fields optional with None default
                    InputModel.__annotations__[name] = Optional[Any]

            # Validate input against model
            try:
                validated_input = InputModel(**tool_input)
                # Use tool_input directly but validated through the model
                return result
            except ValidationError as e:
                logger.warning(
                    f"Validation error for tool input: {e}, using direct extraction instead"
                )
                return result

        except Exception as e:
            logger.error(f"Error processing tool input: {e}")
            # Return original input if anything goes wrong
            return tool_input
