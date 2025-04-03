"""
Tool for delegating tasks to specialist agents.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from anthropic.types.beta import BetaMessageParam, BetaToolUnionParam

from computer_use_demo.agents.logging import get_logger
from computer_use_demo.agents.specialist_types import get_all_specialist_descriptions
from computer_use_demo.interfaces import APIProvider, ToolVersion

from .base import BaseAnthropicTool, ToolResult

# Type hinting imports (only used during type checking)
if TYPE_CHECKING:
    from computer_use_demo.agents.manager import ManagerAgent

# Get logger for this module
logger = get_logger("computer_use_demo.tools.agent_tool")


class ReturnToManagerTool20250124(BaseAnthropicTool):
    """Tool for returning control from specialist agent back to manager agent."""

    name = "return_to_manager"
    description = "Return control back to the Manager Agent"

    # Reference to manager agent (to access history_tree)
    # Use string type annotation to avoid circular imports
    manager_agent: Optional[ManagerAgent] = None

    # ID инструмента agent, который создал специалиста (обязательный параметр)
    agent_tool_use_id: str

    def __init__(
        self, manager_agent: Optional[ManagerAgent] = None, agent_tool_use_id: str = ""
    ):
        """Initialize ReturnToManagerTool with required parameters.

        Args:
            manager_agent: Reference to the manager agent
            agent_tool_use_id: ID of the agent tool use that created the specialist
        """
        super().__init__()

        if not manager_agent:
            raise ValueError("ReturnToManagerTool requires a valid manager_agent")

        # Проверяем наличие необходимых атрибутов вместо проверки типа
        if not hasattr(manager_agent, "set_active_agent") or not hasattr(
            manager_agent, "history"
        ):
            error_msg = "ReturnToManagerTool requires a manager_agent with set_active_agent method and history attribute"
            logger.error(error_msg)
            raise TypeError(error_msg)

        self.manager_agent = manager_agent

        if not agent_tool_use_id:
            # Просто выбрасываем исключение - самый простой подход
            raise ValueError("ReturnToManagerTool requires a valid agent_tool_use_id")

        self.agent_tool_use_id = agent_tool_use_id
        logger.debug(
            f"ReturnToManagerTool20250124 initialized with agent_tool_use_id: {agent_tool_use_id}"
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool.

        Returns:
            The JSON schema
        """
        return {
            "name": self.name,
            "description": (
                "Use this tool when you want to return control back to the Manager Agent. "
                "Use this when one of these conditions is met:\n"
                "1. You have completed the task assigned to you\n"
                "2. The user explicitly asks to speak with the manager\n"
                "3. You encounter a situation outside your specialization that the manager should handle"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what you've accomplished and why you're returning control to the manager.",
                    },
                    "completed": {
                        "type": "boolean",
                        "description": "Whether the task was fully completed (true) or needs further work (false).",
                    },
                },
                "required": ["summary", "completed"],
            },
        }

    async def _call(
        self,
        summary: str,
        completed: bool,
    ) -> ToolResult:
        """Return control back to the manager agent.

        Args:
            summary: Brief summary of what was accomplished and why control is being returned
            completed: Whether the task was completed successfully

        Returns:
            ToolResult containing information about the control transfer
        """
        try:
            logger.debug(
                f"Specialist returning control to manager with summary: {summary[:50]}..."
            )

            # Проверка manager_agent нужна для линтера, но конструктор уже гарантирует, что manager_agent не None
            if not self.manager_agent:
                raise ValueError(
                    "ReturnToManagerTool._call: manager_agent is None, which should never happen"
                )

            # Проверяем наличие метода set_active_agent вместо проверки типа
            if not hasattr(self.manager_agent, "set_active_agent"):
                error_msg = "Cannot return to manager: manager_agent has no set_active_agent method"
                logger.error(error_msg)
                raise AttributeError(error_msg)

            # Reset active agent to manager
            self.manager_agent.set_active_agent("manager")

            # Get the specialist's ID (from the agent that called this tool)
            specialist_id = "unknown"
            if hasattr(self, "calling_agent_id") and self.calling_agent_id:
                specialist_id = self.calling_agent_id

            # Create a status message for the manager
            status = "completed successfully" if completed else "not fully completed"

            result_message = (
                f"Control returned from specialist '{specialist_id}' to Manager.\n\n"
                f"Task status: {status}\n\n"
                f"Summary: {summary}"
            )

            logger.info(
                f"Control returned to manager from specialist '{specialist_id}'"
            )

            # Используем agent_tool_use_id установленный при создании инструмента
            agent_tool_id = self.agent_tool_use_id

            # Если есть ID, создаем tool_result для связи с вызовом инструмента agent
            logger.debug(
                f"Creating tool_result with agent_tool_use_id: {agent_tool_id}"
            )

            # Удаляем добавление в историю менеджера, вместо этого полагаемся на стандартный механизм в Agent.run()
            # Мы оставляем только создание ToolResult, который будет обработан в Agent.run()

            return ToolResult(
                output=result_message,
                system=f"Control returned to manager agent. Task {status}.",
            )

        except Exception as e:
            logger.error(f"Error returning control to manager: {str(e)}")
            raise e

    async def __call__(self, **kwargs: Any) -> ToolResult:
        """Implement the abstract __call__ method from BaseAnthropicTool."""
        # Set the calling agent ID if available in the context
        if "calling_agent_id" in kwargs:
            self.calling_agent_id = kwargs.pop("calling_agent_id")

        return await self._call(
            summary=kwargs.get("summary", "Task execution completed."),
            completed=kwargs.get("completed", False),
        )

    def to_params(self) -> BetaToolUnionParam:
        """Implement the abstract to_params method from BaseAnthropicTool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_schema()["input_schema"],
        }


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
    # Use string type annotation to avoid circular imports
    manager_agent: Optional[ManagerAgent] = None

    def __init__(self, manager_agent: Optional[ManagerAgent] = None):
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
                raise ValueError(
                    "Agent tool not properly configured. Could not automatically fix the configuration."
                )

            # Проверка наличия manager_agent для создания специалиста
            if not self.manager_agent:
                error_msg = "Cannot create specialist: manager_agent reference not set"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create a new specialist agent for this task
            # We'll use a timestamp in the ID to ensure uniqueness
            specialist_id = f"{specialist}_{datetime.now().timestamp()}"

            # Принудительно устанавливаем tool_version для специалиста на specialist_only_20250124
            specialist_tool_version = "specialist_only_20250124"
            logger.debug(
                f"Setting specialist tool version to: {specialist_tool_version}"
            )

            # Проверка наличия ID инструмента agent для передачи специалисту
            agent_tool_use_id = getattr(self, "calling_tool_use_id", "")
            if not agent_tool_use_id:
                error_msg = (
                    "CRITICAL: Cannot create specialist without valid agent_tool_use_id"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(
                f"Creating specialist with agent_tool_use_id: {agent_tool_use_id}"
            )
            # Using is not None instead of a boolean check to fix linter error
            logger.debug(f"Manager agent exists: {self.manager_agent is not None}")

            # Создаем специалиста с правильной версией инструментов
            # Получаем history_tree от manager_agent
            if not (
                self.manager_agent
                and hasattr(self.manager_agent, "history_tree")
                and self.manager_agent.history_tree
            ):
                logger.error(
                    "CRITICAL: Невозможно создать специалиста без history_tree от менеджера"
                )
                raise ValueError(
                    "Cannot create specialist: No history_tree available from manager_agent"
                )

            # Log more detailed debugging information
            logger.debug(
                f"Creating specialist with agent_tool_use_id: {agent_tool_use_id}"
            )
            logger.debug(f"Manager agent exists: {self.manager_agent is not None}")

            specialist_agent = SpecialistAgent(
                history_tree=self.manager_agent.history_tree,
                agent_id=specialist_id,
                manager_agent=self.manager_agent,
                specialist_type=specialist,
                tool_version=specialist_tool_version,  # Используем фиксированную версию для специалистов
                agent_tool_use_id=agent_tool_use_id,  # Передаем ID инструмента agent
            )
            logger.debug(
                f"Created specialist agent '{specialist_id}' with manager_agent reference"
            )

            # Проверяем, что manager_agent правильно установлен
            if not specialist_agent.manager_agent:
                logger.error(
                    "CRITICAL: specialist_agent.manager_agent is None after initialization"
                )
                raise ValueError(
                    "Failed to properly initialize specialist with manager_agent reference"
                )

            # Double-check that agent_tool_use_id was passed correctly
            if (
                not hasattr(specialist_agent, "agent_tool_use_id")
                or not specialist_agent.agent_tool_use_id
            ):
                logger.error(
                    "CRITICAL: specialist_agent.agent_tool_use_id is missing or empty"
                )
                raise ValueError(
                    "Failed to properly set agent_tool_use_id on specialist agent"
                )

            logger.debug(
                f"Verified specialist '{specialist_id}' has agent_tool_use_id: {specialist_agent.agent_tool_use_id}"
            )

            # Connect to history tree if manager agent is available
            if (
                self.manager_agent
                and hasattr(self.manager_agent, "history_tree")
                and self.manager_agent.history_tree
            ):
                # Share the manager's history tree with the specialist
                # (Примечание: теперь это уже делается в конструкторе)
                logger.debug(
                    f"History tree already shared with specialist '{specialist_id}' via constructor"
                )

            # Register the specialist with the manager if available
            # This will also set it as the active agent
            if self.manager_agent:
                self.manager_agent.register_specialist(specialist_id, specialist_agent)
                logger.debug(f"Registered specialist '{specialist_id}' with manager")

            # Prepare initial message for the specialist
            task_content = task
            if context:
                task_content = f"Context:\n{context}\n\nTask:\n{task}"

            # Ensure the task content is not empty
            if not task_content.strip():
                task_content = "Please provide information or take action based on previous context."
                logger.warning(
                    f"Empty task content detected for specialist '{specialist_id}', using placeholder"
                )

            # Create initial message for the specialist
            specialist_message: BetaMessageParam = {
                "role": "user",
                "content": [{"type": "text", "text": task_content}],
            }

            # Set specialist's history to just this message
            specialist_agent.history.messages = [specialist_message]

            # Run the specialist agent - with no callbacks
            try:
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
            except Exception as run_error:
                # Handle API errors from the specialist run
                error_message = str(run_error)
                logger.error(
                    f"Error running specialist '{specialist_id}': {error_message}"
                )

                # Reset active agent back to manager if needed
                if self.manager_agent:
                    self.manager_agent.set_active_agent("manager")
                    logger.debug(
                        "Reset active agent back to manager after specialist error"
                    )

                raise run_error

            # Reset active agent back to manager if needed
            if self.manager_agent:
                self.manager_agent.set_active_agent("manager")
                logger.debug(
                    "Reset active agent back to manager after specialist completed"
                )

            # Проверяем, был ли вызван инструмент return_to_manager
            # Если специалист вызвал return_to_manager, active_agent_id уже будет "manager"
            was_returned_to_manager = (
                self.manager_agent and self.manager_agent.active_agent_id == "manager"
            )
            logger.debug(
                f"Checking if specialist returned control to manager: {was_returned_to_manager}"
            )

            # Extract the final assistant response
            assistant_messages = [
                msg for msg in result_messages if msg["role"] == "assistant"
            ]
            if not assistant_messages:
                logger.error(f"Specialist '{specialist_id}' did not provide a response")
                raise ValueError(
                    f"Specialist '{specialist_id}' did not provide a response"
                )

            # Get the last assistant message
            last_message = assistant_messages[-1]

            # Format the response
            response_blocks: List[str] = []
            if isinstance(last_message["content"], list):
                # Extract text content from the message
                for block in last_message["content"]:
                    if block.get("type") == "text":
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

            # Если специалист вызвал return_to_manager, логируем это событие
            if was_returned_to_manager:
                logger.debug(
                    "Specialist used return_to_manager, manager will continue on next user interaction"
                )
                # Менеджер уже должен был получить сообщение с результатом через инструмент return_to_manager,
                # поэтому нам не нужно запускать его здесь

            logger.info(
                f"Specialist '{specialist_id}' task execution completed successfully"
            )
            return ToolResult(
                output=summary,
                system=f"Task delegated to specialist '{specialist}' successfully.",
            )
        except Exception as e:
            logger.error(
                f"Error delegating task to specialist '{specialist}': {str(e)}"
            )
            raise e

    async def __call__(
        self, calling_tool_use_id: str = "", **kwargs: Any
    ) -> ToolResult:
        """Implement the abstract __call__ method from BaseAnthropicTool.

        Args:
            calling_tool_use_id: ID of the tool call that triggered this agent
            **kwargs: Other arguments specific to the tool
        """
        # Ensure configuration is valid before each call
        self.ensure_config()

        # Сохраняем ID инструмента, явно переданный при вызове
        if calling_tool_use_id:
            self.calling_tool_use_id = calling_tool_use_id
            logger.debug(
                f"AgentTool received calling_tool_use_id: {self.calling_tool_use_id}"
            )
        else:
            logger.error("AgentTool called without valid calling_tool_use_id")
            # Здесь мы не устанавливаем пустое значение, чтобы _call мог проверить и выбросить ошибку

        return await self._call(
            specialist=kwargs.get("specialist", ""),
            task=kwargs.get("task", ""),
            context=kwargs.get("context", None),
        )

    def to_params(self) -> BetaToolUnionParam:
        """Implement the abstract to_params method from BaseAnthropicTool."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_schema()["input_schema"],
        }
