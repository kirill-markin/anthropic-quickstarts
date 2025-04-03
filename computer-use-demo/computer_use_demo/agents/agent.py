"""
Agent implementation for Computer Use Demo.
"""

import platform
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, cast

import httpx
from anthropic import APIError, APIResponseValidationError, APIStatusError
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

# Import HistoryTree
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.interfaces import APIProvider, ToolVersion
from computer_use_demo.tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
)

from .history import History
from .logging import get_logger

# Get logger for this module
logger = get_logger("computer_use_demo.agents.agent")

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"
MAX_SPECIALIST_CYCLES = 100


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""


def response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    """Convert a BetaMessage to a list of BetaContentBlockParam."""
    res: list[BetaContentBlockParam] = []
    thinking_blocks: list[BetaContentBlockParam] = []
    other_blocks: list[BetaContentBlockParam] = []

    # First, separate thinking blocks from other blocks
    for block in response.content:
        if getattr(block, "type", None) == "thinking":
            # Handle thinking blocks - preserve them exactly as received from the API
            # We must not modify or create thinking blocks as they contain cryptographic signatures
            thinking_block = cast(BetaContentBlockParam, block.model_dump())
            thinking_blocks.append(thinking_block)
        elif isinstance(block, BetaTextBlock) and block.text and block.text.strip():
            other_blocks.append(
                cast(
                    BetaContentBlockParam,
                    BetaTextBlockParam(type="text", text=block.text),
                )
            )
        else:
            # Handle tool use blocks normally
            other_blocks.append(
                cast(
                    BetaContentBlockParam,
                    block.model_dump(),
                )
            )

    # Put thinking blocks first (required when thinking is enabled)
    res = thinking_blocks + other_blocks

    # If all blocks were filtered out, add a placeholder block
    if not res:
        logger.warning("All content blocks were empty, adding placeholder block")
        res.append(
            cast(
                BetaContentBlockParam,
                BetaTextBlockParam(
                    type="text", text="No meaningful response content available."
                ),
            )
        )

    # We will NOT attempt to create placeholder thinking blocks as they require valid signatures
    # that can only be generated by the Claude API

    return res


def inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints conservatively to avoid exceeding API limits
    Maximum 3 cache breakpoints are allowed for the entire request
    (system prompt gets 1, leaving 2 for user messages)
    """
    # We'll only add cache_control to at most one message block to avoid the API limit
    breakpoints_remaining = 0  # Changed from 1 to 0 to be extremely conservative
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            # First, remove any existing cache_control to avoid accumulation
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    cast(Dict[str, Any], block).pop("cache_control", None)

            # Then add a new one if we have budget and it's a non-empty text block
            if breakpoints_remaining and content and len(content) > 0:
                last_block = content[-1]
                # Only add cache_control for non-empty text blocks
                if (
                    isinstance(last_block, dict)
                    and last_block.get("type") == "text"
                    and last_block.get("text", "").strip()
                ):
                    breakpoints_remaining -= 1
                    # Use type ignore to bypass TypedDict check until SDK types are updated
                    last_block["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                        {"type": "ephemeral"}
                    )
            # No need to continue after processing one message
            break


def maybe_prepend_system_tool_result(result: ToolResult, result_text: str) -> str:
    """Prepend system information to tool result if available."""
    if result.system:
        result_text = f"<s>{result.system}</s>\n{result_text}"
    return result_text


def make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


class Agent:
    """Base agent class for Computer Use Demo."""

    def __init__(
        self,
        agent_id: str,
        history_tree: HistoryTree,
        system_prompt: str = SYSTEM_PROMPT,
        tool_version: ToolVersion = "computer_use_20250124",
        manager_agent: Optional["Agent"] = None,
    ) -> None:
        """Initialize Agent with required parameters.

        Args:
            agent_id: Unique identifier for this agent
            history_tree: Global history tree to track interactions (required)
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use
            manager_agent: Optional reference to the manager agent
        """
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.tool_version = tool_version
        self.history = History()
        self.history_tree = history_tree
        # Флаг для отслеживания необходимости прерывания текущего цикла
        self.is_interrupted = False
        # Флаг для отслеживания того, что агент уже запущен
        self.is_running = False
        # Ссылка на manager_agent
        self.manager_agent = manager_agent
        logger.debug(f"Agent '{agent_id}' initialized with tool_version={tool_version}")

    def _fix_unclosed_tool_calls(self) -> None:
        """Check for unclosed tool calls in history and add synthetic tool results.

        This method scans the message history for tool_use blocks that don't have
        corresponding tool_result blocks in the next message, and adds synthetic
        tool_result blocks to fix the conversation history.
        """
        if not self.history.messages or len(self.history.messages) < 2:
            return

        messages = self.history.messages

        # Scan each pair of messages for unclosed tool calls
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            # Skip if not assistant message (only they can contain tool calls)
            if current_msg.get("role") != "assistant":
                continue

            # Extract tool_use IDs from current message
            tool_use_ids = []
            for content_item in current_msg.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "tool_use"
                ):
                    tool_use_ids.append(content_item.get("id", ""))

            # Skip if no tool uses
            if not tool_use_ids:
                continue

            # Skip if next message is not from user (should be tool results)
            if next_msg.get("role") != "user":
                continue

            # Find which tool_use IDs have matching tool_result blocks
            found_tool_results = set()
            for content_item in next_msg.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "tool_result"
                ):
                    found_tool_results.add(content_item.get("tool_use_id", ""))

            # Find missing tool results
            missing_tool_ids = [
                id for id in tool_use_ids if id not in found_tool_results
            ]

            # Add synthetic tool results for missing ones
            if missing_tool_ids:
                logger.debug(
                    f"Agent '{self.agent_id}' found {len(missing_tool_ids)} unclosed tool calls: {missing_tool_ids}"
                )

                # Create synthetic tool results
                new_tool_results = []
                for tool_id in missing_tool_ids:
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {
                                "type": "text",
                                "text": "human stopped or interrupted tool execution",
                            }
                        ],
                    }
                    new_tool_results.append(tool_result)

                    # Also add to history tree if available
                    if self.history_tree:
                        self.history_tree.add_tool_result(
                            agent_id="user",
                            tool_result=ToolResult(
                                output="(user stopped or interrupted and wrote the following)",
                                error="human stopped or interrupted tool execution",
                                base64_image="",
                            ),
                        )

                # Add all new tool results to the beginning of next message content
                content_list = list(next_msg.get("content", []))
                next_msg["content"] = new_tool_results + content_list

                logger.debug(
                    f"Agent '{self.agent_id}' added {len(new_tool_results)} synthetic tool results"
                )

    async def handle_user_message(self, message: str) -> None:
        """Handle a new user message, interrupting any ongoing tool calls.

        This method:
        1. Устанавливает флаг прерывания для остановки текущего цикла
        2. Checks for and handles any unclosed tool calls
        3. Adds the new user message to the agent's history
        4. Adds the user message to the history tree (if available)

        Args:
            message: The user message text
        """
        logger.debug(
            f"Agent '{self.agent_id}' handling user message: {message[:30]}..."
        )

        # Step 1: Устанавливаем флаг прерывания
        logger.debug(
            f"Agent '{self.agent_id}' setting interrupt flag to stop current run loop"
        )
        self.is_interrupted = True

        # Step 2: Check for and handle unclosed tool calls
        self._ensure_history_consistency()

        # Step 3: Create a text block for the message
        from anthropic.types.beta import BetaTextBlockParam

        text_block = BetaTextBlockParam(type="text", text=message)

        # Step 4: Add the message to the agent's history
        user_message = {"role": "user", "content": [text_block]}
        self.history.append(cast(BetaMessageParam, user_message))
        logger.debug(f"Agent '{self.agent_id}' added user message to history")

        # Step 5: Add message to the history tree (if available)
        if self.history_tree:
            self.history_tree.add_user_message(agent_id="user", content=[text_block])
            logger.debug(f"Agent '{self.agent_id}' added user message to history tree")

    async def run(
        self,
        *,
        messages: List[BetaMessageParam],
        model: str,
        provider: APIProvider,
        system_prompt_suffix: str,
        output_callback: Optional[Callable[[BetaContentBlockParam], None]] = None,
        tool_output_callback: Optional[Callable[[ToolResult, str], None]] = None,
        api_response_callback: Optional[
            Callable[
                [httpx.Request, httpx.Response | object | None, Exception | None], None
            ]
        ] = None,
        api_key: str,
        only_n_most_recent_images: Optional[int] = None,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        token_efficient_tools_beta: bool = False,
    ) -> List[BetaMessageParam]:
        """Run the agent loop with the given parameters.

        This is a wrapper around the original sampling_loop function, but maintains
        the history within the agent object.

        Args:
            messages: List of messages to process
            model: Model name to use
            provider: API provider (anthropic, bedrock, vertex)
            system_prompt_suffix: Additional text to append to system prompt
            output_callback: Optional callback for model output
            tool_output_callback: Optional callback for tool output
            api_response_callback: Optional callback for API responses
            api_key: API key for authentication
            only_n_most_recent_images: Limit history to N most recent images
            max_tokens: Maximum tokens for model response
            thinking_budget: Budget for model thinking
            token_efficient_tools_beta: Whether to use token efficient tools beta

        Returns:
            Updated list of messages after processing
        """
        logger.debug(f"Agent '{self.agent_id}' run method called with model={model}")

        # Сбрасываем флаг прерывания при начале нового цикла
        self.is_interrupted = False
        # Устанавливаем флаг, что агент запущен
        self.is_running = True
        logger.debug(
            f"Agent '{self.agent_id}' reset interrupt flag and set running state"
        )

        # Initialize a counter for loop cycles (for specialists to return to manager)
        loop_counter = 0
        is_specialist = self.agent_id != "manager"
        logger.debug(
            f"Agent '{self.agent_id}' is_specialist={is_specialist}, initializing loop_counter=0"
        )

        # Always ensure history is in a consistent state before proceeding
        self._ensure_history_consistency()

        # Use the provided messages initially, but maintain our own history for later
        if not self.history.messages:
            self.history.messages = messages
            logger.debug(
                f"Agent '{self.agent_id}' initialized history with {len(messages)} messages"
            )

            # Add initial messages to global history tree if available
            if self.history_tree:
                for msg in messages:
                    # Add each message to the history tree
                    role = msg.get("role", "")
                    content = msg.get("content", [])

                    # Convert content to list if it's a string
                    if isinstance(content, str):
                        content_list = [BetaTextBlockParam(type="text", text=content)]
                    else:
                        content_list = list(content)

                    if role == "user":
                        self.history_tree.add_user_message(
                            agent_id=self.agent_id,
                            content=cast(List[BetaContentBlockParam], content_list),
                        )
                    elif role == "assistant":
                        self.history_tree.add_assistant_message(
                            agent_id=self.agent_id,
                            content=cast(List[BetaContentBlockParam], content_list),
                        )
                    elif role == "system":
                        self.history_tree.add_system_message(
                            agent_id=self.agent_id,
                            content=cast(List[BetaContentBlockParam], content_list),
                        )
                logger.debug(
                    f"Agent '{self.agent_id}' added {len(messages)} initial messages to history tree"
                )
        else:
            # If we already have messages in history, just update with the latest
            for msg in messages:
                if msg not in self.history.messages:
                    self.history.append(msg)

                    # Add new message to global history tree if available
                    if self.history_tree:
                        role = msg.get("role", "")
                        content = msg.get("content", [])

                        # Convert content to list if it's a string
                        if isinstance(content, str):
                            content_list = [
                                BetaTextBlockParam(type="text", text=content)
                            ]
                        else:
                            content_list = list(content)

                        if role == "user":
                            self.history_tree.add_user_message(
                                agent_id=self.agent_id,
                                content=cast(List[BetaContentBlockParam], content_list),
                            )
                        elif role == "assistant":
                            self.history_tree.add_assistant_message(
                                agent_id=self.agent_id,
                                content=cast(List[BetaContentBlockParam], content_list),
                            )
                        elif role == "system":
                            self.history_tree.add_system_message(
                                agent_id=self.agent_id,
                                content=cast(List[BetaContentBlockParam], content_list),
                            )

            logger.debug(
                f"Agent '{self.agent_id}' updated history, now has {len(self.history.messages)} messages"
            )

        # Get tool collection for the specified version
        tool_group = TOOL_GROUPS_BY_VERSION[self.tool_version]

        # Pass self (the agent) as manager_agent to ToolCollection if this is the manager agent
        if self.agent_id == "manager":
            tool_collection = ToolCollection(
                *(ToolCls for ToolCls in tool_group.tools),
                manager_agent=self,
                agent_id=self.agent_id,
            )
            logger.debug(
                f"Agent '{self.agent_id}' created tool collection with manager_agent reference"
            )
        else:
            # Для специалистов тоже передаем ссылку на manager_agent, если она доступна
            manager_ref = getattr(self, "manager_agent", None)

            # Add logging to debug agent_tool_use_id
            has_agent_tool_use_id = hasattr(self, "agent_tool_use_id")
            agent_tool_use_id = (
                getattr(self, "agent_tool_use_id", "") if has_agent_tool_use_id else ""
            )
            logger.debug(
                f"Agent '{self.agent_id}' has agent_tool_use_id attribute: {has_agent_tool_use_id}, value: {agent_tool_use_id}"
            )

            # Дополнительный отладочный вывод для диагностики проблемы с ReturnToManagerTool
            logger.debug(
                f"TOOLS DEBUG: Agent '{self.agent_id}' tool_version={self.tool_version} used for tool group"
            )
            logger.debug(
                f"TOOLS DEBUG: Tool classes in group: {[ToolCls.__name__ for ToolCls in tool_group.tools]}"
            )

            if manager_ref:
                tool_collection = ToolCollection(
                    *(ToolCls for ToolCls in tool_group.tools),
                    manager_agent=manager_ref,
                    agent_id=self.agent_id,
                    agent_tool_use_id=agent_tool_use_id,
                )
                logger.debug(
                    f"Agent '{self.agent_id}' created tool collection with real manager_agent reference"
                )
            else:
                raise RuntimeError(
                    "manager_agent reference is required for specialist agents"
                )

        # Always log available tools for easier debugging
        logger.debug(
            f"TOOLS DEBUG: Agent '{self.agent_id}' has tools: {list(tool_collection.tool_by_name.keys())}"
        )

        # Проверка наличия необходимых инструментов для специалистов
        if self.agent_id != "manager" and "specialist" in self.tool_version:
            # Выводим полный список инструментов для отладки
            logger.debug(
                f"TOOLS DEBUG: Agent '{self.agent_id}' has tools: {list(tool_collection.tool_by_name.keys())}"
            )

            # Проверяем, есть ли инструмент return_to_manager для специалистов
            if "return_to_manager" not in tool_collection.tool_by_name:
                # Добавим детальную диагностику
                logger.error(
                    f"CRITICAL: Специалист '{self.agent_id}' не имеет доступа к инструменту 'return_to_manager'"
                )
                logger.error(f"TOOLS DEBUG: Tool version: {self.tool_version}")
                logger.error(
                    f"TOOLS DEBUG: Available tools: {list(tool_collection.tool_by_name.keys())}"
                )
                logger.error(
                    f"TOOLS DEBUG: manager_agent reference: {getattr(self, 'manager_agent', None) is not None}"
                )

                # Выбрасываем ошибку без попытки восстановления
                error_msg = f"CRITICAL: Специалист '{self.agent_id}' не имеет доступа к инструменту 'return_to_manager'. Убедитесь, что manager_agent передан правильно."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            logger.debug(
                f"Agent '{self.agent_id}' has required tool 'return_to_manager' available"
            )

        logger.debug(
            f"Agent '{self.agent_id}' created tool collection for version {self.tool_version}"
        )

        # Create system prompt with suffix if provided
        system = BetaTextBlockParam(
            type="text",
            text=f"{self.system_prompt}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
        )
        logger.debug(f"Agent '{self.agent_id}' created system prompt")

        # Main loop
        while True:
            # Increment loop counter for specialists
            if is_specialist:
                loop_counter += 1
                # Check if we've reached the cycle limit (50)
                if loop_counter >= MAX_SPECIALIST_CYCLES:
                    logger.debug(
                        f"Specialist '{self.agent_id}' reached {loop_counter} cycles, adding message to return to manager"
                    )

                    # Create an assistant message to return to manager
                    text_block = BetaTextBlockParam(
                        type="text",
                        text="Reached maximum cycle count (50). Return to manager with current results and overview of the message history.",
                    )

                    # Add the message to agent's history
                    self.history.messages.append(
                        {"role": "user", "content": [text_block]}
                    )

                    # Add to history tree if available
                    if self.history_tree:
                        self.history_tree.add_user_message(
                            agent_id=self.agent_id, content=[text_block]
                        )

                    # Reset counter so we don't keep triggering this
                    logger.debug(
                        f"Specialist '{self.agent_id}' reset loop counter after adding return message"
                    )

                    # Continue with the loop - the agent will naturally call return_to_manager
                    # on the next iteration as a response to this message

            # Проверяем флаг прерывания в начале каждой итерации цикла
            if self.is_interrupted:
                logger.debug(
                    f"Agent '{self.agent_id}' detected interrupt flag, breaking run loop"
                )
                # Сбрасываем флаг, но не запускаем новый цикл
                self.is_interrupted = False
                break

            logger.debug(f"Agent '{self.agent_id}' starting API call iteration")

            # IMPORTANT: Make sure history is consistent before each API call
            # This ensures there are no unclosed tool calls which would cause API errors
            unclosed_fixed = self._ensure_history_consistency()
            if unclosed_fixed:
                logger.debug(
                    f"Agent '{self.agent_id}' fixed unclosed tool calls before API call"
                )

            enable_prompt_caching = False
            betas = [tool_group.beta_flag] if tool_group.beta_flag else []
            if token_efficient_tools_beta:
                betas.append("token-efficient-tools-2025-02-19")

            # Configure API client based on provider
            if provider == APIProvider.ANTHROPIC:
                from anthropic import Anthropic

                client = Anthropic(api_key=api_key, max_retries=4)
                enable_prompt_caching = True
                logger.debug(f"Agent '{self.agent_id}' created Anthropic client")
            elif provider == APIProvider.VERTEX:
                from anthropic import AnthropicVertex

                client = AnthropicVertex()
                logger.debug(f"Agent '{self.agent_id}' created AnthropicVertex client")
            elif provider == APIProvider.BEDROCK:
                from anthropic import AnthropicBedrock

                client = AnthropicBedrock()
                logger.debug(f"Agent '{self.agent_id}' created AnthropicBedrock client")

            # Setup prompt caching if enabled
            if enable_prompt_caching:
                betas.append(PROMPT_CACHING_BETA_FLAG)
                inject_prompt_caching(self.history.messages)
                # Because cached reads are 10% of the price, we don't think it's
                # ever sensible to break the cache by truncating images
                only_n_most_recent_images = 0

                # Check existing cache_control blocks in messages first
                existing_cache_blocks = 0
                for msg in self.history.messages:
                    if isinstance(msg.get("content", []), list):
                        for block in msg["content"]:
                            if (
                                isinstance(block, dict)
                                and block.get("cache_control") is not None
                            ):
                                existing_cache_blocks += 1

                # Only add cache_control to system if we're not exceeding the limit
                if (
                    existing_cache_blocks < 3
                ):  # API allows maximum 4, we leave room for error
                    # Use type ignore to bypass TypedDict check until SDK types are updated
                    system["cache_control"] = {"type": "ephemeral"}  # type: ignore
                    logger.debug(
                        f"Agent '{self.agent_id}' added cache_control to system prompt"
                    )
                else:
                    logger.warning(
                        f"Agent '{self.agent_id}' skipped adding cache_control to system prompt - too many existing blocks ({existing_cache_blocks})"
                    )

                logger.debug(f"Agent '{self.agent_id}' enabled prompt caching")

            # Optimize history if needed
            if only_n_most_recent_images:
                self.history.optimize(
                    only_n_most_recent_images,
                    min_removal_threshold=only_n_most_recent_images,
                )
                logger.debug(
                    f"Agent '{self.agent_id}' optimized history to keep {only_n_most_recent_images} images"
                )

            # Setup for thinking if enabled
            extra_body = {}
            if thinking_budget:
                extra_body = {
                    "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
                }
                logger.debug(
                    f"Agent '{self.agent_id}' enabled thinking with budget {thinking_budget}"
                )

            # Call the API
            try:
                logger.debug(
                    f"Agent '{self.agent_id}' sending API request to {provider} with model {model}"
                )

                # Снова проверяем флаг прерывания перед отправкой запроса API
                if self.is_interrupted:
                    logger.debug(
                        f"Agent '{self.agent_id}' detected interrupt flag before API call, breaking run loop"
                    )
                    # Сбрасываем флаг, но не запускаем новый цикл
                    self.is_interrupted = False
                    break

                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=self.history.messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                    extra_body=extra_body,
                )
                logger.debug(f"Agent '{self.agent_id}' received API response")
            except (APIStatusError, APIResponseValidationError) as e:
                logger.error(f"Agent '{self.agent_id}' API error: {e}")
                if api_response_callback:
                    api_response_callback(e.request, e.response, e)
                # Сбрасываем флаг running, так как цикл завершается с ошибкой
                self.is_running = False
                return self.history.messages
            except APIError as e:
                logger.error(f"Agent '{self.agent_id}' API error: {e}")
                if api_response_callback:
                    api_response_callback(e.request, e.body, e)
                # Сбрасываем флаг running, так как цикл завершается с ошибкой
                self.is_running = False
                return self.history.messages

            if api_response_callback:
                api_response_callback(
                    raw_response.http_response.request, raw_response.http_response, None
                )

            # Ещё раз проверяем флаг прерывания после получения ответа
            if self.is_interrupted:
                logger.debug(
                    f"Agent '{self.agent_id}' detected interrupt flag after API response, breaking run loop"
                )
                # Сбрасываем флаг, но не запускаем новый цикл
                self.is_interrupted = False
                break

            response = raw_response.parse()

            # Process the response
            response_params = response_to_params(response)

            # TOOLS DEBUG: Log information about response blocks and tool calls
            try:
                block_types = [
                    block.get("type", "unknown") for block in response_params
                ]
                logger.debug(
                    f"TOOLS DEBUG: Agent '{self.agent_id}' received blocks of types: {block_types}"
                )

                # Check for tool_use blocks and log details
                for block in response_params:
                    if block.get("type") == "tool_use":
                        logger.debug(
                            f"TOOLS DEBUG: Agent '{self.agent_id}' received tool_use for tool: '{block.get('name')}' with input: {block.get('input')}"
                        )
            except Exception as e:
                logger.debug(f"TOOLS DEBUG ERROR: {str(e)}")

            self.history.append(
                {
                    "role": "assistant",
                    "content": response_params,
                }
            )
            logger.debug(f"Agent '{self.agent_id}' added assistant response to history")

            # Add assistant response to global history tree if available
            if self.history_tree:
                # Собираем все блоки thinking в один текст
                thinking_blocks = [
                    block
                    for block in response_params
                    if block.get("type") == "thinking"
                ]
                # Фильтруем обычные блоки контента (без thinking)
                non_thinking_blocks = [
                    block
                    for block in response_params
                    if block.get("type") != "thinking"
                ]

                # Формируем единый текст thinking, если есть thinking блоки
                combined_thinking = None
                if thinking_blocks:
                    # Объединяем все блоки thinking в один текст с разделителями
                    thinking_texts = []
                    for block in thinking_blocks:
                        thinking_content = block.get("thinking", "")
                        if thinking_content and thinking_content.strip():
                            thinking_texts.append(thinking_content)

                    if thinking_texts:
                        combined_thinking = "\n\n---\n\n".join(thinking_texts)
                        logger.debug(
                            f"Agent '{self.agent_id}' combined {len(thinking_texts)} thinking blocks"
                        )

                # Добавляем сообщение ассистента с объединенным thinking контентом
                if non_thinking_blocks:
                    self.history_tree.add_assistant_message(
                        agent_id=self.agent_id,
                        content=cast(List[BetaContentBlockParam], non_thinking_blocks),
                        thinking_content=combined_thinking,
                    )
                else:
                    # Если все блоки были thinking, добавляем заполнитель и thinking контент
                    self.history_tree.add_assistant_message(
                        agent_id=self.agent_id,
                        content=[{"type": "text", "text": "Processing..."}],
                        thinking_content=combined_thinking,
                    )
                logger.debug(
                    f"Agent '{self.agent_id}' added assistant response to history tree with thinking"
                )

            # Process tool calls
            tool_result_content: List[BetaToolResultBlockParam] = []
            for content_block in response_params:
                if output_callback:
                    output_callback(content_block)

                # Дополнительный отладочный вывод для всех блоков в ответе модели
                logger.debug(
                    f"RESPONSE DEBUG: Agent '{self.agent_id}' received block type: {content_block.get('type', 'unknown')}"
                )

                # Ещё раз проверяем флаг прерывания перед выполнением инструмента
                if self.is_interrupted:
                    logger.debug(
                        f"Agent '{self.agent_id}' detected interrupt flag before tool execution, breaking run loop"
                    )
                    # Добавляем синтетический результат для инструмента, чтобы не нарушать формат беседы
                    if content_block["type"] == "tool_use":
                        logger.debug(
                            f"Agent '{self.agent_id}' adding synthetic result for interrupted tool: {content_block['name']}"
                        )
                        interrupted_result = ToolResult(
                            output="",
                            error="human interrupted tool execution",
                            base64_image="",
                            system="Operation was interrupted by user message",
                        )
                        tool_result_block = make_api_tool_result(
                            interrupted_result, content_block["id"]
                        )
                        tool_result_content.append(tool_result_block)
                        if tool_output_callback:
                            tool_output_callback(
                                interrupted_result, content_block["id"]
                            )
                    continue

                if content_block["type"] == "tool_use":
                    logger.debug(
                        f"Agent '{self.agent_id}' executing tool: {content_block['name']}"
                    )

                    # Проверяем, не является ли это инструментом return_to_manager
                    if content_block["name"] == "return_to_manager":
                        logger.debug(
                            f"Agent '{self.agent_id}' detected return_to_manager tool call, will exit early after execution"
                        )

                        # Дополнительный отладочный вывод для инструментов
                        try:
                            logger.debug(
                                f"TOOL USE DEBUG: {content_block['name']} with input: {content_block.get('input', {})}"
                            )
                        except Exception as e:
                            logger.debug(f"TOOL USE DEBUG ERROR: {str(e)}")

                        # Add tool call to history tree if available
                        if self.history_tree:
                            self.history_tree.add_tool_call(
                                agent_id=self.agent_id,
                                tool_call=cast(BetaToolUseBlockParam, content_block),
                            )

                        # Execute the tool
                        result = await tool_collection.run(
                            name=content_block["name"],
                            tool_input=cast(dict[str, Any], content_block["input"]),
                        )

                        # Add tool result to history tree if available
                        if self.history_tree:
                            # Convert tool result to API format for history tree
                            api_result = make_api_tool_result(
                                result, content_block["id"]
                            )
                            self.history_tree.add_tool_result(
                                agent_id=self.agent_id, tool_result=api_result
                            )

                        # Add result to response
                        tool_result_block = make_api_tool_result(
                            result, content_block["id"]
                        )
                        tool_result_content.append(tool_result_block)

                        if tool_output_callback:
                            tool_output_callback(result, content_block["id"])

                        # Add tool results to history before exiting early
                        self.history.append(
                            {"content": tool_result_content, "role": "user"}
                        )
                        logger.debug(
                            f"Agent '{self.agent_id}' added tool results to history and exiting due to return_to_manager"
                        )

                        # Reset running flag
                        self.is_running = False

                        # Return early from the run method to return control to the caller
                        return self.history.messages

                    # Если это инструмент agent для создания специалиста, передадим tool_use_id, чтобы связать
                    # будущий вызов return_to_manager с этим вызовом agent
                    calling_tool_use_id = ""
                    if content_block["name"] == "agent":
                        calling_tool_use_id = content_block.get("id", "")
                        if not calling_tool_use_id:
                            error_msg = f"Agent '{self.agent_id}': Cannot execute agent tool without valid tool_use_id"
                            logger.error(error_msg)
                            # Просто выбрасываем исключение - самый простой способ обработки ошибки
                            raise ValueError(error_msg)

                        logger.debug(
                            f"Agent '{self.agent_id}' executing agent tool to create specialist, setting calling_tool_use_id: {calling_tool_use_id}"
                        )
                        # Добавляем более подробный отладочный вывод
                        logger.info(
                            f"CRITICAL DEBUG: Agent '{self.agent_id}' passing calling_tool_use_id={calling_tool_use_id} to create specialist"
                        )

                    # Дополнительный отладочный вывод для инструментов
                    try:
                        logger.debug(
                            f"TOOL USE DEBUG: {content_block['name']} with input: {content_block.get('input', {})}"
                        )
                    except Exception as e:
                        logger.debug(f"TOOL USE DEBUG ERROR: {str(e)}")

                    # Add tool call to history tree if available
                    if self.history_tree:
                        self.history_tree.add_tool_call(
                            agent_id=self.agent_id,
                            tool_call=cast(BetaToolUseBlockParam, content_block),
                        )
                        logger.debug(
                            f"Agent '{self.agent_id}' added tool call to history tree"
                        )

                    # Execute the tool
                    result = await tool_collection.run(
                        name=content_block["name"],
                        tool_input=cast(dict[str, Any], content_block["input"]),
                        calling_tool_use_id=calling_tool_use_id,
                    )

                    # Add tool result to history tree if available
                    if self.history_tree:
                        # Convert tool result to API format for history tree
                        api_result = make_api_tool_result(result, content_block["id"])
                        # Add metadata to help identify if this result contains an image
                        if result.base64_image:
                            self.history_tree.add_tool_result(
                                agent_id=self.agent_id,
                                tool_result=api_result,
                                extra_metadata={
                                    "has_image": True,
                                    "image_data": result.base64_image,
                                },
                            )
                        else:
                            self.history_tree.add_tool_result(
                                agent_id=self.agent_id, tool_result=api_result
                            )
                        logger.debug(
                            f"Agent '{self.agent_id}' added tool result to history tree"
                        )

                    # Add result to response
                    tool_result_block = make_api_tool_result(
                        result, content_block["id"]
                    )
                    tool_result_content.append(tool_result_block)

                    if tool_output_callback:
                        tool_output_callback(result, content_block["id"])

                    logger.debug(
                        f"Agent '{self.agent_id}' completed tool execution: {content_block['name']}"
                    )

            # If we were interrupted, break the loop after processing current tools
            if self.is_interrupted:
                logger.debug(
                    f"Agent '{self.agent_id}' detected interrupt flag after tool execution, breaking run loop"
                )
                # Add any tool results to history before breaking
                if tool_result_content:
                    self.history.append(
                        {"content": tool_result_content, "role": "user"}
                    )
                    logger.debug(
                        f"Agent '{self.agent_id}' added tool results to history before breaking"
                    )
                # Сбрасываем флаг, но не запускаем новый цикл
                self.is_interrupted = False
                break

            # If no tool calls, we're done
            if not tool_result_content:
                logger.debug(
                    f"Agent '{self.agent_id}' completed run with no tool calls"
                )
                # Сбрасываем флаг running, так как цикл завершается естественным образом
                self.is_running = False
                return self.history.messages

            # Add tool results to history
            self.history.append({"content": tool_result_content, "role": "user"})
            logger.debug(f"Agent '{self.agent_id}' added tool results to history")

            # Note: Tool results are already added to history_tree in the tool execution loop

        # Сбрасываем флаг running перед возвратом
        self.is_running = False
        logger.debug(f"Agent '{self.agent_id}' completed run due to interruption")
        # Return the updated message history
        return self.history.messages

    def _ensure_history_consistency(self) -> bool:
        """Ensure the history is in a consistent state before making API calls.

        This method checks for and fixes various consistency issues in the message history:
        1. Unclosed tool calls (tool_use without a corresponding tool_result)
        2. Mismatched tool_use/tool_result IDs
        3. Other potential issues that could cause API errors

        Returns:
            bool: True if any fixes were applied, False otherwise
        """
        logger.debug(
            f"Agent '{self.agent_id}' ensuring history consistency - handling any interrupted tools"
        )
        fixed_any = False

        # Проверка всей истории на несогласованность
        logger.debug(f"Agent '{self.agent_id}' checking history for unclosed tools")

        # Детально логируем историю сообщений для диагностики
        logger.debug(
            f"Agent '{self.agent_id}' total message history length: {len(self.history.messages)}"
        )

        # Шаг 1: Проверка сообщений с tool_use, за которыми не следует tool_result
        i = 0
        while i < len(self.history.messages) - 1:
            current_msg = self.history.messages[i]
            next_msg = self.history.messages[i + 1]

            # Если текущее сообщение от ассистента
            if current_msg.get("role") == "assistant":
                # Собираем tool_use ID из текущего сообщения
                tool_use_ids = []
                tool_use_names = {}  # tool_id -> tool_name

                for content_item in current_msg.get("content", []):
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "tool_use"
                    ):
                        tool_id = content_item.get("id", "")
                        tool_name = content_item.get("name", "unknown")
                        if tool_id:
                            tool_use_ids.append(tool_id)
                            tool_use_names[tool_id] = tool_name
                            logger.debug(
                                f"Agent '{self.agent_id}' found tool_use ID: {tool_id} for tool: {tool_name} at message index {i}"
                            )

                # Если есть tool_use, проверяем следующее сообщение
                if tool_use_ids:
                    # Следующее сообщение должно быть от пользователя
                    if next_msg.get("role") != "user":
                        # Если нет, создаем синтетическое сообщение пользователя
                        logger.debug(
                            f"Agent '{self.agent_id}' found message with tool_use but next message is not from user, inserting synthetic tool_result"
                        )

                        # Создаем результаты для всех tool_use
                        tool_result_content = []
                        for tool_id in tool_use_ids:
                            tool_name = tool_use_names.get(tool_id, "unknown")
                            logger.debug(
                                f"Creating synthetic result for tool_id {tool_id} (tool: {tool_name})"
                            )
                            tool_result = {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "human interrupted tool execution from previous conversation",
                                    }
                                ],
                                "is_error": True,
                            }
                            tool_result_content.append(tool_result)

                            # Также добавляем в дерево истории, если доступно
                            if self.history_tree:
                                self.history_tree.add_tool_result(
                                    agent_id="user",
                                    tool_result=ToolResult(
                                        output="",
                                        error=f"human interrupted tool execution (tool: {tool_name})",
                                        base64_image="",
                                        system="This tool was interrupted by a user message and never completed",
                                    ),
                                )

                        # Вставляем синтетическое сообщение пользователя сразу после текущего сообщения
                        self.history.messages.insert(
                            i + 1, {"role": "user", "content": tool_result_content}
                        )
                        logger.debug(
                            f"Agent '{self.agent_id}' inserted synthetic tool_result message after message {i}"
                        )
                        fixed_any = True

                        # Увеличиваем i на 1, так как мы вставили новое сообщение
                        i += 1
                    else:
                        # Если следующее сообщение от пользователя, проверяем, содержит ли оно все tool_result
                        result_ids = set()
                        for content_item in next_msg.get("content", []):
                            if (
                                isinstance(content_item, dict)
                                and content_item.get("type") == "tool_result"
                            ):
                                result_id = content_item.get("tool_use_id", "")
                                if result_id:
                                    result_ids.add(result_id)
                                    logger.debug(
                                        f"Agent '{self.agent_id}' found tool_result for ID: {result_id} at message index {i+1}"
                                    )

                        # Находим tool_use, для которых нет tool_result
                        missing_ids = [
                            id for id in tool_use_ids if id not in result_ids
                        ]

                        if missing_ids:
                            logger.debug(
                                f"Agent '{self.agent_id}' found {len(missing_ids)} tool_use without matching tool_result in next message: {missing_ids}"
                            )

                            # Создаем синтетические tool_result для недостающих инструментов
                            new_tool_results = []
                            for tool_id in missing_ids:
                                tool_name = tool_use_names.get(tool_id, "unknown")
                                logger.debug(
                                    f"Creating synthetic result for tool_id {tool_id} (tool: {tool_name})"
                                )

                                tool_result = {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "human interrupted tool execution",
                                        }
                                    ],
                                    "is_error": True,
                                }
                                new_tool_results.append(tool_result)

                                # Также добавляем в дерево истории, если доступно
                                if self.history_tree:
                                    self.history_tree.add_tool_result(
                                        agent_id="user",
                                        tool_result=ToolResult(
                                            output="",
                                            error=f"human interrupted tool execution (tool: {tool_name})",
                                            base64_image="",
                                            system="This tool was interrupted by a user message and never completed",
                                        ),
                                    )

                            # Добавляем новые tool_result в начало содержимого следующего сообщения
                            next_content = list(next_msg.get("content", []))
                            next_msg["content"] = new_tool_results + next_content
                            logger.debug(
                                f"Agent '{self.agent_id}' added {len(new_tool_results)} synthetic tool_result to existing message at index {i+1}"
                            )
                            fixed_any = True

            # Переходим к следующему сообщению
            i += 1

        # Шаг 2: Проверка последнего сообщения, если оно от ассистента и содержит tool_use
        if (
            self.history.messages
            and self.history.messages[-1].get("role") == "assistant"
        ):
            last_msg = self.history.messages[-1]
            tool_use_ids = []
            tool_use_names = {}

            for content_item in last_msg.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "tool_use"
                ):
                    tool_id = content_item.get("id", "")
                    tool_name = content_item.get("name", "unknown")
                    if tool_id:
                        tool_use_ids.append(tool_id)
                        tool_use_names[tool_id] = tool_name
                        logger.debug(
                            f"Agent '{self.agent_id}' found tool_use ID: {tool_id} in last message"
                        )

            if tool_use_ids:
                logger.debug(
                    f"Agent '{self.agent_id}' found {len(tool_use_ids)} unclosed tool_use in last message"
                )

                # Создаем синтетические результаты для всех инструментов
                tool_result_content = []
                for tool_id in tool_use_ids:
                    tool_name = tool_use_names.get(tool_id, "unknown")
                    logger.debug(
                        f"Creating synthetic result for tool_id {tool_id} (tool: {tool_name}) in last message"
                    )

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {
                                "type": "text",
                                "text": "human stopped or interrupted tool execution",
                            }
                        ],
                        "is_error": True,
                    }
                    tool_result_content.append(tool_result)

                    # Также добавляем в дерево истории, если доступно
                    if self.history_tree:
                        self.history_tree.add_tool_result(
                            agent_id="user",
                            tool_result=ToolResult(
                                output="(user stopped or interrupted and wrote the following)",
                                error="human stopped or interrupted tool execution",
                                base64_image="",
                                system="Tool execution was interrupted by user",
                            ),
                        )

                # Добавляем синтетическое сообщение пользователя
                self.history.append({"role": "user", "content": tool_result_content})
                logger.debug(
                    f"Agent '{self.agent_id}' appended synthetic tool_result message for last assistant message"
                )
                fixed_any = True

        # Шаг 3: Проверка на дублирующиеся tool_result с одинаковым tool_use_id
        # Эта проблема возникает, когда несколько tool_result ссылаются на один и тот же tool_use_id
        i = 0
        while i < len(self.history.messages):
            msg = self.history.messages[i]
            if msg.get("role") == "user":
                # Собираем все tool_result и группируем их по tool_use_id
                tool_results_by_id = {}
                if "content" in msg and isinstance(msg["content"], list):
                    for j, content_item in enumerate(msg["content"]):
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "tool_result"
                        ):
                            tool_use_id = content_item.get("tool_use_id", "")
                            if tool_use_id:
                                if tool_use_id not in tool_results_by_id:
                                    tool_results_by_id[tool_use_id] = []
                                tool_results_by_id[tool_use_id].append(
                                    (j, content_item)
                                )

                # Проверяем на дубликаты
                has_duplicates = False
                for tool_use_id, results in tool_results_by_id.items():
                    if len(results) > 1:
                        logger.warning(
                            f"Agent '{self.agent_id}' found {len(results)} duplicate tool_result for ID: {tool_use_id}"
                        )
                        has_duplicates = True

                # Если обнаружены дубликаты, исправляем их, оставляя только первый tool_result для каждого tool_use_id
                if has_duplicates:
                    new_content = []
                    seen_tool_use_ids = set()

                    for content_item in msg.get("content", []):
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "tool_result"
                        ):
                            tool_use_id = content_item.get("tool_use_id", "")
                            if tool_use_id:
                                if tool_use_id not in seen_tool_use_ids:
                                    new_content.append(content_item)
                                    seen_tool_use_ids.add(tool_use_id)
                                    logger.debug(
                                        f"Agent '{self.agent_id}' keeping first tool_result for ID: {tool_use_id}"
                                    )
                                else:
                                    logger.debug(
                                        f"Agent '{self.agent_id}' removing duplicate tool_result for ID: {tool_use_id}"
                                    )
                            else:
                                new_content.append(content_item)
                        else:
                            new_content.append(content_item)

                    msg["content"] = new_content
                    logger.debug(
                        f"Agent '{self.agent_id}' removed duplicate tool_result entries at message index {i}"
                    )
                    fixed_any = True

            i += 1

        return fixed_any
