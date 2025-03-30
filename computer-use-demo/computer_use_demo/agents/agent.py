"""
Agent implementation for Computer Use Demo.
"""

import platform
from datetime import datetime
from enum import StrEnum
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

from computer_use_demo.tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

from .history import History
from .logging import get_logger

# Get logger for this module
logger = get_logger("computer_use_demo.agents.agent")

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


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
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """
    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                # Use type ignore to bypass TypedDict check until SDK types are updated
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                    {"type": "ephemeral"}
                )
            else:
                # Add type cast to avoid Pyright error on pop operation
                cast(Dict[str, Any], content[-1]).pop("cache_control", None)
                # we'll only every have one extra turn per loop
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
        system_prompt: str = SYSTEM_PROMPT,
        tool_version: ToolVersion = "computer_use_20250124",
    ) -> None:
        """Initialize Agent with required parameters.

        Args:
            agent_id: Unique identifier for this agent
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use
        """
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.tool_version = tool_version
        self.history = History()
        logger.debug(f"Agent '{agent_id}' initialized with tool_version={tool_version}")

    async def run(
        self,
        *,
        messages: List[BetaMessageParam],
        model: str,
        provider: APIProvider,
        system_prompt_suffix: str,
        output_callback: Callable[[BetaContentBlockParam], None],
        tool_output_callback: Callable[[ToolResult, str], None],
        api_response_callback: Callable[
            [httpx.Request, httpx.Response | object | None, Exception | None], None
        ],
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
            output_callback: Callback for model output
            tool_output_callback: Callback for tool output
            api_response_callback: Callback for API responses
            api_key: API key for authentication
            only_n_most_recent_images: Limit history to N most recent images
            max_tokens: Maximum tokens for model response
            thinking_budget: Budget for model thinking
            token_efficient_tools_beta: Whether to use token efficient tools beta

        Returns:
            Updated list of messages after processing
        """
        logger.debug(f"Agent '{self.agent_id}' run method called with model={model}")

        # Use the provided messages initially, but maintain our own history for later
        if not self.history.messages:
            self.history.messages = messages
            logger.debug(
                f"Agent '{self.agent_id}' initialized history with {len(messages)} messages"
            )
        else:
            # If we already have messages in history, just update with the latest
            for msg in messages:
                if msg not in self.history.messages:
                    self.history.append(msg)
            logger.debug(
                f"Agent '{self.agent_id}' updated history, now has {len(self.history.messages)} messages"
            )

        # Get tool collection for the specified version
        tool_group = TOOL_GROUPS_BY_VERSION[self.tool_version]
        tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
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
            logger.debug(f"Agent '{self.agent_id}' starting API call iteration")
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
                # Use type ignore to bypass TypedDict check until SDK types are updated
                system["cache_control"] = {"type": "ephemeral"}  # type: ignore
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
                api_response_callback(e.request, e.response, e)
                return self.history.messages
            except APIError as e:
                logger.error(f"Agent '{self.agent_id}' API error: {e}")
                api_response_callback(e.request, e.body, e)
                return self.history.messages

            api_response_callback(
                raw_response.http_response.request, raw_response.http_response, None
            )

            response = raw_response.parse()

            # Process the response
            response_params = response_to_params(response)
            self.history.append(
                {
                    "role": "assistant",
                    "content": response_params,
                }
            )
            logger.debug(f"Agent '{self.agent_id}' added assistant response to history")

            # Process tool calls
            tool_result_content: List[BetaToolResultBlockParam] = []
            for content_block in response_params:
                output_callback(content_block)
                if content_block["type"] == "tool_use":
                    logger.debug(
                        f"Agent '{self.agent_id}' executing tool: {content_block['name']}"
                    )
                    result = await tool_collection.run(
                        name=content_block["name"],
                        tool_input=cast(dict[str, Any], content_block["input"]),
                    )
                    tool_result_content.append(
                        make_api_tool_result(result, content_block["id"])
                    )
                    tool_output_callback(result, content_block["id"])
                    logger.debug(
                        f"Agent '{self.agent_id}' completed tool execution: {content_block['name']}"
                    )

            # If no tool calls, we're done
            if not tool_result_content:
                logger.debug(
                    f"Agent '{self.agent_id}' completed run with no tool calls"
                )
                return self.history.messages

            # Add tool results to history
            self.history.append({"content": tool_result_content, "role": "user"})
            logger.debug(f"Agent '{self.agent_id}' added tool results to history")

        # Return the updated message history
        return self.history.messages
