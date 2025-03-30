"""
Agent implementation for Computer Use Demo.
"""

from typing import Any, Callable, List, Optional, cast

import httpx
from anthropic import APIError, APIResponseValidationError, APIStatusError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from computer_use_demo.loop import (
    PROMPT_CACHING_BETA_FLAG,
    SYSTEM_PROMPT,
    APIProvider,
    _inject_prompt_caching,
    _make_api_tool_result,
    _response_to_params,
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
                _inject_prompt_caching(self.history.messages)
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
            response_params = _response_to_params(response)
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
                        _make_api_tool_result(result, content_block["id"])
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
