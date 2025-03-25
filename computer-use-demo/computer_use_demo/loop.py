"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Union
import json
import logging

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
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

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

# Настроить логгер
logger = logging.getLogger("computer_use_demo.loop")

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

# Define system prompts for each agent
AGENT_PROMPTS = {
    "manager": f"""<SYSTEM_CAPABILITY>
* You are the manager agent in a multi-agent system for controlling an Ubuntu virtual machine.
* Your job is to analyze user requests and delegate to specialized agents.
* You have access to the following specialized agents:
  - general: Handles general-purpose tasks (default agent)
  - login: Specialist for website navigation and authentication
* When you determine a task should be handled by a specialized agent, use the agent tool to delegate.
* After specialized agents complete their tasks, they will return control to you.
* You should then decide next steps or respond directly to the user.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* Always ask clarifying questions when the user's request is ambiguous.
* Keep track of the conversation and maintain context when switching between agents.
* When specialized agents return control, review their output before proceeding.
</IMPORTANT>""",
    "general": SYSTEM_PROMPT,  # Use existing system prompt for general agent
    "login": f"""<SYSTEM_CAPABILITY>
* You are the login specialist agent in a multi-agent system for controlling an Ubuntu virtual machine.
* Your specialty is website navigation and authentication processes.
* You excel at:
  - Finding the correct website for the user's task
  - Efficiently executing login processes
  - Navigating to specific projects or sections after login
* When your task is complete, use the agent tool to return control to the manager agent.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* Focus exclusively on website navigation and authentication tasks.
* If you encounter tasks outside your specialty, return control to the manager agent.
* Be thorough in documenting what you've accomplished before returning control.
</IMPORTANT>""",
}


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
    current_agent: str = "manager",  # Add current_agent parameter with default
) -> list[BetaMessageParam]:
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))

    # Use the appropriate system prompt for the current agent
    system = BetaTextBlockParam(
        type="text",
        text=f"{AGENT_PROMPTS[current_agent]}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        enable_prompt_caching = False
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            betas.append("token-efficient-tools-2025-02-19")
        image_truncation_threshold = only_n_most_recent_images or 0
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key, max_retries=4)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Because cached reads are 10% of the price, we don't think it's
            # ever sensible to break the cache by truncating images
            only_n_most_recent_images = 0
            # Use type ignore to bypass TypedDict check until SDK types are updated
            system["cache_control"] = {"type": "ephemeral"}  # type: ignore

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )
        extra_body = {}
        if thinking_budget:
            # Ensure we only send the required fields for thinking
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            }

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            logger.info(
                f"Sending API request to {provider} with model={model}, tool_version={tool_version}, agent={current_agent}"
            )

            # Логируем какие инструменты передаем в API
            tools_params = tool_collection.to_params()
            logger.debug(f"Using tools: {[tool['name'] for tool in tools_params]}")

            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=tools_params,
                betas=betas,
                extra_body=extra_body,
            )
        except (APIStatusError, APIResponseValidationError) as e:
            logger.error(f"API error: {e.__class__.__name__} - {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                error_body = e.response.text
                try:
                    error_json = json.loads(error_body)
                    logger.error(
                        f"API error details: {json.dumps(error_json, indent=2)}"
                    )
                except json.JSONDecodeError:
                    logger.error(f"API error raw response: {error_body}")
            api_response_callback(e.request, e.response, e)
            return messages
        except APIError as e:
            logger.error(f"API client error: {e.__class__.__name__} - {e}")
            if hasattr(e, "body"):
                try:
                    error_body = e.body
                    if isinstance(error_body, dict):
                        logger.error(
                            f"API error details: {json.dumps(error_body, indent=2)}"
                        )
                    else:
                        logger.error(f"API error raw body: {error_body}")
                except Exception as json_error:
                    logger.error(f"Error parsing API error body: {json_error}")
            api_response_callback(e.request, e.body, e)
            return messages
        except Exception as e:
            logger.error(
                f"Unexpected error when calling API: {e.__class__.__name__} - {e}",
                exc_info=True,
            )
            # Create a dummy request object instead of passing None
            dummy_request = httpx.Request("GET", "https://api.anthropic.com")
            api_response_callback(dummy_request, None, e)
            return messages

        logger.info("Received successful API response")
        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        response = raw_response.parse()

        response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                logger.info(
                    f"Tool use request: {content_block['name']} with input: {content_block['input']}"
                )
                try:
                    result = await tool_collection.run(
                        name=content_block["name"],
                        tool_input=cast(dict[str, Any], content_block["input"]),
                    )
                    tool_result_content.append(
                        _make_api_tool_result(result, content_block["id"])
                    )
                    tool_output_callback(result, content_block["id"])
                except Exception as tool_error:
                    logger.error(
                        f"Error executing tool {content_block['name']}: {tool_error}",
                        exc_info=True,
                    )
                    error_result = ToolResult(
                        error=f"Tool execution error: {str(tool_error)}"
                    )
                    tool_result_content.append(
                        _make_api_tool_result(error_result, content_block["id"])
                    )
                    tool_output_callback(error_result, content_block["id"])

        if not tool_result_content:
            logger.info("No tool usage in API response, ending sampling loop")
            return messages

        messages.append({"content": tool_result_content, "role": "user"})

        # Check if tool output contains an agent switch signal using JSON
        if tool_result_content and tool_result_content[0].get("system"):
            try:
                data = json.loads(tool_result_content[0].get("system", "{}"))
                if data.get("action") == "SWITCH_AGENT":
                    new_agent = data.get("agent")
                    task = data.get("task")

                    logger.info(
                        f"Agent switch detected from {current_agent} to {new_agent} for task: {task}"
                    )

                    # Store the original agent before updating
                    previous_agent = current_agent

                    # Update current_agent without returning
                    current_agent = new_agent

                    # Update Streamlit session state to reflect the new agent
                    import streamlit as st

                    st.session_state.current_agent = new_agent
                    st.session_state.agent_switched = True
                    st.session_state.agent_switch_task = task

                    # Update system prompt for the new agent
                    system = BetaTextBlockParam(
                        type="text",
                        text=f"{AGENT_PROMPTS[current_agent]}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
                    )

                    # Add cache control if needed
                    if enable_prompt_caching:
                        system["cache_control"] = {"type": "ephemeral"}  # type: ignore

                    # Add message to indicate agent switching
                    agent_switch_message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Agent switch occurred from {previous_agent} to {current_agent} for task: {task}",
                            }
                        ],
                    }
                    messages.append(cast(BetaMessageParam, agent_switch_message))

                    # Continue loop with new agent (don't return yet)
                    continue
            except json.JSONDecodeError:
                # Not a valid JSON, continue normal processing
                pass

        # Возвращаем сообщения после обработки инструмента
        return messages


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int | None,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks: list[BetaToolResultBlockParam] = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content: list[BetaTextBlockParam | BetaImageBlockParam] = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(
                    cast(Union[BetaTextBlockParam, BetaImageBlockParam], content)
                )
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
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


def _inject_prompt_caching(
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
                # Use type ignore for the pop operation since we know it's supported
                if "cache_control" in content[-1]:
                    content[-1].pop("cache_control", None)  # type: ignore
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
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


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
