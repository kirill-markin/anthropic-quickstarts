"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Dict, cast, get_args

import streamlit as st
from anthropic import RateLimitError

from computer_use_demo.agents import ManagerAgent, get_logger, setup_logging
from computer_use_demo.agents.agent import APIProvider
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.tools import ToolVersion

# Logging setup
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
streamlit_log_level = os.environ.get("STREAMLIT_LOGLEVEL", "INFO").upper()

# If STREAMLIT_LOGLEVEL is set to debug, also set LOG_LEVEL to DEBUG
if streamlit_log_level == "DEBUG":
    log_level = "DEBUG"

# Configure logging with the new function
setup_logging(log_level=log_level)

# Get logger for this module
logger = get_logger(__name__)
logger.debug("Streamlit application starting")

PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}

# Constants for handling user interruptions
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"


@dataclass(kw_only=True, frozen=True)
class ModelConfig:
    tool_version: ToolVersion
    max_output_tokens: int
    default_output_tokens: int
    has_thinking: bool = False


SONNET_3_5_NEW = ModelConfig(
    tool_version="computer_use_20241022",
    max_output_tokens=1024 * 8,
    default_output_tokens=1024 * 4,
)

SONNET_3_7 = ModelConfig(
    tool_version="computer_use_20250124",
    max_output_tokens=128_000,
    default_output_tokens=1024 * 16,
    has_thinking=True,
)

MODEL_TO_MODEL_CONF: dict[str, ModelConfig] = {
    "claude-3-7-sonnet-20250219": SONNET_3_7,
}

CONFIG_DIR = os.path.expanduser("~/.anthropic")
API_KEY_FILE = os.path.join(CONFIG_DIR, "api_key")
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }

    /* Styles for different agent types */
    /* Add a left border for specialist messages to distinguish them */
    [data-testid="stChatMessage"] .stCaption {
        color: #3366CC;
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Style system messages */
    [data-testid="stChatMessage"][data-chatmessageauthor="system"] {
        padding-left: 10px;
        padding-right: 10px;
    }

    /* Style manager agent messages */
    [data-testid="stChatMessage"][data-chatmessageauthor="manager"] {
        background-color: #f0f7ff;
        border-left: 3px solid #4a86e8;
    }

    /* Style specialist agent messages with different colors */
    [data-testid="stChatMessage"][data-chatmessageauthor="general"] {
        background-color: #f3f3f3;
        border-left: 3px solid #666666;
    }

    [data-testid="stChatMessage"][data-chatmessageauthor^="general_"] {
        background-color: #f3f3f3;
        border-left: 3px solid #666666;
    }

    [data-testid="stChatMessage"][data-chatmessageauthor="web_auth"] {
        background-color: #f5f0ff;
        border-left: 3px solid #9c27b0;
    }

    [data-testid="stChatMessage"][data-chatmessageauthor^="web_auth_"] {
        background-color: #f5f0ff;
        border-left: 3px solid #9c27b0;
    }

    [data-testid="stChatMessage"][data-chatmessageauthor="lovable_bot"] {
        background-color: #fff8e1;
        border-left: 3px solid #ffc107;
    }

    [data-testid="stChatMessage"][data-chatmessageauthor^="lovable_bot_"] {
        background-color: #fff8e1;
        border-left: 3px solid #ffc107;
    }

    /* Make tabs more visible in agent history display */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-left: 10px;
        padding-right: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 2px solid #4a86e8;
    }

    /* Style specialist session blocks */
    .specialist-session {
        border-left: 3px solid #4a86e8;
        padding-left: 10px;
        margin-left: 15px;
        margin-bottom: 10px;
    }

    .session-header {
        font-weight: bold;
        background-color: #f0f7ff;
        padding: 5px 10px;
        border-radius: 4px 4px 0 0;
        margin-bottom: 5px;
        border-left: 3px solid #4a86e8;
    }

    .session-footer {
        font-style: italic;
        color: #666;
        padding: 5px 10px;
        background-color: #f9f9f9;
        border-radius: 0 0 4px 4px;
        margin-top: 5px;
        border-left: 3px solid #4a86e8;
    }

    .depth-0 {
        margin-left: 0px;
    }

    .depth-1 {
        margin-left: 20px;
    }

    .depth-2 {
        margin-left: 40px;
    }

    .depth-3 {
        margin-left: 60px;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


def setup_state():
    """Setup Streamlit session state with necessary initial values."""
    # API Authentication
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
    if "provider" not in st.session_state:
        st.session_state.provider = (
            os.environ.get("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        )
    if "provider_radio" not in st.session_state:
        st.session_state.provider_radio = st.session_state.provider
    if "model" not in st.session_state:
        _reset_model()
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False

    # History storage
    if "history_tree" not in st.session_state:
        st.session_state.history_tree = HistoryTree()
        logger.debug("Initialized global history tree")

    # UI placeholders for real-time updates
    if "messages_placeholder" not in st.session_state:
        st.session_state.messages_placeholder = None
    if "update_counter" not in st.session_state:
        st.session_state.update_counter = 0

    # Temporary state for sampling loop
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False

    # User preferences
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False

    # Check if thinking should be enabled via environment variable
    if "thinking" not in st.session_state:
        st.session_state.thinking = os.environ.get("ENABLE_THINKING", "").lower() in [
            "true",
            "1",
            "yes",
        ]

    # Set default agent settings
    if "agent_settings" not in st.session_state:
        st.session_state.agent_settings = {
            "only_n_most_recent_images": 3,
            "output_tokens": 4096,
            "thinking_enabled": st.session_state.thinking,
            "thinking_budget": 2048,
            "hide_images": False,
            "token_efficient_tools_beta": True,
            "model": "",  # Will be set at runtime
            "provider": cast(APIProvider, st.session_state.provider),
            "api_key": st.session_state.api_key,
        }

    # Initialize the manager agent (single entry point)
    if "manager_agent" not in st.session_state:
        st.session_state.manager_agent = None


def _reset_model():
    """Reset model settings when provider changes."""
    st.session_state.model = PROVIDER_TO_DEFAULT_MODEL_NAME[
        cast(APIProvider, st.session_state.provider)
    ]
    _reset_model_conf()


def _reset_model_conf():
    """Reset model configuration based on selected model."""
    model_conf = (
        SONNET_3_7
        if "3-7" in st.session_state.model
        else MODEL_TO_MODEL_CONF.get(st.session_state.model, SONNET_3_5_NEW)
    )
    st.session_state.tool_version = model_conf.tool_version
    st.session_state.has_thinking = model_conf.has_thinking
    st.session_state.output_tokens = model_conf.default_output_tokens
    st.session_state.max_output_tokens = model_conf.max_output_tokens
    st.session_state.thinking_budget = int(model_conf.default_output_tokens / 2)

    # Update agent settings with the new model info
    if "agent_settings" in st.session_state:
        st.session_state.agent_settings["output_tokens"] = (
            model_conf.default_output_tokens
        )
        st.session_state.agent_settings["thinking_budget"] = int(
            model_conf.default_output_tokens / 2
        )
        st.session_state.agent_settings["model"] = st.session_state.model

    # Update manager agent with new tool version if it exists
    if hasattr(st.session_state, "manager_agent") and st.session_state.manager_agent:
        st.session_state.manager_agent.tool_version = st.session_state.tool_version
        logger.debug(
            f"Updated tool version for manager agent to {st.session_state.tool_version}"
        )


def update_ui():
    """Update the UI with the latest history tree data."""
    try:
        if st.session_state.messages_placeholder is not None:
            with st.session_state.messages_placeholder.container():
                # Increase update counter to force a refresh
                st.session_state.update_counter += 1
                # Clear the container and redraw all messages
                st.empty()

                # Get the conversation history
                history_data = st.session_state.history_tree.get_recursive_conversation(
                    include_tool_calls=True
                )
                # Render each node
                for item in history_data:
                    _render_node(item)
    except Exception as e:
        # Log the error but don't crash the app
        logger.error(f"Error updating UI in real-time: {str(e)}")
        # Don't throw the exception further - we want the app to continue running


async def main():
    """Render loop for streamlit"""
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Claude Computer Use Demo")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:

        def _reset_api_provider():
            if st.session_state.provider_radio != st.session_state.provider:
                _reset_model()
                st.session_state.provider = st.session_state.provider_radio
                st.session_state.auth_validated = False

                # Update agent settings
                if "agent_settings" in st.session_state:
                    st.session_state.agent_settings["provider"] = cast(
                        APIProvider, st.session_state.provider
                    )

        provider_options = [option.value for option in APIProvider]
        st.radio(
            "API Provider",
            options=provider_options,
            key="provider_radio",
            format_func=lambda x: x.title(),
            on_change=_reset_api_provider,
        )

        st.text_input("Model", key="model", on_change=_reset_model_conf)

        # API key input
        if st.session_state.provider == APIProvider.ANTHROPIC:
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                key="api_key",
                on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
            )

            # Update agent settings
            if "agent_settings" in st.session_state:
                st.session_state.agent_settings["api_key"] = api_key

        # Display current settings
        st.divider()
        st.subheader("Current Agent Settings")

        settings = st.session_state.agent_settings
        st.text(f"🖼️ Max Recent Images: {settings['only_n_most_recent_images']}")
        st.text(f"🔢 Output Tokens: {settings['output_tokens']}")
        st.text(f"🤔 Thinking Enabled: {settings['thinking_enabled']}")

        if settings["thinking_enabled"]:
            st.text(f"💭 Thinking Budget: {settings['thinking_budget']}")

        st.text(f"👁️ Hide Images: {settings['hide_images']}")
        st.text(f"🪙 Token Efficient Tools: {settings['token_efficient_tools_beta']}")

        versions = get_args(ToolVersion)
        st.radio(
            "Tool Versions",
            key="tool_versions",
            options=versions,
            index=versions.index(st.session_state.tool_version),
            on_change=lambda: _update_agent_tool_version(),
        )

        if st.button("Reset", type="primary"):
            with st.spinner("Resetting..."):
                st.session_state.clear()
                # Clear any temporary files in /tmp directory
                subprocess.run("rm -f /tmp/screen*.png", shell=True)  # noqa: ASYNC221
                setup_state()
                subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                await asyncio.sleep(1)
                subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

        if validate_auth(
            cast(APIProvider, st.session_state.provider), st.session_state.api_key
        ):
            st.session_state.auth_validated = True

    if not st.session_state.auth_validated:
        st.error("Please set your Anthropic API key to get started.")
        return

    # Create a placeholder for messages that can be updated in real-time
    if st.session_state.messages_placeholder is None:
        st.session_state.messages_placeholder = st.empty()

    # Setup the callback for real-time updates
    st.session_state.history_tree.set_update_callback(update_ui)

    # Initial render of conversation history
    update_ui()

    new_message = st.chat_input("Message Claude")

    # Initialize the manager agent if it doesn't exist
    if not st.session_state.manager_agent:
        # Create a manager agent with manager_only tools and history tree
        st.session_state.manager_agent = ManagerAgent(
            history_tree=st.session_state.history_tree,
            agent_id="manager",
            tool_version="manager_only_20250124",
        )
        logger.debug("Created manager agent with history tree")

        # Update settings on the manager
        st.session_state.manager_agent.settings.update(
            {
                "model": st.session_state.model,
                "provider": cast(APIProvider, st.session_state.provider),
                "api_key": st.session_state.api_key,
                "output_tokens": st.session_state.agent_settings["output_tokens"],
                "thinking_budget": st.session_state.agent_settings["thinking_budget"],
                "thinking_enabled": st.session_state.agent_settings["thinking_enabled"],
                "only_n_most_recent_images": st.session_state.agent_settings[
                    "only_n_most_recent_images"
                ],
                "token_efficient_tools_beta": st.session_state.agent_settings[
                    "token_efficient_tools_beta"
                ],
            }
        )
        logger.debug("Updated manager agent settings")

    # Process new user message
    if new_message:
        # Let the manager agent handle the message routing and tool interruption
        if st.session_state.manager_agent:
            with track_sampling_loop():
                try:
                    # This will handle tool interruption, adding message to history,
                    # and routing to the appropriate agent
                    await st.session_state.manager_agent.handle_user_message(
                        new_message
                    )

                    # Get the active agent after handling the message
                    active_agent = st.session_state.manager_agent.get_active_agent()
                    active_agent_id = st.session_state.manager_agent.active_agent_id

                    logger.debug(
                        f"Running active agent after message: {active_agent_id}"
                    )

                    # Запоминаем текущий ID активного агента перед запуском
                    prev_active_agent_id = active_agent_id

                    # Run the active agent with the message
                    await active_agent.run(
                        messages=active_agent.history.messages,
                        model=st.session_state.model,
                        provider=cast(APIProvider, st.session_state.provider),
                        system_prompt_suffix=st.session_state.custom_system_prompt,
                        # Set callbacks to None - we use the history tree instead
                        output_callback=None,
                        tool_output_callback=None,
                        api_response_callback=None,
                        api_key=st.session_state.api_key,
                        # Use settings
                        only_n_most_recent_images=st.session_state.agent_settings[
                            "only_n_most_recent_images"
                        ],
                        max_tokens=st.session_state.agent_settings["output_tokens"],
                        thinking_budget=st.session_state.agent_settings[
                            "thinking_budget"
                        ]
                        if st.session_state.agent_settings["thinking_enabled"]
                        else None,
                        token_efficient_tools_beta=st.session_state.agent_settings[
                            "token_efficient_tools_beta"
                        ],
                    )

                    # Проверяем, изменился ли активный агент после выполнения
                    # Это будет происходить, если специалист вызвал return_to_manager
                    current_active_agent_id = (
                        st.session_state.manager_agent.active_agent_id
                    )

                    if current_active_agent_id != prev_active_agent_id:
                        logger.debug(
                            f"Active agent changed from {prev_active_agent_id} to {current_active_agent_id}, automatically running new active agent"
                        )

                        # Получаем нового активного агента (теперь это менеджер)
                        new_active_agent = (
                            st.session_state.manager_agent.get_active_agent()
                        )

                        # Запускаем менеджера автоматически
                        await new_active_agent.run(
                            messages=new_active_agent.history.messages,
                            model=st.session_state.model,
                            provider=cast(APIProvider, st.session_state.provider),
                            system_prompt_suffix=st.session_state.custom_system_prompt,
                            # Set callbacks to None - we use the history tree instead
                            output_callback=None,
                            tool_output_callback=None,
                            api_response_callback=None,
                            api_key=st.session_state.api_key,
                            # Use settings
                            only_n_most_recent_images=st.session_state.agent_settings[
                                "only_n_most_recent_images"
                            ],
                            max_tokens=st.session_state.agent_settings["output_tokens"],
                            thinking_budget=st.session_state.agent_settings[
                                "thinking_budget"
                            ]
                            if st.session_state.agent_settings["thinking_enabled"]
                            else None,
                            token_efficient_tools_beta=st.session_state.agent_settings[
                                "token_efficient_tools_beta"
                            ],
                        )
                        logger.debug(
                            "Automatically ran manager agent after return_to_manager"
                        )
                except RateLimitError as e:
                    st.error(
                        f"Claude API rate limit reached: {e.message if hasattr(e, 'message') else str(e)}"
                    )
                    _render_error(e)
                except Exception as e:
                    st.error(f"Error: {e}")
                    _render_error(e)

                    # Add error to history tree for visibility
                    st.session_state.history_tree.add_system_message(
                        agent_id="manager",
                        content=[
                            {"type": "text", "text": f"⚠️ **Error occurred**: {str(e)}"}
                        ],
                    )

                    # Update UI immediately
                    update_ui()

                    # Log detailed error information
                    logger.error(f"Error during agent run: {str(e)}")
                    # Check for HTTP response details in a safe way
                    try:
                        if hasattr(e, "response"):
                            response = e.response
                            if hasattr(response, "text"):
                                logger.error(f"API response: {response.text}")
                            elif hasattr(response, "json") and callable(response.json):
                                try:
                                    logger.error(
                                        f"API response JSON: {response.json()}"
                                    )
                                except Exception:
                                    pass
                    except Exception as log_error:
                        logger.debug(
                            f"Error while logging API response details: {log_error}"
                        )

                    logger.error(f"Error traceback: {traceback.format_exc()}")


@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False


def validate_auth(provider: APIProvider, api_key: str | None) -> bool:
    """Validate authentication credentials."""
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return False
    if provider == APIProvider.BEDROCK:
        try:
            import boto3

            if not boto3.Session().get_credentials():
                return False
        except (ImportError, Exception):
            return False
    if provider == APIProvider.VERTEX:
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError

            if not os.environ.get("CLOUD_ML_REGION"):
                return False
            try:
                google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            except DefaultCredentialsError:
                return False
        except (ImportError, Exception):
            return False

    # If all checks passed, return True
    return True


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = os.path.join(CONFIG_DIR, filename)
        if os.path.exists(file_path):
            with open(file_path) as f:
                data = f.read().strip()
            if data:
                return data
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        file_path = os.path.join(CONFIG_DIR, filename)
        with open(file_path, "w") as f:
            f.write(data)
        # Ensure only user can read/write the file
        os.chmod(file_path, 0o600)
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}")


def _render_error(error: Exception):
    """Render error information to the UI."""
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_node(node_data: Dict[str, Any]):
    """Render a history tree node in the UI.

    This function handles rendering different types of history tree nodes
    based on their node type, and applies appropriate styling.

    Args:
        node_data: Dictionary representation of a history tree node
    """
    node_type = node_data.get("type", "")
    agent_id = node_data.get("agent_id", "")
    depth = node_data.get("depth", 0)

    # Apply depth-based indentation
    if depth > 0:
        st.markdown(
            f"""<div class="depth-{min(depth, 3)}"></div>""", unsafe_allow_html=True
        )

    # Handle different node types
    if node_type == "user_message":
        with st.chat_message("user"):
            # Process content consistently with assistant_message
            for content_item in node_data.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "text"
                ):
                    st.markdown(content_item.get("text", ""))

    elif node_type == "assistant_message":
        # Use agent_id as the sender for display and styling
        with st.chat_message(agent_id):
            # Always add agent type caption to clearly identify the sender
            if agent_id == "manager":
                st.caption("Manager")
            elif agent_id != "user" and agent_id != "system":
                # Show specialist identifier (e.g., "general_1234567890")
                st.caption(f"Specialist: {agent_id}")

            # Сначала отобразим основной контент
            for content_item in node_data.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "text"
                ):
                    st.markdown(content_item.get("text", ""))

            # Проверяем, есть ли thinking в данном узле
            if node_data.get("has_thinking_content") and node_data.get("thinking"):
                # Отображаем thinking в отдельном expandable блоке
                with st.expander("Show Thinking"):
                    st.markdown(f"```\n{node_data.get('thinking', '')}\n```")

                # Добавляем индикатор, что сообщение содержит thinking
                st.caption("💭 This message includes thinking")

    elif node_type == "tool_call":
        tool_name = node_data.get("tool_name", "")
        tool_input = node_data.get("tool_input", {})

        with st.chat_message("tool"):
            # Add agent identification for tool calls
            st.caption(f"Tool call by: {agent_id}")
            # Format tool calls nicely
            st.code(f"Tool: {tool_name}\nInput: {tool_input}")

    elif node_type == "tool_result":
        with st.chat_message("tool"):
            # Add agent identification for tool results
            st.caption(f"Tool result for: {agent_id}")

            # Debug logging
            logger.debug(
                f"Tool result data: has_image={node_data.get('has_image', False)}, image_data exists={bool(node_data.get('image_data', ''))}, content format={type(node_data.get('content', ''))}"
            )

            # Handle different tool result formats
            image_displayed = False  # Flag to track if we've already shown an image

            if "output" in node_data:
                st.markdown(node_data.get("output", ""))
            elif "error" in node_data:
                st.error(node_data.get("error", ""))
            elif "content" in node_data:
                content = node_data.get("content", "")
                if isinstance(content, str):
                    st.markdown(content)
                else:
                    # Handle complex content structures
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                st.markdown(item.get("text", ""))
                            elif (
                                item.get("type") == "image"
                                and not st.session_state.hide_images
                            ):
                                # Direct display of image from content structure
                                image_source = item.get("source", {})
                                if image_source.get(
                                    "type"
                                ) == "base64" and image_source.get("data"):
                                    logger.debug(
                                        f"Found image in content: {image_source.get('type')}"
                                    )
                                    st.image(base64.b64decode(image_source["data"]))
                                    image_displayed = (
                                        True  # Mark that we've displayed an image
                                    )

            # Display image if present and not hidden and not already displayed
            if (
                not image_displayed
                and node_data.get("has_image", False)
                and not st.session_state.hide_images
                and node_data.get("image_data")
            ):
                logger.debug(
                    f"Displaying image from image_data field, length: {len(node_data.get('image_data', ''))}"
                )
                st.image(base64.b64decode(node_data.get("image_data", "")))

    elif node_type == "thinking":
        with st.chat_message(agent_id):
            st.caption(f"Thinking by: {agent_id}")
            st.markdown(f"[Thinking]\n\n{node_data.get('thinking', '')}")

    elif node_type == "specialist_session":
        # Display specialist session information as a collapsible section
        with st.expander(
            f"Specialist Session: {node_data.get('metadata', {}).get('specialist_id', 'unknown')}"
        ):
            st.markdown(
                f"**Task:** {node_data.get('delegate_info', {}).get('task', 'Unknown task')}"
            )
            if "delegate_info" in node_data and "context" in node_data["delegate_info"]:
                st.markdown(f"**Context:** {node_data['delegate_info']['context']}")

    elif node_type == "session_header":
        # Display a header for specialist session
        st.markdown(
            f"""<div class="session-header depth-{min(depth, 3)}">
            Specialist Session: {node_data.get('specialist_id', 'unknown')}
            <br/>Task: {node_data.get('task', 'Unknown task')}
        </div>""",
            unsafe_allow_html=True,
        )

    elif node_type == "session_footer":
        # Display a footer for specialist session
        st.markdown(
            f"""<div class="session-footer depth-{min(depth, 3)}">
            End of session with {node_data.get('specialist_id', 'unknown')}
        </div>""",
            unsafe_allow_html=True,
        )

    elif node_type == "system_message":
        with st.chat_message("system"):
            for content_item in node_data.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "text"
                ):
                    st.markdown(content_item.get("text", ""))

    # For other node types, just render a debug representation
    else:
        with st.chat_message("system"):
            st.text(f"Node type: {node_type}, Agent: {agent_id}")

    # Add spacing for better readability between nodes
    if depth == 0:
        st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)


def _update_agent_tool_version():
    """Update tool version for the manager agent."""
    if st.session_state.manager_agent:
        old_version = st.session_state.manager_agent.tool_version
        st.session_state.manager_agent.tool_version = st.session_state.tool_versions
        st.session_state.tool_version = st.session_state.tool_versions
        logger.debug(
            f"Updated agent tool_version from {old_version} to {st.session_state.tool_versions}"
        )


if __name__ == "__main__":
    asyncio.run(main())
