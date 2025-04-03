"""
Specialist type definitions for multi-agent system.

This module defines the available specialist types, their prompts, and descriptions.
It serves as a single source of truth for specialist configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

# Используем TYPE_CHECKING для избежания циклических импортов
if TYPE_CHECKING:
    pass


@dataclass
class SpecialistType:
    """Configuration for a specialist agent type."""

    id: str
    name: str
    description: str
    prompt_suffix: str
    # Default settings for this specialist
    only_n_most_recent_images: int = 3
    hide_images: bool = False
    token_efficient_tools_beta: bool = True
    output_tokens: int = 4096
    thinking_enabled: bool = True
    thinking_budget: int = 2048


# Base prompt part common to all specialists
SPECIALIST_PROMPT_BASE = """
<AGENT_ROLE>
You are a Specialist Agent in a multi-agent system. You receive specific tasks from a Manager Agent
and are responsible for executing them with your specialized tools and knowledge.

Your responsibilities are:
1. Focus exclusively on the task assigned by the Manager
2. Execute the task using your available tools
3. Report results clearly and concisely
4. Provide detailed information about what you've accomplished and any issues encountered

Remember that you should only work on the specific task assigned to you. Don't expand the scope
beyond what you were asked to do. If you encounter limitations or need additional information,
complete as much as you can and clearly report what you were unable to do.
</AGENT_ROLE>
"""

# Define all specialist types in one place
SPECIALIST_TYPES: Dict[str, SpecialistType] = {
    "web_auth": SpecialistType(
        id="web_auth",
        name="Web Authentication Specialist",
        description="Specialist for opening URLs and handling authentication processes in web applications. You MUST use the web browser to open URLs and handle authentication processes.",
        prompt_suffix="""
<SPECIALIZATION>
You specialize in opening URLs and handling authentication processes, including:
- Navigating to websites and opening specific URLs
- Filling in login forms and handling authentication flows
- Dealing with 2FA and security challenges when needed
- Maintaining secure login sessions
- Taking screenshots to verify successful logins
- Handling login errors and providing clear explanations
- Before switch to manager please make shure related apps are in full screen mode and all ready to use

You understand security best practices and know how to approach login processes systematically.
</SPECIALIZATION>
""",
        only_n_most_recent_images=5,  # Need more images for authentication flows
        hide_images=False,  # Screenshots are vital for auth verification
        token_efficient_tools_beta=True,
        output_tokens=4096,
        thinking_enabled=True,
        thinking_budget=2048,
    ),
    "lovable_bot": SpecialistType(
        id="lovable_bot",
        name="Lovable Development Specialist",
        description="Specialist for developing and configuring bots on the Lovable platform. You MUST use the Lovable platform to develop products.",
        prompt_suffix="""
<SPECIALIZATION>
You specialize in developing bots for the Lovable platform, including:
- Creating and configuring new Lovable bots
- Writing efficient and well-structured bot code
- Implementing conversation flows and logic
- Integrating APIs and external services with the bot
- Testing and debugging bot functionality
- Optimizing bot performance and user experience

Hints:
- Before interacting with Lovable platform, you should always check if we scrolled down and you see the bottom of the message list to make sure you see all the messages from the Lovable platform.
- After each sended message to Lovable platform, you should wait munimum 60 seconds before sending the next message.

You understand the Lovable platform architecture, coding patterns, and best practices for bot development.

Before any change you MUST to scroll Lovable chat to the bottom to make sure you see all the messages from the Lovable platform.
</SPECIALIZATION>
""",
        only_n_most_recent_images=2,  # Less focused on UI interactions
        hide_images=False,
        token_efficient_tools_beta=True,
        output_tokens=8192,  # Needs more tokens for code generation
        thinking_enabled=True,  # Enable thinking for code development
        thinking_budget=4096,
    ),
    "general": SpecialistType(
        id="general",
        name="General Computer Specialist",
        description="General-purpose specialist capable of handling a wide variety of computer tasks",
        prompt_suffix="""
<SPECIALIZATION>
You are a general-purpose specialist capable of handling a wide variety of computer tasks. You can:
- Browse websites and interact with web content
- Execute terminal commands and manage system resources
- Create and edit files of various types
- Perform integrated tasks that require multiple tool types
- Adapt to new requirements and tools

You excel at tasks that require flexibility and a broad knowledge base rather than deep specialization.
</SPECIALIZATION>
""",
        only_n_most_recent_images=3,  # Balanced approach
        hide_images=False,
        token_efficient_tools_beta=True,
        output_tokens=4096,
        thinking_enabled=True,
        thinking_budget=2048,
    ),
}


# Convenience function to get full prompt for a specialist type
def get_full_prompt(specialist_type_id: str) -> str:
    """Get the full prompt for a specialist type.

    Args:
        specialist_type_id: ID of the specialist type

    Returns:
        The full prompt (base + specialization)

    Raises:
        KeyError: If specialist_type_id is not valid
    """
    if specialist_type_id not in SPECIALIST_TYPES:
        raise KeyError(f"Unknown specialist type: {specialist_type_id}")

    return SPECIALIST_PROMPT_BASE + SPECIALIST_TYPES[specialist_type_id].prompt_suffix


def get_all_specialist_descriptions() -> Dict[str, str]:
    """Get all specialist descriptions.

    Returns:
        Dictionary mapping specialist IDs to descriptions
    """
    return {id: spec.description for id, spec in SPECIALIST_TYPES.items()}
