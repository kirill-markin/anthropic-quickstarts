"""
History management for agents.
"""

from typing import List, Optional, cast

from anthropic.types.beta import (
    BetaMessageParam,
    BetaToolResultBlockParam,
)

from .logging import get_logger

# Get logger for this module
logger = get_logger("computer_use_demo.agents.history")


class History:
    """Class for managing agent conversation history."""

    def __init__(self, messages: Optional[List[BetaMessageParam]] = None) -> None:
        """Initialize History with optional existing messages.

        Args:
            messages: Optional list of existing messages to initialize history with
        """
        self.messages: List[BetaMessageParam] = messages or []
        logger.debug(f"History initialized with {len(self.messages)} messages")

    def append(self, message: BetaMessageParam) -> None:
        """Append a message to the history.

        Args:
            message: The message to append
        """
        self.messages.append(message)
        logger.debug(
            f"Message added to history (role={message.get('role')}), now total: {len(self.messages)}"
        )

    def get_messages(self) -> List[BetaMessageParam]:
        """Get all messages in history.

        Returns:
            List of messages
        """
        logger.debug(f"Getting all {len(self.messages)} messages from history")
        return self.messages

    def optimize(self, images_to_keep: int, min_removal_threshold: int = 2) -> None:
        """Optimize history by removing old screenshots beyond a certain limit.

        Args:
            images_to_keep: Number of most recent images to keep
            min_removal_threshold: Minimum number of images to remove at once
        """
        logger.debug(
            f"Optimizing history to keep {images_to_keep} images (threshold={min_removal_threshold})"
        )
        if images_to_keep <= 0:
            logger.debug("No images to keep, skipping optimization")
            return

        tool_result_blocks = cast(
            List[BetaToolResultBlockParam],
            [
                item
                for message in self.messages
                for item in (
                    message["content"] if isinstance(message["content"], list) else []
                )
                if isinstance(item, dict) and item.get("type") == "tool_result"
            ],
        )

        logger.debug(f"Found {len(tool_result_blocks)} tool result blocks")

        total_images = sum(
            1
            for tool_result in tool_result_blocks
            for content in tool_result.get("content", [])
            if isinstance(content, dict) and content.get("type") == "image"
        )

        logger.debug(f"Found {total_images} total images")

        images_to_remove = total_images - images_to_keep
        # for better cache behavior, we want to remove in chunks
        images_to_remove -= images_to_remove % min_removal_threshold

        logger.debug(f"Will remove {images_to_remove} images")

        removed_count = 0
        for tool_result in tool_result_blocks:
            if isinstance(tool_result.get("content"), list):
                new_content = []
                for content in tool_result.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        if images_to_remove > 0:
                            images_to_remove -= 1
                            removed_count += 1
                            continue
                    new_content.append(content)
                tool_result["content"] = new_content

        logger.debug(f"Removed {removed_count} images from history")
