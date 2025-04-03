from dataclasses import dataclass
from typing import Literal

from computer_use_demo.agents.logging import get_logger
from computer_use_demo.interfaces import ToolVersion

from .agent_tool import AgentTool20250124, ReturnToManagerTool20250124
from .base import BaseAnthropicTool
from .bash import BashTool20241022, BashTool20250124
from .computer import (
    ComputerTool20241022,
    ComputerTool20250124,
)
from .edit import EditTool20241022, EditTool20250124

# Get logger for this module
logger = get_logger("computer_use_demo.tools.groups")

# Keep literal type for backward compatibility
ToolVersionLiteral = Literal[
    "computer_use_20250124",
    "computer_use_20241022",
    "manager_only_20250124",
    "specialist_only_20250124",
]
BetaFlag = Literal["computer-use-2024-10-22", "computer-use-2025-01-24"]


@dataclass(frozen=True, kw_only=True)
class ToolGroup:
    version: ToolVersion
    tools: list[type[BaseAnthropicTool]]
    beta_flag: BetaFlag | None = None


TOOL_GROUPS: list[ToolGroup] = [
    ToolGroup(
        version="computer_use_20241022",
        tools=[ComputerTool20241022, EditTool20241022, BashTool20241022],
        beta_flag="computer-use-2024-10-22",
    ),
    ToolGroup(
        version="computer_use_20250124",
        tools=[
            ComputerTool20250124,
            EditTool20250124,
            BashTool20250124,
        ],
        beta_flag="computer-use-2025-01-24",
    ),
    # Tool group for the manager agent with screenshot-only capability
    ToolGroup(
        version="manager_only_20250124",
        tools=[
            ComputerTool20250124,
            EditTool20250124,
            AgentTool20250124,
        ],
        beta_flag="computer-use-2025-01-24",
    ),
    # New tool group for specialist agents
    ToolGroup(
        version="specialist_only_20250124",
        tools=[
            ComputerTool20250124,
            EditTool20250124,
            BashTool20250124,
            ReturnToManagerTool20250124,
        ],
        beta_flag="computer-use-2025-01-24",
    ),
]

TOOL_GROUPS_BY_VERSION = {tool_group.version: tool_group for tool_group in TOOL_GROUPS}
