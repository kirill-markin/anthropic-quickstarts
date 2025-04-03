"""
Agents module for Computer Use Demo.
"""

from .agent import Agent
from .history import History
from .logging import get_logger, setup_logging
from .manager import ManagerAgent
from .specialist import SpecialistAgent
from .specialist_types import (
    SPECIALIST_TYPES,
    get_full_prompt,
)

__all__ = [
    "Agent",
    "get_logger",
    "setup_logging",
    "History",
    "ManagerAgent",
    "SpecialistAgent",
    "SPECIALIST_TYPES",
    "get_full_prompt",
]
