"""
History tree structure for tracking interactions across agents.

This module provides a strongly typed tree structure for tracking all interactions
in the multi-agent system, supporting nested sessions and specialized visualization.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)
from pydantic import BaseModel, ConfigDict, Field

from computer_use_demo.tools import ToolResult


class NodeType(str, Enum):
    """Possible types of history tree nodes."""

    # Basic message types
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"

    # Tool interaction
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Special nodes
    THINKING = "thinking"
    SPECIALIST_SESSION = "specialist_session"  # Contains a nested HistoryTree
    DELEGATE_TASK = "delegate_task"  # For tracking task delegation
    SESSION_RESULT = "session_result"  # Result of a specialist session


class HistoryNode(BaseModel):
    """Strongly typed node in the history tree."""

    # Allow arbitrary types (including forward references to HistoryTree)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core fields
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    timestamp: datetime = Field(default_factory=datetime.now)
    parent_id: Optional[str] = None

    # Content fields - exactly one must be specified based on node_type
    message_content: Optional[List[BetaContentBlockParam]] = None
    tool_call: Optional[BetaToolUseBlockParam] = None
    tool_result: Optional[Union[BetaToolResultBlockParam, ToolResult]] = None
    thinking_content: Optional[str] = None
    delegate_info: Optional[Dict[str, str]] = None
    session_result: Optional[Dict[str, Any]] = None

    # For specialist_session, this will contain a nested tree
    nested_tree: Optional["HistoryTree"] = None

    # Metadata
    agent_id: str
    role: Optional[str] = None  # user, assistant, system
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize node with validation based on node_type."""
        super().__init__(**data)
        self._validate_fields_for_type()

    def _validate_fields_for_type(self) -> None:
        """Ensure required fields are set based on node_type."""
        if self.node_type == NodeType.USER_MESSAGE:
            if not self.message_content:
                raise ValueError("message_content is required for USER_MESSAGE nodes")
            if self.role != "user":
                raise ValueError("role must be 'user' for USER_MESSAGE nodes")

        elif self.node_type == NodeType.ASSISTANT_MESSAGE:
            if not self.message_content:
                raise ValueError(
                    "message_content is required for ASSISTANT_MESSAGE nodes"
                )
            if self.role != "assistant":
                raise ValueError("role must be 'assistant' for ASSISTANT_MESSAGE nodes")

        elif self.node_type == NodeType.SYSTEM_MESSAGE:
            if not self.message_content:
                raise ValueError("message_content is required for SYSTEM_MESSAGE nodes")
            if self.role != "system":
                raise ValueError("role must be 'system' for SYSTEM_MESSAGE nodes")

        elif self.node_type == NodeType.TOOL_CALL:
            if not self.tool_call:
                raise ValueError("tool_call is required for TOOL_CALL nodes")

        elif self.node_type == NodeType.TOOL_RESULT:
            if not self.tool_result:
                raise ValueError("tool_result is required for TOOL_RESULT nodes")

        elif self.node_type == NodeType.THINKING:
            if not self.thinking_content:
                raise ValueError("thinking_content is required for THINKING nodes")

        elif self.node_type == NodeType.SPECIALIST_SESSION:
            if not self.nested_tree:
                raise ValueError("nested_tree is required for SPECIALIST_SESSION nodes")
            if not self.delegate_info:
                raise ValueError(
                    "delegate_info is required for SPECIALIST_SESSION nodes"
                )

        elif self.node_type == NodeType.DELEGATE_TASK:
            if not self.delegate_info:
                raise ValueError("delegate_info is required for DELEGATE_TASK nodes")

        elif self.node_type == NodeType.SESSION_RESULT:
            if not self.session_result:
                raise ValueError("session_result is required for SESSION_RESULT nodes")

    def add_child(self, node: "HistoryNode") -> None:
        """Add child relationship to this node."""
        node.parent_id = self.node_id

    def to_ui_dict(self) -> Dict[str, Any]:
        """Convert node to a dictionary suitable for UI rendering."""
        ui_dict = {
            "id": self.node_id,
            "type": self.node_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
        }

        if self.role:
            ui_dict["role"] = self.role

        if self.node_type == NodeType.USER_MESSAGE:
            ui_dict["content"] = self.message_content

        elif self.node_type == NodeType.ASSISTANT_MESSAGE:
            ui_dict["content"] = self.message_content

        elif self.node_type == NodeType.SYSTEM_MESSAGE:
            ui_dict["content"] = self.message_content

        elif self.node_type == NodeType.TOOL_CALL:
            ui_dict["tool_name"] = self.tool_call["name"]
            ui_dict["tool_input"] = self.tool_call["input"]
            ui_dict["tool_id"] = self.tool_call["id"]

        elif self.node_type == NodeType.TOOL_RESULT:
            if isinstance(self.tool_result, dict):
                # It's a BetaToolResultBlockParam
                ui_dict["is_error"] = self.tool_result.get("is_error", False)
                ui_dict["content"] = self.tool_result.get("content", "")
                ui_dict["tool_use_id"] = self.tool_result.get("tool_use_id", "")

                # Check if there's an image in the content
                if isinstance(self.tool_result.get("content", []), list):
                    for item in self.tool_result["content"]:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "image"
                            and item.get("source", {}).get("type") == "base64"
                        ):
                            ui_dict["has_image"] = True
                            ui_dict["image_data"] = item["source"]["data"]
            else:
                # It's a ToolResult
                ui_dict["output"] = self.tool_result.output
                ui_dict["error"] = self.tool_result.error
                ui_dict["has_image"] = bool(self.tool_result.base64_image)
                if self.tool_result.base64_image:
                    ui_dict["image_data"] = self.tool_result.base64_image

            # Check if extra image data exists in metadata
            if self.metadata.get("has_image") and self.metadata.get("image_data"):
                ui_dict["has_image"] = True
                ui_dict["image_data"] = self.metadata["image_data"]

        elif self.node_type == NodeType.THINKING:
            ui_dict["thinking"] = self.thinking_content

        elif self.node_type == NodeType.SPECIALIST_SESSION:
            ui_dict["specialist_id"] = self.delegate_info.get("specialist_id", "")
            ui_dict["task"] = self.delegate_info.get("task", "")
            ui_dict["nested_tree"] = (
                self.nested_tree.to_ui_dict() if self.nested_tree else None
            )

        elif self.node_type == NodeType.DELEGATE_TASK:
            ui_dict["specialist_id"] = self.delegate_info.get("specialist_id", "")
            ui_dict["task"] = self.delegate_info.get("task", "")

        elif self.node_type == NodeType.SESSION_RESULT:
            ui_dict["result"] = self.session_result.get("result", "")
            ui_dict["success"] = self.session_result.get("success", True)

        # Add metadata if present
        if self.metadata:
            ui_dict["metadata"] = self.metadata

        return ui_dict


class HistoryTree:
    """Tree structure for tracking all interactions in the system."""

    def __init__(self, tree_id: Optional[str] = None):
        """Initialize a new history tree."""
        self.nodes: Dict[str, HistoryNode] = {}
        self.root_nodes: List[str] = []  # Top-level nodes
        self.tree_id = tree_id or str(uuid.uuid4())
        self.current_context_id: Optional[str] = None  # Current active context
        self.on_update_callback: Optional[Callable] = None

    def set_update_callback(self, callback: Callable) -> None:
        """Set a callback function to be called when the tree is updated."""
        self.on_update_callback = callback

    def _trigger_update(self) -> None:
        """Trigger the update callback if set."""
        if self.on_update_callback:
            self.on_update_callback()

    def add_node(self, node: HistoryNode, parent_id: Optional[str] = None) -> str:
        """Add a node to the tree and return its ID."""
        # If parent specified, associate with it
        if parent_id and parent_id in self.nodes:
            node.parent_id = parent_id
            # No explicit child list in HistoryNode to keep it simple
        elif parent_id:
            raise ValueError(f"Parent node {parent_id} not found in tree")
        else:
            # No parent, this is a root node
            self.root_nodes.append(node.node_id)

        # Store the node
        self.nodes[node.node_id] = node

        # Update current context if not set
        if self.current_context_id is None:
            self.current_context_id = node.node_id

        # Trigger update callback
        self._trigger_update()

        return node.node_id

    def add_user_message(
        self,
        agent_id: str,
        content: List[BetaContentBlockParam],
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a user message node."""
        node = HistoryNode(
            node_type=NodeType.USER_MESSAGE,
            message_content=content,
            agent_id=agent_id,
            role="user",
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def add_assistant_message(
        self,
        agent_id: str,
        content: List[BetaContentBlockParam],
        parent_id: Optional[str] = None,
    ) -> str:
        """Add an assistant message node."""
        node = HistoryNode(
            node_type=NodeType.ASSISTANT_MESSAGE,
            message_content=content,
            agent_id=agent_id,
            role="assistant",
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def add_system_message(
        self,
        agent_id: str,
        content: List[BetaContentBlockParam],
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a system message node."""
        node = HistoryNode(
            node_type=NodeType.SYSTEM_MESSAGE,
            message_content=content,
            agent_id=agent_id,
            role="system",
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def add_thinking(
        self, agent_id: str, thinking_content: str, parent_id: Optional[str] = None
    ) -> str:
        """Add a thinking node."""
        node = HistoryNode(
            node_type=NodeType.THINKING,
            thinking_content=thinking_content,
            agent_id=agent_id,
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def add_tool_call(
        self,
        agent_id: str,
        tool_call: BetaToolUseBlockParam,
        parent_id: Optional[str] = None,
    ) -> str:
        """Add a tool call node."""
        node = HistoryNode(
            node_type=NodeType.TOOL_CALL, tool_call=tool_call, agent_id=agent_id
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def add_tool_result(
        self,
        agent_id: str,
        tool_result: Union[BetaToolResultBlockParam, ToolResult],
        parent_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a tool result node."""
        # Create metadata dictionary if provided
        metadata = {}
        if extra_metadata:
            metadata.update(extra_metadata)

        node = HistoryNode(
            node_type=NodeType.TOOL_RESULT,
            tool_result=tool_result,
            agent_id=agent_id,
            metadata=metadata,
        )
        return self.add_node(node, parent_id or self.current_context_id)

    def start_specialist_session(
        self,
        manager_id: str,
        specialist_id: str,
        task: str,
        context: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Start a new specialist session with a nested tree."""
        # Create a new nested tree for this session
        nested_tree = HistoryTree()

        # Create delegate info
        delegate_info = {
            "specialist_id": specialist_id,
            "task": task,
            "start_time": datetime.now().isoformat(),
        }

        if context:
            delegate_info["context"] = context

        # Create the session node
        node = HistoryNode(
            node_type=NodeType.SPECIALIST_SESSION,
            delegate_info=delegate_info,
            nested_tree=nested_tree,
            agent_id=manager_id,
            metadata={"specialist_id": specialist_id},
        )

        # Add to tree
        session_id = self.add_node(node, parent_id or self.current_context_id)

        # Add delegation task info as first node in nested tree
        nested_tree.add_node(
            HistoryNode(
                node_type=NodeType.DELEGATE_TASK,
                delegate_info=delegate_info,
                agent_id=manager_id,
                metadata={"source": "manager", "target": specialist_id},
            )
        )

        # Set the current context to this session
        self.current_context_id = session_id

        return session_id

    def end_specialist_session(
        self, session_id: str, result: Any, success: bool = True
    ) -> None:
        """End a specialist session and return to parent context."""
        if session_id not in self.nodes:
            raise ValueError(f"Session node {session_id} not found in tree")

        session_node = self.nodes[session_id]
        if session_node.node_type != NodeType.SPECIALIST_SESSION:
            raise ValueError(f"Node {session_id} is not a specialist session")

        # Add session result to the nested tree
        if session_node.nested_tree:
            session_node.nested_tree.add_node(
                HistoryNode(
                    node_type=NodeType.SESSION_RESULT,
                    session_result={"result": result, "success": success},
                    agent_id=session_node.metadata.get("specialist_id", "unknown"),
                )
            )

        # Update the delegate info with completion timestamp
        if session_node.delegate_info:
            session_node.delegate_info["end_time"] = datetime.now().isoformat()
            session_node.delegate_info["success"] = success

        # Return to parent context if this is the current context
        if self.current_context_id == session_id:
            self.current_context_id = session_node.parent_id

    def get_node_children(self, node_id: str) -> List[HistoryNode]:
        """Get all direct children of a node."""
        return [node for node in self.nodes.values() if node.parent_id == node_id]

    def get_full_path(self, node_id: str) -> List[HistoryNode]:
        """Get the full path from root to this node."""
        path = []
        current_id = node_id

        while current_id:
            if current_id not in self.nodes:
                break

            node = self.nodes[current_id]
            path.insert(0, node)
            current_id = node.parent_id

        return path

    def get_node_with_ancestors(self, node_id: str) -> Dict[str, HistoryNode]:
        """Get a node and all its ancestors as a dict."""
        result = {}
        path = self.get_full_path(node_id)

        for node in path:
            result[node.node_id] = node

        return result

    def to_ui_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary suitable for UI rendering."""
        # Top level dict with tree metadata
        result = {
            "tree_id": self.tree_id,
            "nodes": {},
            "root_nodes": self.root_nodes.copy(),
        }

        # Add all nodes in dictionary form
        for node_id, node in self.nodes.items():
            result["nodes"][node_id] = node.to_ui_dict()

        return result

    def get_chronological_view(self) -> List[Dict[str, Any]]:
        """Get a flattened chronological view of the tree for UI display."""
        # Sort all nodes by timestamp
        sorted_nodes = sorted(self.nodes.values(), key=lambda node: node.timestamp)

        # Convert to UI dicts
        return [node.to_ui_dict() for node in sorted_nodes]

    def get_flat_conversation(self) -> List[Dict[str, Any]]:
        """Get a flattened conversation view (user/assistant only)."""
        # Get only user/assistant messages
        message_nodes = [
            node
            for node in self.nodes.values()
            if node.node_type in (NodeType.USER_MESSAGE, NodeType.ASSISTANT_MESSAGE)
        ]

        # Sort by timestamp
        sorted_nodes = sorted(message_nodes, key=lambda node: node.timestamp)

        # Convert to UI dicts
        return [node.to_ui_dict() for node in sorted_nodes]

    def get_recursive_conversation(
        self, include_tool_calls: bool = False
    ) -> List[Dict[str, Any]]:
        """Get a recursive view with specialist sessions expanded."""
        result = []

        def process_node(node: HistoryNode, depth: int = 0) -> None:
            """Process a node and its children recursively."""
            # Skip non-conversation nodes unless include_tool_calls is True
            if (
                node.node_type
                not in (
                    NodeType.USER_MESSAGE,
                    NodeType.ASSISTANT_MESSAGE,
                    NodeType.SYSTEM_MESSAGE,
                )
                and not (
                    include_tool_calls
                    and node.node_type in (NodeType.TOOL_CALL, NodeType.TOOL_RESULT)
                )
            ) and node.node_type != NodeType.SPECIALIST_SESSION:
                return

            # Add this node
            node_dict = node.to_ui_dict()
            node_dict["depth"] = depth
            result.append(node_dict)

            # If it's a specialist session, process its nested tree recursively
            if node.node_type == NodeType.SPECIALIST_SESSION and node.nested_tree:
                # Add header for specialist session
                result.append(
                    {
                        "type": "session_header",
                        "specialist_id": node.delegate_info.get("specialist_id", ""),
                        "task": node.delegate_info.get("task", ""),
                        "depth": depth + 1,
                    }
                )

                # Process nodes in nested tree
                for nested_node_id in node.nested_tree.root_nodes:
                    if nested_node_id in node.nested_tree.nodes:
                        nested_node = node.nested_tree.nodes[nested_node_id]
                        process_node(nested_node, depth + 1)

                        # Process children recursively
                        children = node.nested_tree.get_node_children(nested_node_id)
                        for child in children:
                            process_node(child, depth + 1)

                # Add footer for specialist session
                result.append(
                    {
                        "type": "session_footer",
                        "specialist_id": node.delegate_info.get("specialist_id", ""),
                        "depth": depth + 1,
                    }
                )

        # Start with root nodes
        for root_id in self.root_nodes:
            if root_id in self.nodes:
                process_node(self.nodes[root_id])

                # Process direct children
                for child in self.get_node_children(root_id):
                    process_node(child)

        return result
