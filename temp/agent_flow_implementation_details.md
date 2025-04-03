# Agent Flow Implementation Details

## Specific Changes by File

### 1. Modify `ManagerAgent` Class

Instead of creating a separate utility module, we'll add active agent tracking directly to the `ManagerAgent` class:

```python
# computer-use-demo/computer_use_demo/agents/manager.py

class ManagerAgent(BaseAgent):
    """Manager agent that can delegate tasks to specialist agents."""

    def __init__(self, agent_id: str = "manager", **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.specialists = {}
        self.active_agent_id = "manager"  # Track the currently active agent
        logger.debug("Created manager agent with initial active_agent_id='manager'")

    def set_active_agent(self, agent_id: str) -> None:
        """Set the active agent ID.
        
        Args:
            agent_id: ID of the agent to set as active
        """
        self.active_agent_id = agent_id
        logger.debug(f"Set active agent to: {agent_id}")
    
    def get_active_agent(self):
        """Get the currently active agent instance.
        
        Returns:
            Agent instance that is currently active
        """
        if self.active_agent_id != "manager" and self.active_agent_id in self.specialists:
            return self.specialists[self.active_agent_id]
        return self
    
    def register_specialist(self, specialist_id: str, specialist_agent) -> None:
        """Register a specialist agent and set it as active.
        
        Args:
            specialist_id: ID for the specialist
            specialist_agent: The specialist agent instance
        """
        self.specialists[specialist_id] = specialist_agent
        self.set_active_agent(specialist_id)
        logger.debug(f"Registered specialist {specialist_id} and set as active agent")
```

### 2. `computer-use-demo/computer_use_demo/streamlit.py`

#### A. Extract helper function for tool interruption

```python
def handle_tool_interruption(agent_instance, agent_id: str) -> bool:
    """Handle interruption of any ongoing tool calls for the given agent.
    
    Args:
        agent_instance: The agent instance to check for unclosed tool calls
        agent_id: ID of the agent being interrupted
    
    Returns:
        bool: True if a tool was interrupted, False otherwise
    """
    if not agent_instance or not agent_instance.history.messages:
        return False
        
    last_message = agent_instance.history.messages[-1]
    if last_message.get("role") != "assistant" or not last_message.get("content"):
        return False
        
    for content_item in last_message.get("content", []):
        # Check if the item is a tool_use without a following tool_result
        if isinstance(content_item, dict) and content_item.get("type") == "tool_use":
            # Found an unclosed tool call
            logger.debug(
                f"Found unclosed tool call in {agent_id} before user message: {content_item.get('id')}"
            )
            
            # Create a synthetic tool result message
            tool_id = content_item.get("id", "")
            tool_result = BetaToolResultBlockParam(
                type="tool_result",
                tool_use_id=tool_id,
                content=[{"type": "text", "text": INTERRUPT_TOOL_ERROR}],
            )
            
            # Add synthetic tool result to history
            interrupt_message = {"role": "user", "content": [tool_result]}
            agent_instance.history.append(
                cast(BetaMessageParam, interrupt_message)
            )
            
            # Create a ToolResult object for history tree
            interrupt_tool_result = ToolResult(
                output=f"{INTERRUPT_TEXT}",
                error=INTERRUPT_TOOL_ERROR,
                base64_image="",
            )
            
            # Add to history tree for display
            st.session_state.history_tree.add_tool_result(
                agent_id=agent_id, tool_result=interrupt_tool_result
            )
            
            logger.debug(
                f"Added synthetic tool result for interrupted tool call: {tool_id}"
            )
            return True
    
    return False
```

#### B. Modify user message handling in main() function

```python
# Process new user message
if new_message:
    # Get the currently active agent from the manager
    active_agent = st.session_state.manager_agent.get_active_agent()
    active_agent_id = st.session_state.manager_agent.active_agent_id
    
    logger.debug(f"Handling user message with active agent: {active_agent_id}")
    
    # Handle any unclosed tool calls in the active agent
    handle_tool_interruption(active_agent, active_agent_id)
    
    # Create a text block for the message
    text_block = BetaTextBlockParam(type="text", text=new_message)
    
    # Add message to the history tree
    st.session_state.history_tree.add_user_message(
        agent_id="user", content=[text_block]
    )
    
    # Add the message to the active agent's history
    user_message = {"role": "user", "content": [text_block]}
    active_agent.history.append(
        cast(BetaMessageParam, user_message)
    )
    logger.debug(
        f"Added user message to {active_agent_id} agent history: {new_message[:30]}..."
    )
    
    with track_sampling_loop():
        try:
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
                thinking_budget=st.session_state.agent_settings["thinking_budget"]
                if st.session_state.agent_settings["thinking_enabled"]
                else None,
                token_efficient_tools_beta=st.session_state.agent_settings[
                    "token_efficient_tools_beta"
                ],
            )
        except RateLimitError as e:
            # Handle rate limit error
            handle_rate_limit_error(e)
        except Exception as e:
            # Log any other errors
            handle_general_error(e)
```

### 3. `computer-use-demo/computer_use_demo/tools/agent_tool.py`

Modify the `_call` method to use the manager_agent for tracking the active agent:

```python
async def _call(
    self,
    specialist: str,
    task: str,
    context: Optional[str] = None,
) -> ToolResult:
    """Delegate a task to a specialist agent."""
    # Import here to avoid circular dependency
    from computer_use_demo.agents import SpecialistAgent

    try:
        logger.debug(
            f"Creating new specialist '{specialist}' for task: {task[:50]}..."
        )

        # Ensure the tool is configured properly
        if not self.ensure_config():
            logger.error("Failed to ensure proper configuration for AgentTool")
            return ToolResult(
                error="Agent tool not properly configured. Could not automatically fix the configuration."
            )

        # Create a new specialist agent for this task
        # We'll use a timestamp in the ID to ensure uniqueness
        specialist_id = f"{specialist}_{datetime.now().timestamp()}"

        # Create the specialist
        specialist_agent = SpecialistAgent(
            agent_id=specialist_id,
            specialist_type=specialist,
            tool_version=self.tool_version,
        )
        
        # Store the previous active agent to restore after specialist completes
        previous_active_id = self.manager_agent.active_agent_id
        logger.debug(f"Saving previous active agent ID: {previous_active_id}")
        
        # Connect to history tree if manager agent is available
        if (
            self.manager_agent
            and hasattr(self.manager_agent, "history_tree")
            and self.manager_agent.history_tree
        ):
            # Share the manager's history tree with the specialist
            specialist_agent.history_tree = self.manager_agent.history_tree
            
        # Register the specialist with the manager (which also sets it as active)
        if self.manager_agent:
            self.manager_agent.register_specialist(specialist_id, specialist_agent)

        # Prepare initial message for the specialist
        task_content = task
        if context:
            task_content = f"Context:\n{context}\n\nTask:\n{task}"

        # Create initial message for the specialist
        specialist_message: Dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": task_content}],
        }

        # Set specialist's history to just this message
        specialist_agent.history.messages = [specialist_message]

        # Run the specialist agent - with no callbacks
        result_messages = await specialist_agent.run(
            messages=specialist_agent.history.messages,
            model=self.model,
            provider=self.provider,
            system_prompt_suffix=self.system_prompt_suffix,
            output_callback=None,
            tool_output_callback=None,
            api_response_callback=None,
            api_key=self.api_key,
            only_n_most_recent_images=self.only_n_most_recent_images,
            max_tokens=self.max_tokens,
            thinking_budget=self.thinking_budget,
            token_efficient_tools_beta=self.token_efficient_tools_beta,
        )

        # Reset the active agent back to the previous one
        self.manager_agent.set_active_agent(previous_active_id)
        logger.debug(f"Restored active agent to: {previous_active_id}")

        # Process results and return response
        # ... (rest of the existing method)
```

## Benefits of This Implementation

1. **Simplicity**: Active agent management is centralized in the ManagerAgent class
2. **Less State Management**: No need for session state variables just for agent tracking
3. **Clear Ownership**: ManagerAgent controls the flow of agent delegation
4. **Minimal Changes**: Very focused modifications to existing code
5. **Better Encapsulation**: Agent flow logic stays with agent classes

## Edge Cases Handled

1. **Tool Calls in Progress**: Extracted into a dedicated function for clarity
2. **Agent Switching**: Uses manager_agent methods with proper logging
3. **Error Handling**: Restoration of previous active agent on completion
4. **Missing Specialist**: The get_active_agent() method defaults to manager if specialist not found
5. **Context Persistence**: Specialist conversations can continue across multiple user messages

## Testing Plan

1. **Basic Flow**: Test the manager -> specialist -> back to manager flow
2. **Interruption**: Test sending a message while a specialist is active
3. **Tool Interruption**: Test sending a message during a tool execution
4. **Extended Conversation**: Test having an extended conversation with a specialist
5. **Error Recovery**: Test error conditions to ensure active agent state is properly restored 