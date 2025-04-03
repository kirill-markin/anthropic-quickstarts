# Agent Flow Changes

## Current System

Currently, when a new user message arrives while a tool is being executed:
1. The tool execution is stopped with a synthetic tool result
2. The new user message is added to the manager agent
3. The flow continues with the manager agent regardless of which agent was active

## Requested Changes

We want to modify the flow to:
1. Still stop the tool execution with a synthetic result, but identify which agent (not just specialist agent) was running the tool
2. Add the new user message to the currently active agent's history, not always to the manager
3. Continue the flow with the currently active agent (specialist if specialist was active, manager if manager was active)

## Files to Modify

### 1. `computer-use-demo/computer_use_demo/agents/manager.py`

This is the main file to modify as we'll add active agent tracking directly to the manager agent class:

Main changes:
- Add a new field `active_agent_id` to track which agent is currently processing
- Add methods to get/set the active agent
- Modify the specialist agent registration to set it as active

### 2. `computer-use-demo/computer_use_demo/streamlit.py`

Key changes:
- Update user message handling to get the active agent from manager_agent
- Route the new user message to the active agent instead of always to the manager
- Continue processing with the active agent instead of always with the manager

### 3. `computer-use-demo/computer_use_demo/tools/agent_tool.py`

When the specialist agent is invoked via the agent tool:
- Track that the specialist is now the active agent via manager_agent
- Save previous active agent and restore it when specialist processing completes

## Implementation Approach

1. Enhance the manager_agent:
   - Add `active_agent_id` field to ManagerAgent, initialized to "manager"
   - Add methods `set_active_agent(agent_id)` and `get_active_agent()`
   - Update `register_specialist` to set the new specialist as active

2. Modify user message handling:
   - In streamlit.py, use manager_agent.get_active_agent() to determine which agent should receive the message
   - Ensure tool interruption still works but properly identifies which agent was executing the tool

3. Update flow control:
   - Change the logic after receiving a user message to continue with the active agent
   - When a specialist session ends, restore the previous active agent via manager_agent.set_active_agent()

4. Agent tool modifications:
   - Save the previous active agent ID before setting specialist as active
   - When specialist completes, reset active agent back to previous

## Testing Plan

1. Test sending a message while manager is active
2. Test sending a message while a specialist agent is active
3. Test interrupting a tool execution in both manager and specialist contexts
4. Test multiple back-and-forth exchanges with a specialist 