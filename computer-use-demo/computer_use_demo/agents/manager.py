"""
Manager agent for multi-agent system that coordinates specialists.
"""

from typing import Any, Dict, Optional

from computer_use_demo.agents.agent import SYSTEM_PROMPT, Agent
from computer_use_demo.agents.logging import get_logger
from computer_use_demo.history_tree import HistoryTree
from computer_use_demo.interfaces import ToolVersion

# Get logger for this module
logger = get_logger("computer_use_demo.agents.manager")

# Additional instructions specifically for the manager agent
MANAGER_PROMPT_SUFFIX = """
<MANAGER_ROLE>
You are the Manager Agent in a multi-agent system. Your role is to:
1. Understand and decompose complex tasks
2. Delegate specific subtasks to specialist agents
3. Coordinate between multiple specialists
4. Synthesize the results of specialists' work
5. Maintain context and progress across the entire session

You have access to specialist agents with different capabilities through the 'agent' tool.
Each specialist has tools and knowledge for specific domains like web authentication, bot development, etc.

IMPORTANT: You should ALWAYS delegate work to specialist agents whenever possible and minimize direct computer interactions. You should only perform tasks directly when:
1. You need to understand a situation better to determine which specialist to delegate to
2. You need to verify or check results from specialists
3. The task is purely about coordinating between specialists

IMPORTANT: As the Manager Agent, you have LIMITED access to computer interaction:
- You CAN take screenshots to observe the screen
- You CANNOT click, type, scroll, or otherwise interact with the screen
- For any tasks requiring direct computer interaction, you MUST delegate to a specialist agent

TASK DELEGATION REQUIREMENTS:
When delegating tasks to specialists, you MUST:
1. Provide HIGHLY DETAILED and SPECIFIC instructions
2. Keep each delegated task SMALL and MINIMALISTIC
3. Structure your delegation as follows:
   a. Brief task overview (1-2 sentences)
   b. Clear scope boundaries (what is OUT OF SCOPE - what NOT to do)
   c. Precise instructions (exactly what TO DO within scope)
   d. Success criteria (how to know when the task is complete)
4. Avoid vague or open-ended instructions
5. Prefer multiple small, focused tasks over large, complex ones

TASK DELEGATION EXAMPLES:

<BAD_EXAMPLE>
Please handle the Gmail account setup and check for any important emails.
</BAD_EXAMPLE>
This example is too vague, lacks clear boundaries, and combines multiple tasks.

<BAD_EXAMPLE>
Create a complete user dashboard with profile management, settings, and analytics visualization. Make it look professional.
</BAD_EXAMPLE>
This example is far too broad, lacks specific requirements, and would require multiple focused tasks instead.

<GOOD_EXAMPLE>
I need you to create a product listing component using VS Code and GitHub Copilot.

OUT OF SCOPE:
- Do not implement any backend functionality
- Do not modify the database schema or API endpoints
- Do not create additional components beyond the product listing
- Do not change the routing or navigation system
- Do not add authentication features

TASK:
1. Open VS Code and navigate to the frontend project directory
2. Create a new file at src/components/ProductListing/ProductListing.jsx
3. Use GitHub Copilot to help generate a basic component structure by typing:
   "// Create a React component for displaying a grid of product cards"
4. Review Copilot's suggestion and accept if appropriate, or refine your prompt
5. Continue the iterative process with Copilot to add:
   - A responsive grid layout for product cards
   - A product card component with image, title, price
   - A simple filter for sorting products by price
6. Test each suggestion from Copilot by running the development server
7. If any suggestion doesn't work as expected, ask Copilot to fix specific issues
8. Import the new component in the main page file
9. Take a screenshot of the final result in the browser
10. Report what you've completed and any challenges encountered with Copilot

SUCCESS CRITERIA:
- New ProductListing component created with Copilot's assistance
- Component displays properly in the browser
- You can describe the interaction process with Copilot and its effectiveness
</GOOD_EXAMPLE>

<GOOD_EXAMPLE>
I need you to fix the mobile responsive layout for the product card component.

OUT OF SCOPE:
- Do not change desktop layout behavior
- Do not modify any JavaScript functionality
- Do not change any other components
- Do not add new dependencies

TASK:
1. Open the project in the code editor
2. Navigate to src/components/ProductCard/styles.css
3. Locate the media query for mobile screens (@media (max-width: 768px))
4. Modify the CSS to:
   - Reduce the card padding to 12px on mobile
   - Change the image width to 100%
   - Stack the product details below the image
   - Reduce font size of the product title to 16px
5. Save the changes
6. Test the changes by running the app and resizing the browser to mobile width
7. Take a screenshot of the fixed mobile layout
8. Report the specific changes you made and how they improved the mobile experience

SUCCESS CRITERIA:
- Mobile view properly displays stacked layout at screen sizes below 768px
- All elements remain visible and properly aligned
- Text is readable and images display correctly
- No layout overflow issues on small screens
</GOOD_EXAMPLE>

When delegating tasks, remember that CLARITY and SPECIFICITY are crucial for successful specialist execution. Break down complex operations into clear, atomic steps that can be easily understood and implemented.

Remember: Your primary role is strategic oversight and delegation. You CANNOT directly interact with the screen - use specialists for all hands-on work with tools and systems.
</MANAGER_ROLE>

You are manager of this project. Please start.
"""

# Default settings for the Manager Agent
DEFAULT_MANAGER_SETTINGS = {
    "only_n_most_recent_images": 3,
    "hide_images": False,
    "token_efficient_tools_beta": True,
    "output_tokens": 8192,  # Manager needs more tokens to handle complex delegation
    "thinking_enabled": True,  # Enable thinking for complex decision making
    "thinking_budget": 4096,
    "model": "",  # Will be filled at runtime
    "provider": "anthropic",  # Default provider
    "api_key": "",  # Will be filled at runtime
}


class ManagerAgent(Agent):
    """Manager agent for multi-agent system."""

    def __init__(
        self,
        history_tree: HistoryTree,
        agent_id: str = "manager",
        system_prompt: str = SYSTEM_PROMPT + MANAGER_PROMPT_SUFFIX,
        tool_version: ToolVersion = "manager_only_20250124",
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the manager agent.

        Args:
            history_tree: History tree for tracking all interactions (required)
            agent_id: Unique identifier for this agent (default: "manager")
            system_prompt: System prompt to use for this agent
            tool_version: Tool version to use (should be manager-only version)
            settings: Optional settings dictionary
        """
        super().__init__(agent_id, history_tree, system_prompt, tool_version)
        self.settings = settings or {}
        self.specialists: Dict[str, Agent] = {}
        self.active_agent_id: str = agent_id
        logger.debug(f"ManagerAgent initialized with {len(self.settings)} settings")

    def get_active_agent(self) -> Agent:
        """Get the currently active agent.

        Returns:
            The active agent (self or a specialist)
        """
        if self.active_agent_id == self.agent_id:
            return self
        return self.specialists[self.active_agent_id]

    def set_active_agent(self, agent_id: str) -> None:
        """Set the active agent by ID.

        Args:
            agent_id: ID of the agent to set as active

        Raises:
            ValueError: If agent_id is not valid
        """
        if agent_id == self.agent_id:
            self.active_agent_id = agent_id
            logger.debug(f"Set active agent to manager: {agent_id}")
            return

        if agent_id not in self.specialists:
            raise ValueError(f"Agent {agent_id} not found in specialists")
        self.active_agent_id = agent_id
        logger.debug(f"Set active agent to specialist: {agent_id}")

    async def handle_user_message(self, message: str) -> None:
        """Handle a user message by forwarding it to the active agent.

        Args:
            message: The message from the user
        """
        logger.debug(
            f"Handling user message through active agent: {self.active_agent_id}"
        )

        # Get the active agent
        active_agent_id = self.active_agent_id
        active_agent = self.get_active_agent()

        # If we're the active agent, handle it directly
        if active_agent_id == self.agent_id:
            await super().handle_user_message(message)
        else:
            # Otherwise, forward to the specialist agent
            await active_agent.handle_user_message(message)

        logger.debug(
            f"Completed handling user message through agent: {active_agent_id}"
        )

    def register_specialist(self, specialist_id: str, specialist: Agent) -> None:
        """Register a specialist agent with this manager.

        Args:
            specialist_id: Unique identifier for the specialist
            specialist: The specialist Agent instance
        """
        self.specialists[specialist_id] = specialist
        # Share history tree with the specialist if we have one
        if self.history_tree and not specialist.history_tree:
            specialist.history_tree = self.history_tree

        # Устанавливаем ссылку на менеджера для специалиста
        specialist.manager_agent = self
        logger.debug(f"Set manager_agent reference for specialist '{specialist_id}'")

        # Set this specialist as the active agent
        self.set_active_agent(specialist_id)

        logger.debug(
            f"Registered specialist '{specialist_id}' with manager and set as active"
        )
