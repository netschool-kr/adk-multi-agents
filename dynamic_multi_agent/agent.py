# Import necessary libraries and modules
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner # For running the agent
from google.adk.sessions import InMemorySessionService # For session management
import asyncio # For running async code

# --- Tool Definitions ---
def get_weather(location: str) -> str:
    """Tool to get weather information for a specified location."""
    # In a real scenario, this would call a weather API
    return f"The weather in {location} is sunny and warm."

# --- Specialized Agent Definitions ---

# 1. Greeting Agent
greeting_agent = Agent(
    model=LiteLlm(model="anthropic/claude-3-sonnet-20240229"),
    name="greeting_agent",
    description="Specializes in providing friendly greetings to the user.",
    instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. Be concise and welcoming."
)

# 2. Farewell Agent
farewell_agent = Agent(
    model=LiteLlm(model="anthropic/claude-3-sonnet-20240229"),
    name="farewell_agent",
    description="Specializes in providing friendly farewells to the user.",
    instruction="You are the Farewell Agent. Your ONLY task is to provide a friendly farewell when the user indicates they are leaving. Be concise."
)

# 3. Question Answering Agent
Youtube_agent = LlmAgent(
    model="gemini-1.5-flash-latest", # Using a Gemini model capable of general Q&A
    name="Youtube_agent",
    description="A helpful assistant agent that can answer general questions on a variety of topics.",
    instruction="""You are a Question Answering Agent. Answer the user's questions clearly, concisely, and accurately.
If the question is outside your capabilities or knowledge, politely state that you cannot answer it."""
)

# 4. Weather Specific Agent
weather_specific_agent = LlmAgent(
    model="gemini-1.5-flash-latest", # Using a Gemini model
    name="weather_specific_agent",
    description="Specializes in providing weather forecasts and information using available tools.",
    instruction="""You are the Weather Agent. Your task is to provide weather information.
Use the `get_weather` tool if a location is mentioned or can be inferred.
If no location is provided, you can ask the user for a location.""",
    tools=[get_weather]
)

# --- Root Agent Definition ---
# This agent will orchestrate calls to the specialized agents.
root_agent = LlmAgent(
    model="gemini-1.5-pro-latest", # A more capable model for orchestration
    name="root_orchestrator_agent",
    description="The main orchestrator agent that understands user requests and delegates to specialized agents for greetings, farewells, general questions, and weather information.",
    instruction="""You are a master orchestrator agent. Your primary role is to understand the user's intent and delegate the task to the appropriate specialized sub-agent.
- If the user offers a greeting (e.g., "hello", "hi"), delegate to `greeting_agent`.
- If the user says goodbye or indicates they are leaving (e.g., "bye", "see you"), delegate to `farewell_agent`.
- If the user asks a general question (e.g., "What is the capital of France?", "Explain quantum physics."), delegate to `Youtube_agent`.
- If the user asks about the weather (e.g., "What's the weather in London?", "Will it rain tomorrow?"), delegate to `weather_specific_agent`.
- If the query is ambiguous or you are unsure, you can ask for clarification or state your capabilities.
Do not answer directly if a specialized agent can handle the request. Focus on delegation.
Provide only the response from the delegated agent, without adding your own conversational remarks unless necessary for clarification of delegation.
""",
    sub_agents=[
        greeting_agent,
        farewell_agent,
        Youtube_agent,
        weather_specific_agent
    ]
)

# --- Main Execution Block (Example) ---
async def main():
    print("Initializing agents and session...")
    # Configure session service
    session_service = InMemorySessionService()

    # Configure Runner with the root_agent
    runner = Runner(agent=root_agent, session_service=session_service)
    print("Agent system ready. Type 'exit' to end the conversation.")
    print("-" * 30)

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            print("Ending conversation.")
            break
        if not user_input:
            continue

        print("Root Agent Processing...")
        # Generate response asynchronously
        # The ADK runner handles the delegation based on the root_agent's LLM understanding
        # of the input and the descriptions/instructions of its sub_agents.
        response_stream = await runner.run_async(utterance=user_input)

        full_response = ""
        async for message in response_stream:
            if message.is_llm_response and message.content:
                print(f"Agent: {message.content}", end="", flush=True)
                full_response += message.content
            elif message.is_tool_call:
                print(f"\n[Tool Call: {message.tool_name} with args {message.tool_input}]", flush=True)
            elif message.is_tool_response:
                print(f"\n[Tool Response from {message.tool_name}: {message.content}]", flush=True)
            # Add more message type handling if needed (e.g., errors, agent start/end)

        if not full_response: # If only tool calls/responses occurred without direct LLM output to user
            print() # Ensure a newline if there was no direct agent text response
        print("\n" + "-" * 30)


if __name__ == "__main__":
    # Note: To run this, you need to have the google-adk library installed
    # and your environment configured for the LLM providers (e.g., Google AI, Anthropic).
    # This includes setting API keys.
    # Example: GOOGLE_API_KEY, ANTHROPIC_API_KEY (depending on LiteLLM setup)
    #
    # You might need to install specific packages:
    # pip install google-adk litellm google-generativeai anthropic
    print("--- ADK Orchestrator Example ---")
    print("This script demonstrates a root agent delegating tasks to specialized sub-agents.")
    print("Ensure your API keys for LLM providers are set in your environment.\n")

    # Example of how to run (actual API calls will be made)
    # For a real run, ensure API keys are set.
    # Example:
    # import os
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    # os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to user interruption.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have the necessary libraries installed and API keys configured.")
