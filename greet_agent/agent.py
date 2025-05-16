import asyncio
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import uuid
MODEL_NAME = "gemini-2.0-flash"
# ---- Use google.genai.types for Content and Part ----
try:
    from google.genai import types as adk_types # Assuming google.genai is google.generativeai or similar
except ImportError:
    print("CRITICAL ERROR: Could not import 'types' from 'google.genai'.")
    print("Please ensure 'google-generativeai' library is installed ('pip install google-generativeai').")
    adk_types = None

# ---- ADK Session Management Imports ----
try:
    from google.adk.sessions.core import Session, SessionContext
    # from google.adk.messages import Message # Import if you need to manipulate message history directly
except ImportError:
    print("CRITICAL ERROR: Could not import 'Session' or 'SessionContext' from 'google.adk.sessions.core'.")
    print("Please ensure the ADK library is correctly installed and these modules are available.")
    # Define dummy classes if import fails to prevent NameError, but functionality will be broken
    class Session: pass
    class SessionContext: pass


# --- Agent Definitions ---
def get_weather(location: str) -> str:
    """Tool to get weather information for a specified location (example)"""
    return f"The weather in {location} is sunny and warm."

greeting_agent = Agent(
    model=LiteLlm(model="anthropic/claude-3-sonnet-20240229"),
    name="greeting_agent",
    description="Specializes in providing friendly greetings to the user.",
    instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. Be concise and welcoming."
)

farewell_agent = Agent(
    model=LiteLlm(model="anthropic/claude-3-sonnet-20240229"),
    name="farewell_agent",
    description="Specializes in providing friendly farewells to the user.",
    instruction="You are the Farewell Agent. Your ONLY task is to provide a friendly farewell when the user indicates they are leaving. Be concise."
)

question_answer_agent = LlmAgent(
    model=MODEL_NAME,
    name="question_answer_agent",
    description="A helpful assistant agent that can answer general questions on a variety of topics.",
    instruction="""You are a Question Answering Agent. Answer the user's questions clearly, concisely, and accurately.
If the question is outside your capabilities or knowledge, politely state that you cannot answer it."""
)

weather_specific_agent = LlmAgent(
    model=MODEL_NAME,
    name="weather_specific_agent",
    description="Specializes in providing weather forecasts and information using available tools.",
    instruction="""You are the Weather Agent. Your task is to provide weather information.
Use the `get_weather` tool if a location is mentioned or can be inferred.
If no location is provided, you can ask the user for a location.""",
    tools=[get_weather]
)

root_agent = LlmAgent(
    model=MODEL_NAME,
    name="root_orchestrator_agent",
    description="The main orchestrator agent that understands user requests and delegates to specialized agents.",
    instruction="""You are a master orchestrator agent. Your primary role is to understand the user's intent and delegate the task to the appropriate specialized sub-agent.
- If the user offers a greeting (e.g., "hello", "hi"), delegate to `greeting_agent`.
- If the user says goodbye or indicates they are leaving (e.g., "bye", "see you"), delegate to `farewell_agent`.
- If the user asks a general question (e.g., "What is the capital of France?", "Explain quantum physics."), delegate to `question_answer_agent`.
- If the user asks about the weather (e.g., "What's the weather in London?", "Will it rain tomorrow?"), delegate to `weather_specific_agent`.
Do not answer directly if a specialized agent can handle the request. Focus on delegation.
Provide only the response from the delegated agent, without adding your own conversational remarks unless necessary for clarification of delegation.
""",
    sub_agents=[
        greeting_agent,
        farewell_agent,
        question_answer_agent,
        weather_specific_agent
    ]
)
# --- End of Agent Definitions ---

async def main():
    print("Initializing agents and session...")
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name="adk_orchestrator_app"
    )
            
    print("Agent system ready. Type 'exit' to end the conversation.")
    print("-" * 30)

    current_user_id = str(uuid.uuid4())
    current_session_id = str(uuid.uuid4()) # This is the string ID
    print(f"Using User ID: {current_user_id}")
    print(f"Using Session ID (string): {current_session_id}")

    # ---- Explicitly create and save the session ----
    try:
        if 'Session' in globals() and 'SessionContext' in globals() and Session is not None and SessionContext is not None:
            # Create an ADK Session object
            # The Session object from ADK typically uses 'id' for its identifier field.
            initial_adk_session = Session(
                id=current_session_id, # Use the string ID for the ADK Session object's 'id'
                user_id=current_user_id,
                context=SessionContext(), # Provide an empty context
                history=[] # Provide an empty history (list of Message objects)
                # metadata=None # Or some default metadata if applicable
            )
            session_service.save_session(initial_adk_session)
            print(f"ADK Session object for ID '{current_session_id}' explicitly created and saved.")
        else:
            print("CRITICAL: ADK Session or SessionContext not imported correctly. Cannot explicitly create session.")
            # Potentially exit or handle this critical failure
            return # Exit main if session management types are not available
    except Exception as e:
        print(f"Error explicitly creating/saving ADK session: {e}")
        # Decide if to continue or not if this fails; for now, we'll let it try to proceed
    print("-" * 30)


    while True:
        user_input_text = input("User: ")
        if user_input_text.lower() == 'exit':
            print("Ending conversation.")
            break
        if not user_input_text:
            continue

        print("Root Agent Processing...")

        message_content = None
        if adk_types is None:
            print("\nCRITICAL: 'adk_types' module not loaded. Cannot create message content.")
            continue 

        try:
            if hasattr(adk_types, 'Content') and hasattr(adk_types, 'Part'):
                message_content = adk_types.Content(
                    parts=[adk_types.Part(text=user_input_text)]
                )
            else:
                print("\nError: 'Content' or 'Part' attributes not found in the imported 'adk_types' module.")
                continue
        except Exception as e:
            print(f"\nError creating Content object: {e}")
            continue
        
        if message_content is None:
            print("\nError: Failed to create message_content. Skipping processing.")
            continue
            
        try:
            # No 'await' here as runner.run_async returns an async generator
            response_stream = runner.run_async(
                user_id=current_user_id,
                session_id=current_session_id, # Pass the string session ID
                new_message=message_content
            )
        except Exception as e:
            print(f"\nError calling runner.run_async: {e}")
            continue 

        full_response = ""
        try:
            async for message_event in response_stream:
                if message_event.is_llm_response and message_event.content:
                    print(f"Agent: {message_event.content}", end="", flush=True)
                    full_response += message_event.content
                elif message_event.is_tool_call:
                    print(f"\n[Tool Call: {message_event.tool_name} with args {message_event.tool_input}]", flush=True)
                elif message_event.is_tool_response:
                    print(f"\n[Tool Response from {message_event.tool_name}: {message_event.content}]", flush=True)
        except Exception as e:
            print(f"\nError processing response stream: {e}") # This is where "Session not found" occurred

        if not full_response:
            print() 
        print("\n" + "-" * 30)

if __name__ == "__main__":
    print("--- ADK Orchestrator Example ---")
    print("This script demonstrates a root agent delegating tasks to specialized sub-agents.")
    print("Ensure your API keys for LLM providers are set in your environment.")
    print("This script expects 'google.genai.types' for Content/Part. Install with: pip install google-generativeai")
    print("It also expects ADK session types from 'google.adk.sessions.core'.\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting due to user interruption.")
    except Exception as e:
        print(f"\nAn unexpected error occurred at the top level: {e}")
