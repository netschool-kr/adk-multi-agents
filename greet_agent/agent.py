from dotenv import load_dotenv
load_dotenv()
from google.adk.agents import LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
import uuid
MODEL_NAME = "gemini-2.0-flash"
from google.genai import types

# ---- Use google.genai.types for Content and Part ----
try:
    from google.genai import types as adk_types # Assuming google.genai is google.generativeai or similar
except ImportError:
    print("CRITICAL ERROR: Could not import 'types' from 'google.genai'.")
    print("Please ensure 'google-generativeai' library is installed ('pip install google-generativeai').")
    adk_types = None



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

if __name__ == "__main__":
    
    # Session and runner setup
    APP_NAME = "greet_agent"
    USER_ID = "default_user"
    SESSION_ID = "session_1"

    # Initialize session service
    session_service = InMemorySessionService()
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # Create runner
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print("Running Greet_Agent... (Ctrl+C to exit)")
    while True:
        try:
            user_input = input("User Input: ")
            content = types.Content(role="user", parts=[types.Part(text=user_input)])
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
            for event in events:
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    final_text = event.content.parts[0].text
                    print(final_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")    
