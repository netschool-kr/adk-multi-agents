from dotenv import load_dotenv
load_dotenv()

import requests
from google.adk.tools import ToolContext

def get_weather(city: str, tool_context: ToolContext) -> str:
    """Get the current temperature in the specified city."""
    # Map city to coordinates (for demo purposes; a real app might call a geocoding API)
    coords = {"London": (51.5072, -0.1276), "New York": (40.7128, -74.0060)}
    if city not in coords:
        return f"Sorry, I don't have data for {city}."
    lat, lon = coords[city]

    # Call the external weather API
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        resp = requests.get(url, timeout=5)
    except requests.exceptions.RequestException as e:
        # Handle network errors (timeout, DNS failure, etc.)
        return "Error: Could not retrieve weather data at this time."
    if resp.status_code != 200:
        return f"Error: Weather service responded with status {resp.status_code}."
    
    data = resp.json()
    if "current_weather" not in data:
        return "Error: Unexpected response format from weather API."
    temp_c = data["current_weather"]["temperature"]

    # Save result in session state for potential future use
    tool_context.state["last_weather"] = {"city": city, "temperature_c": temp_c}
    return f"Currently, it is {temp_c}°C in {city}."

from google.adk.agents.llm_agent import LlmAgent
# Create an LLM-powered agent and equip it with the weather tool
weather_agent = LlmAgent(
    model="gemini-2.0-flash",  # or another model available in your environment
    name="WeatherBot",
    instruction="You are a weather assistant. Use tools to get real-time data when needed.",
    tools=[get_weather]     # Register our tool function
)

root_agent = weather_agent



if __name__ == "__main__":
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types

    # Session and runner setup
    APP_NAME = "weather"
    USER_ID = "default_user"
    SESSION_ID = "session_1"

    # Initialize session service
    session_service = InMemorySessionService()
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # Create runner
    runner = Runner(agent=weather_agent, app_name=APP_NAME, session_service=session_service)

    print("Running CountryCapitalInfoAgent... (Ctrl+C to exit)")
    while True:
        try:
            city = input("Enter a City name: ")
            user_input = f"What is the weather in {city} right now?"
            content = types.Content(role="user", parts=[types.Part(text=user_input)])
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
            for event in events:
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    final_text = event.content.parts[0].text
                    print("Agent says:", final_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
