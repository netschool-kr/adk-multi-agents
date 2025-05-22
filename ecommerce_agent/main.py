from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google.adk.runners import Runner

# Import root_agent from agent module (support both package and script contexts)
try:
    from ecommerce_agent.agent import root_agent
except ImportError:
    from agent import root_agent

from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from typing import Optional

app = FastAPI()

class UserRequest(BaseModel):
    user_input: str
    user_id: str = "user1"
    session_id: str = "session1"

# Initialize in-memory session service
session_service = InMemorySessionService()

async def _process_chat(user_input: str, user_id: str, session_id: str):
    """
    Common chat handler:
    - Ensures a session exists (creates if needed)
    - Wraps user_input into Content and runs root_agent via Runner
    - Extracts and returns the final answer
    """
    # Asynchronously create or retrieve session
    await session_service.create_session(
        app_name="ECommerce_app",
        user_id=user_id,
        session_id=session_id
    )
    # Prepare runner
    runner = Runner(
        agent=root_agent,
        app_name="ECommerce_app",
        session_service=session_service
    )
    # Wrap user input into Content for ADK
    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_input)]
    )
    try:
        # Invoke the agent
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract final response text
    final_answer = ""
    for event in events:
        if hasattr(event, "is_final_response") and event.is_final_response():
            if event.content and event.content.parts:
                final_answer = "".join(
                    part.text for part in event.content.parts if hasattr(part, 'text') and part.text
                )
            break
    return {"response": final_answer}

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Home page with a chat input form."""
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <title>Chat with E-Commerce Agent</title>
      </head>
      <body>
        <h1>Chat with E-Commerce Agent</h1>
        <form action=\"/chat\" method=\"get\">
          <label for=\"user_input\">Message:</label>
          <input type=\"text\" id=\"user_input\" name=\"user_input\" required>
          <br/>
          <label for=\"user_id\">User ID:</label>
          <input type=\"text\" id=\"user_id\" name=\"user_id\" value=\"user1\">
          <br/>
          <label for=\"session_id\">Session ID:</label>
          <input type=\"text\" id=\"session_id\" name=\"session_id\" value=\"session1\">
          <br/>
          <button type=\"submit\">Send</button>
        </form>
      </body>
    </html>
    """

@app.post("/chat")
async def chat(request: UserRequest):
    """
    POST /chat
    Process a chat request using JSON body parameters.
    """
    return await _process_chat(request.user_input, request.user_id, request.session_id)

@app.get("/chat")
async def chat_get(
    user_input: Optional[str] = Query(None, description="The user's message"),
    user_id: str = Query("user1", description="Identifier for the user"),
    session_id: str = Query("session1", description="Identifier for the session")
):
    """
    GET /chat
    Process a chat request using query parameters.
    """
    if not user_input:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'user_input' is required. Example: /chat?user_input=Hello"
        )
    return await _process_chat(user_input, user_id, session_id)
