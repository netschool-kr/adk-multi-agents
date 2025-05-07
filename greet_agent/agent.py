from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import os

model_name = os.environ.get("ADK_MODEL", "gemini-2.0-flash")
root_agent = LlmAgent(name="greeter", model=model_name)
#print("root_agent defined:", root_agent)

def greet(name: str) -> str:
    """Returns a greeting message for the given name."""
    return f"Hello, {name}!"

# Create an LLM-backed agent that uses the greet tool
simple_agent = LlmAgent(
    name="greet_agent",  # unique name for the agent
    model=model_name,  # specify an LLM model; this could be OpenAI or Vertex model
    description="An agent that greets the user by name.",
    instruction="You are a greeting agent. When given a name, you respond with a greeting using the greet tool.",
    tools=[greet]  # provide the greet function as one of the agent's tools
)


if __name__ == "__main__":
    # 세션 및 러너 설정
    APP_NAME = "greet_agent"
    USER_ID = "default_user"
    SESSION_ID = "session_1"

    # 인메모리 세션 서비스 생성 및 세션 초기화
    session_service = InMemorySessionService()
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)

    # Runner 생성 (Agent 실행 엔진)
    runner = Runner(agent=simple_agent, app_name=APP_NAME, session_service=session_service)

    print("Running agent greet_agent... (Ctrl+C to exit)")
    while True:
        # 사용자 입력 수신
        user_input = input(">> ")
        # 메시지 객체로 래핑
        content = types.Content(role="user", parts=[types.Part(text=user_input)])
        # 에이전트 실행 및 이벤트 수신
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
        # 최종 응답 이벤트 출력
        for event in events:
            if hasattr(event, 'is_final_response') and event.is_final_response():
                # 메시지 콘텐츠에서 텍스트 추출
                final_text = event.content.parts[0].text
                print(final_text)
                break

