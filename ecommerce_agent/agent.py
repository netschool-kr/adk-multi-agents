import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Vertex AI SDK
import vertexai
# from vertexai.language_models import TextGenerationModel # Commented out or delete as we are using Gemini
from vertexai.generative_models import GenerativeModel # Using GenerativeModel class for Gemini models
vertexai.init(project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])

# Change to the successfully tested Gemini model ID
MODEL_NAME = "gemini-2.0-flash-001"

# Sample product catalog (could be replaced with a database or Google Sheet in a real app)
PRODUCT_CATALOG = [
    {"id": "P1001", "name": "Wireless Mouse", "category": "Electronics", "price": 25.0},
    {"id": "P1002", "name": "Bluetooth Keyboard", "category": "Electronics", "price": 45.0},
    {"id": "P2001", "name": "Running Shoes", "category": "Sportswear", "price": 60.0},
    # ... (additional products)
]

def search_product_catalog(query: str) -> dict:
    """Lookup a product by name or category in the product catalog."""
    query_lower = query.lower()
    for product in PRODUCT_CATALOG:
        # Match by name or category
        if query_lower in product["name"].lower() or query_lower in product["category"].lower():
            return {
                "id": product["id"],
                "name": product["name"],
                "price": product["price"]
            }
    return {}  # return empty dict if not found

INVENTORY_DB = {
    "P1001": 15,   # Wireless Mouse has 15 units in stock
    "P1002": 8,    # Bluetooth Keyboard has 8 units
    "P2001": 0,    # Running Shoes is out of stock
    # ... etc.
}

def check_stock(product_id: str, quantity: int) -> str:
    """Check if the given product_id has at least the requested quantity available in stock."""
    available = INVENTORY_DB.get(product_id, 0)
    if available >= quantity:
        return "In Stock"
    else:
        return "Out of Stock"


import json
from google.adk.tools import ToolContext

def process_order(product_id: str, quantity: int, tool_context: ToolContext) -> str:
    """Simulate order processing and payment, and return a confirmation in JSON format."""
    # In a real integration, here you'd call a payment API (e.g., Stripe) and create an order record.
    # For this demo, we'll just simulate success and generate an order ID.
    import datetime, random
    order_id = f"ORD-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    confirmation = {
        "order_id": order_id,
        "product_id": product_id,
        "quantity": quantity,
        "status": "confirmed",
        "payment": "approved"
    }
    # If we want this tool's output to be the final user-facing answer, we can skip LLM summarization:
    tool_context.actions.skip_summarization = True
    return json.dumps(confirmation)


from google.adk.agents import LlmAgent  # or Agent
from google.adk.tools import FunctionTool
product_tool = FunctionTool(func=search_product_catalog)

product_agent = LlmAgent(
    name="ProductLookupAgent",
    model=MODEL_NAME,
    description="Finds product details (id, name, price) given a product name or category.",
    instruction=(
       "You are a product catalog assistant. You have access to a function called `search_product_catalog` to look up products by name. "
       "Whenever the user asks for product info (price, details, etc.), **use the `search_product_catalog` tool** with the product name to get the data. "
       "Do not answer from memory. After using the tool, respond with the product info in a JSON format like {\"id\": 123, \"name\": \"...\", \"price\": 45.67}."
    ),
    tools=[product_tool],
        output_key="product_info"
)

inventory_tool = FunctionTool(func=check_stock)
inventory_agent = LlmAgent(
    name="InventoryAgent",
    model=MODEL_NAME,
    description="Checks product availability in inventory for a given product and quantity.",
    instruction="""You are an inventory checking agent. Determine if the requested product is available in the desired quantity.
                - You will be given product details and quantity from the previous step, in JSON:
                {product_info}
                - Parse the JSON to get the product_id and quantity.
                - Use the check_stock tool with product_id and quantity to check availability.
                - If the tool returns "In Stock", then output exactly "In Stock".
                - If the tool returns "Out of Stock", then output exactly "Out of Stock".
                - Do not add any other text besides the stock status.
                """,
    tools=[inventory_tool],
    output_key="stock_status"
)

order_tool = FunctionTool(func=process_order)
order_agent = LlmAgent(
    name="OrderAgent",
    model=MODEL_NAME,
    description="Processes orders by confirming stock and completing payment, outputs order confirmation.",
    instruction="""You are an order processing agent. Your job is to finalize the order if possible.
                - You have the product details and the stock status:
                Product: {product_info}
                Availability: {stock_status}
                - If the stock status is "Out of Stock", do NOT call any tool. Just output a brief message: "Unable to complete order: item is out of stock."
                - If the stock status is "In Stock", use the process_order tool with the product_id and quantity to complete the purchase.
                - The tool will return a JSON confirmation. Output that JSON directly as the final answer (do not add extra text).
                """,
    tools=[order_tool]
    # We don't necessarily need an output_key for the final agent, 
    # since its output will be the end result. If we wanted to store it, we could set output_key="order_confirmation".
)

from google.adk.agents import SequentialAgent

workflow_agent = SequentialAgent(
    name="EcommerceWorkflow",
    sub_agents=[product_agent, inventory_agent, order_agent]
)
root_agent = workflow_agent

# Agent execution example (InMemory session and Runner setup)
if __name__ == "__main__":
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types as genai_types # Use alias to differentiate from ADK's potential types

    # google.genai.types 는 이미 genai_types 로 alias 되어 있습니다.
    # from google.genai import types as genai_types
    search_product_catalog("I want to buy 2 units of Wireless")
    session_service = InMemorySessionService()
    session_service.create_session(app_name="ECommerce_app", user_id="user1", session_id="session1")
    runner = Runner(agent=root_agent, app_name="ECommerce_app", session_service=session_service)
    print("Agent running... (Ctrl+C to exit)")
    while True:
        try:
            user_input = input("Enter request: ") or "I want to buy 2 units of Wireless"
            user_content = genai_types.Content(role="user", parts=[genai_types.Part(text=user_input)])
            events = runner.run(user_id="user1", session_id="session1", new_message=user_content)
            
            # Print the final agent response
            for event in events:
                if hasattr(event, "is_final_response") and event.is_final_response():
                    final_answer_text = ""
                    if event.content and event.content.parts:
                        # 최종 응답의 모든 텍스트 부분을 수집하여 결합합니다.
                        text_parts = []
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text: # Part 객체에 text 속성이 있고, 비어있지 않은 경우
                                text_parts.append(part.text)
                        
                        if text_parts:
                            final_answer_text = "".join(text_parts)
                    
                    if final_answer_text:
                        print("Agent:", final_answer_text)
                    else:
                        print("Agent: Final response did not contain any displayable text.")
                        # 디버깅을 위해 원시 응답 내용을 출력해 볼 수 있습니다.
                        # if event.content and hasattr(event.content, 'to_dict'):
                        #     print(f"DEBUG: Raw final event content: {event.content.to_dict()}")
                        # elif event.content:
                        #     print(f"DEBUG: Raw final event content: {event.content}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during agent execution: {e}")
            import traceback
            traceback.print_exc()
