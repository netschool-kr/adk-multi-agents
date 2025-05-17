from dotenv import load_dotenv
load_dotenv()
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
MODEL_NAME = "gemini-2.0-flash"
from google.genai import types

MY_LLM_MODEL = "gemini-2.0-flash"

# A mock product database (each product has a name, brand, price, and category/use-case tag)
PRODUCT_DB = [
    {"name": "Budget Gamer 3000", "brand": "Dell", "price": 800, "category": "gaming",
     "description": "entry-level gaming laptop with a mid-range GPU suitable for 1080p gaming"},
    {"name": "OfficePro 15", "brand": "Dell", "price": 900, "category": "office",
     "description": "15-inch productivity laptop with long battery life and lightweight design"},
    {"name": "Ultimate Gamer X", "brand": "Alienware", "price": 1500, "category": "gaming",
     "description": "high-end gaming laptop with an RTX series GPU for maximum performance"},
    {"name": "Ultrabook Air", "brand": "Apple", "price": 1200, "category": "office",
     "description": "premium ultrabook with sleek design, ideal for business and travel"}
]

# Sub-agent to extract budget
budget_agent = LlmAgent(
    name="BudgetAgent",
    model=MY_LLM_MODEL,
    description="Extracts the user's budget from the request.",
    instruction="Determine the user's budget in USD (numerical value) from the query. If the user says 'under $X', 'X dollars', or uses words like 'cheap' or 'affordable', convert that to an approximate number. Output only the number.",
    output_key="budget"
)

# Sub-agent to extract use case
use_case_agent = LlmAgent(
    name="UseCaseAgent",
    model=MY_LLM_MODEL,
    description="Identifies the primary use-case from the request.",
    instruction="Determine what the laptop will primarily be used for, based on the user's request. Respond with a single category: e.g., 'gaming', 'office', 'school', 'video editing', etc.",
    output_key="use_case"
)

# Sub-agent to compose the final recommendation message
final_answer_agent = LlmAgent(
    name="FinalAnswerAgent",
    model=MY_LLM_MODEL,
    description="Generates a final recommendation message for the user.",
    instruction=(
        "You are a helpful sales assistant recommending a laptop based on the user's needs.\n\n"
        "User's Budget: ${budget} USD\n"
        "Primary Use Case: {use_case}\n"
        "Brand Preference: {brand}\n"
        "Recommended Product: {selected_product}\n"
        "Product Description: {product_description}\n\n"
        "Write a concise and natural response recommending the product. Explain how it fits the user's budget, use case, and brand preference (if specified). Include the product description to highlight its features. If no brand preference is specified, note that the recommendation is based on budget and use case. If no product is found, explain why and suggest adjusting criteria."
    ),
    output_key="final_answer"
)

from google.adk.agents import BaseAgent

class ProductRecommendationAgent(BaseAgent):
    
    def __init__(self, budget_agent, use_case_agent, final_agent):
        super().__init__(
            name="ProductRecommendationAgent",
            sub_agents=[budget_agent, use_case_agent, final_agent]
        )

    async def _run_async_impl(self, ctx):
        """Orchestrates the sub-agents and decision logic to produce a product recommendation."""
        # Step 1: Run the BudgetAgent to get the budget
        async for event in self.sub_agents[0].run_async(ctx):
            yield event
        # Step 2: Run the UseCaseAgent to get the use case
        async for event in self.sub_agents[1].run_async(ctx):
            yield event

        # Step 3: Determine brand preference using deterministic logic
        user_query = ctx.user_content.parts[0].text
        known_brands = ["Dell", "Apple", "HP", "Lenovo", "Asus", "Acer", "Alienware"]
        brand = None
        for b in known_brands:
            if b.lower() in user_query.lower():
                brand = b
                break
        ctx.session.state["brand"] = brand if brand else "None"

        # Retrieve the parsed budget and use_case from state
        budget_str = ctx.session.state.get("budget")
        use_case = ctx.session.state.get("use_case")
        try:
            budget_val = float(budget_str) if budget_str is not None else None
        except ValueError:
            budget_val = None

        # Step 4: Filter the product database based on criteria
        candidates = []
        for product in PRODUCT_DB:
            if budget_val is not None and product["price"] > budget_val:
                continue
            if use_case and product["category"].lower() != use_case.lower():
                continue
            if brand and product["brand"].lower() != brand.lower():
                continue
            candidates.append(product)

        # Fallback: Relax use_case filter if no candidates found
        if not candidates:
            for product in PRODUCT_DB:
                if budget_val is not None and product["price"] > budget_val:
                    continue
                if brand and product["brand"].lower() != brand.lower():
                    continue
                candidates.append(product)

        # Choose one product to recommend
        if candidates:
            selected_product = max(candidates, key=lambda p: p["price"])
        else:
            selected_product = None

        # Step 5: Prepare information for the FinalAnswerAgent
        if selected_product:
            product_name = f"{selected_product['brand']} {selected_product['name']}"
            product_price = selected_product["price"]
            ctx.session.state["selected_product"] = f"{product_name} (${product_price})"
            ctx.session.state["product_description"] = selected_product["description"]
        else:
            ctx.session.state["selected_product"] = "No matching product"
            ctx.session.state["product_description"] = ""

        # Step 6: Run the FinalAnswerAgent to generate the output message
        async for event in self.sub_agents[2].run_async(ctx):
            yield event

# Instantiate the agent with the sub-agents
recommendation_agent = ProductRecommendationAgent(budget_agent, use_case_agent, final_answer_agent)
root_agent = recommendation_agent

if __name__ == "__main__":
    APP_NAME = "order_notebook"
    USER_ID = "default_user"
    SESSION_ID = "session_1"
    session_service = InMemorySessionService()
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(agent=recommendation_agent, app_name=APP_NAME, session_service=session_service)

    print("Running Product Recommendation Agent... (Ctrl+C to exit)")
    while True:
        try:
            user_input = input("User Input: ")
            content = types.Content(role="user", parts=[types.Part(text=user_input)])
            events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
            # Debugging: Print all events to inspect their structure
            for event in events:
                print(f"DEBUG: Event - is_final_response: {hasattr(event, 'is_final_response') and event.is_final_response()}, Content: {event.content.parts[0].text if event.content.parts else 'No content'}")
                # Only print the final response from the FinalAnswerAgent
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    final_text = event.content.parts[0].text
                    print("Final Response:", final_text)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
