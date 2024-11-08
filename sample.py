import vertexai
import streamlit as st
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, Content, ChatSession
from services.flight_manager import search_flights

project = "gemini-explorer-428822"
vertexai.init(project = project)

# Define Tool
get_search_flights = generative_models.FunctionDeclaration(
    name="get_search_flights",
    description="Tool for searching a flight with origin, destination, and departure date",
    parameters={
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "The airport of departure for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "destination": {
                "type": "string",
                "description": "The airport of destination for the flight given in airport code such as LAX, SFO, BOS, etc."
            },
            "departure_date": {
                "type": "string",
                "format": "date",
                "description": "The date of departure for the flight in YYYY-MM-DD format"
            },
        },
        "required": [
            "origin",
            "destination",
            "departure_date"
        ]
    },
)

# Define tool and model with tools
search_tool = generative_models.Tool(
    function_declarations=[get_search_flights],
)

config = generative_models.GenerationConfig(temperature=0.4)
# Load model with config
model = GenerativeModel(
    "gemini-pro",
    tools = [search_tool],
    generation_config = config
)

# Helper function to unpack responses
def handle_response(response):
    # Check if candidates exist
    if response.candidates:
        first_candidate = response.candidates[0]
        # Check if the first candidate has content parts
        if first_candidate.content.parts:
            first_part = first_candidate.content.parts[0]
            # Check if function_call exists and has args
            if hasattr(first_part, 'function_call') and first_part.function_call is not None:
                if first_part.function_call.args:
                    # Function call exists, unpack and load into a function
                    response_args = first_part.function_call.args
                    function_params = {key: response_args[key] for key in response_args}
                    
                    results = search_flights(**function_params)
                    
                    if results:
                        intermediate_response = chat.send_message(
                            Part.from_function_response(
                                name="get_search_flights",
                                response=results
                            )
                        )
                        return intermediate_response.candidates[0].content.parts[0].text
                    else:
                        return "Search Failed"
                else:
                    return "Function call has no arguments."
            else:
                # Return just text if no function call
                return first_part.text
    return "No valid response received."

# helper function to display and send streamlit messages
def llm_function(chat: ChatSession, query):
    response = chat.send_message(query)
    output = handle_response(response)
    
    with st.chat_message("model"):
        st.markdown(output)
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )
    st.session_state.messages.append(
        {
            "role": "model",
            "content": output
        }
    )

st.title("Gemini Flights")

chat = model.start_chat()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display and load to chat history
for index, message in enumerate(st.session_state.messages):
    content = Content(
            role = message["role"],
            parts = [ Part.from_text(message["content"]) ]
        )
    
    if index != 0:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chat.history.append(content)

# For Initial message startup
if len(st.session_state.messages) == 0:
    # Invoke initial message
    initial_prompt = "Introduce yourself as a flights management assistant, ReX, powered by Google Gemini and designed to search/book flights. You use emojis to be interactive. For reference, the year for dates is 2024"

    llm_function(chat, initial_prompt)

# For capture user input
query = st.chat_input("Gemini Flights")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    llm_function(chat, query)
