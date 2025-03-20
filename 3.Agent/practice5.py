from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from PIL import Image
import matplotlib.pyplot as plt
import io
import config
from serpapi import GoogleSearch


# Define the tools for the agent to use
@tool
def search(query: str):
    """Search the web using SerpAPI."""
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "api_key": config.SERP_API_KEY
    }
    
    search = GoogleSearch(params)
    result = search.get_dict()
    
    results_list = result.get("organic_results", [])
    search_results = [
        f"{res['title']}: {res['snippet']} - {res['link']}" for res in results_list[:3]
    ]
    return search_results if search_results else ["No results found."]


tools = [search]

tool_node = ToolNode(tools)

model = ChatOpenAI(api_key=config.OPENAI_API_KEY,model_name="gpt-4o-mini").bind_tools(tools)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

checkpointer  = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer )

thread = {"configurable":{"thread_id":"42"}}
inputs = [HumanMessage(content="what is the weather in sf")]
for event in app.stream({"messages":inputs},thread,stream_mode="values"):
    event["messages"][-1].pretty_print()
