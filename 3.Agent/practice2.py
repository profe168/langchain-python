from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image
import matplotlib.pyplot as plt
import io

class State(TypedDict):
    input: str
    step_data: str


def step_1(state: State):
    print("---Step 1---")
    state["step_data"] = "Step1"
    print(f"Step1 -> state:{state}")
    return state


def step_2(state: State):
    print("---Step 2---")
    state["step_data"] = state["step_data"] + " + Step2"
    print(f"Step2 -> state:{state}")
    return state


def step_3(state: State):
    print("---Step 3---")
    state["step_data"] = state["step_data"] + " + Step3"
    print(f"Step3 -> state:{state}")
    return state


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

initial_state ={"input":"初期input","step_data":""}

# Add
graph = builder.compile()
graph.invoke(initial_state)


png_data = graph.get_graph().draw_mermaid_png()
image = io.BytesIO(png_data)

img = Image.open(image)

plt.imshow(img)
plt.axis("off")
plt.show()
