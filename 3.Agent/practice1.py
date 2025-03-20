from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from PIL import Image
import matplotlib.pyplot as plt
import io

class State(TypedDict):
    input: str


def step_1(state):
    print("---Step 1---")
    pass


def step_2(state):
    print("---Step 2---")
    pass


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)


# Add
graph = builder.compile()



png_data = graph.get_graph().draw_mermaid_png()
image = io.BytesIO(png_data)

img = Image.open(image)

plt.imshow(img)
plt.axis("off")
plt.show()
