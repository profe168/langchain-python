from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from PIL import Image
import matplotlib.pyplot as plt
import io
import config
from serpapi import GoogleSearch
import os

# APIキーの環境変数設定
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["SERPAPI_API_KEY"] = config.SERP_API_KEY

# MessagesStateの定義（元々はインポートされていたが、独自に定義）
class MessagesState(TypedDict):
    messages: list


# エージェントが使用するツールを定義
@tool
def search(query: str):
    """SerpAPIを使用してウェブを検索します。"""
    params = {
        "q": query,
        "hl": "ja",
        "gl": "jp",
        "api_key": config.SERP_API_KEY
    }
    
    search = GoogleSearch(params)
    result = search.get_dict()
    
    results_list = result.get("organic_results", [])
    search_results = [
        f"{res['title']}: {res['snippet']} - {res['link']}" for res in results_list[:3]
    ]
    return search_results if search_results else ["検索結果が見つかりませんでした。"]


tools = [search]

tool_node = ToolNode(tools)

model = ChatOpenAI(api_key=config.OPENAI_API_KEY,model_name="gpt-4o-mini").bind_tools(tools)

# 続行するかどうかを決定する関数を定義
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # LLMがツールを呼び出す場合、"tools"ノードに進む
    if last_message.tool_calls:
        return "tools"
    # それ以外の場合は停止（ユーザーに返信）
    return END


# モデルを呼び出す関数を定義
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # リストを返します（既存のリストに追加されるため）
    return {"messages": [response]}


# 新しいグラフを定義
workflow = StateGraph(MessagesState)

# 循環する2つのノードを定義
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# エントリーポイントを`agent`に設定
# これはこのノードが最初に呼び出されることを意味します
workflow.add_edge(START, "agent")

# 条件付きエッジを追加
workflow.add_conditional_edges(
    # 最初に開始ノードを定義します。`agent`を使用します。
    # これは`agent`ノードが呼び出された後に進むエッジを意味します。
    "agent",
    # 次に、次に呼び出されるノードを決定する関数を渡します。
    should_continue,
)

# `tools`から`agent`への通常のエッジを追加
# これは`tools`が呼び出された後、次に`agent`ノードが呼び出されることを意味します
workflow.add_edge("tools", 'agent')

checkpointer  = MemorySaver()

# 最後にコンパイル！
# これはLangChain Runnableにコンパイルされ、
# 他のRunnableと同様に使用できます。
# （オプションで）グラフをコンパイルするときにメモリを渡していることに注意してください
app = workflow.compile(checkpointer=checkpointer )

thread = {"configurable":{"thread_id":"42"}}
inputs = [HumanMessage(content="サンフランシスコの天気はどうですか")]
for event in app.stream({"messages":inputs},thread,stream_mode="values"):
    event["messages"][-1].pretty_print()
