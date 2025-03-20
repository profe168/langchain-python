from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from serpapi import GoogleSearch  

import config

# エージェントが使用するツールを定義
@tool
def search(query: str):
    """SerpAPIを使用してウェブ検索を実行します。"""
    params = {
        "q": query,  # 検索クエリ
        "hl": "ja",  # 言語設定（日本語）
        "gl": "jp",  # 地域設定（日本）
        "api_key": config.SERP_API_KEY  # SerpAPIのAPIキー
    }
    
    search = GoogleSearch(params)  # SerpAPIの検索オブジェクトを作成
    result = search.get_dict()  # 検索結果を辞書形式で取得
    
    results_list = result.get("organic_results", [])  # オーガニック検索結果を取得
    search_results = [
        f"{res['title']}: {res['snippet']} - {res['link']}" for res in results_list[:3]
    ]  # 最初の3件の結果をフォーマットしてリストに格納
    print("\n#########【APIでの検索結果】########")  # フォーマットされた検索結果を表示
    print(search_results)
    return search_results if search_results else ["検索結果が見つかりませんでした。"]  # 結果があれば返す。なければ「結果なし」を返す


tools = [search]  # 使用するツールのリスト

tool_node = ToolNode(tools)  # ツールノードを作成

# OpenAIのChatGPTモデルを設定し、ツールをバインドする
model = ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini").bind_tools(tools)

# 次に進むか終了するかを判定する関数を定義
def should_continue(state: MessagesState) -> Literal["tools", END]:
    print("\n##########【should_continue関数内のstate】##########")  # state全体を表示
    print(state)
    messages = state['messages']  # 現在のメッセージ状態を取得
    last_message = messages[-1]  # last_messageを取得
    print("\n##########【last_message】##########")  # last_messageを表示
    print( last_message)  # last_messageを表示
    
    # LLMがツールを呼び出した場合、"tools"ノードに遷移
    if last_message.tool_calls:
        print("\n→ ツールを呼び出し。次のノード: tools")
        return "tools"
    # それ以外の場合、終了 (END)
    print("\n→ ツール呼び出しなし。次のノード: END")
    return END


# モデルを呼び出す関数を定義
def call_model(state: MessagesState):
    print("\n##########【call_model関数内のstate】##########")  # state全体を表示
    print(state)  # state全体を表示
    messages = state['messages']  # 現在のメッセージ状態を取得
    response = model.invoke(messages)  # モデルを呼び出して応答を取得
    print("\n##########【モデルの応答（response）】##########")  # モデルの応答を表示
    print(response)  # モデルの応答を表示
    # 応答をリスト形式で返す
    return {"messages": [response]}


# 新しいワークフロー（グラフ）を定義
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)  # agentノードを追加
workflow.add_node("tools", tool_node)  # toolsノードを追加

# エントリポイントを "agent" に設定
workflow.add_edge(START, "agent")

# 条件付きエッジを追加
workflow.add_conditional_edges(
    "agent",
    should_continue,  # 次のノードを判定する関数を指定
)

# "tools" から "agent" への通常のエッジを追加
workflow.add_edge("tools", 'agent')

# メモリを保存するオブジェクトを設定
checkpointer = MemorySaver()

# ワークフローをコンパイル
app = workflow.compile(checkpointer=checkpointer)

# スレッドIDを設定
thread = {"configurable": {"thread_id": "42"}}
# 入力メッセージを作成
inputs = [HumanMessage(content="東京の天気は？")]

# ワークフローをストリームモードで実行し、応答を逐次処理
for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
    event["messages"][-1].pretty_print()  # 応答を整形して出力
