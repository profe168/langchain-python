from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),model="gpt-4o-mini")

messages = [
    (
        "system",
        "あなたは優秀な天気予報士です。",
    ),
    ("human", "明日の東京の天気は？"),
]
ai_msg = llm.invoke(messages)
print(ai_msg)