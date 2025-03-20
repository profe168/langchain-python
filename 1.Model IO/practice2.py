from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),model="gpt-4o-mini",temperature=0.5)

messages = [
    (
        "system",
        "あなたは優秀なPythonの専門家です。",
    ),
    ("human", "Pythonとは何ですか？"),
]
ai_msg = llm.invoke(messages)
print(ai_msg)