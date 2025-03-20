from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
)

messages = [
    (
        "system",
        "あなたは優秀な天気予報士です。",
    ),
    ("human", "明日の東京の天気は？"),
]
ai_msg = llm.invoke(messages)
print(ai_msg)