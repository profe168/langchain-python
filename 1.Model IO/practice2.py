from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
    temperature=0.5
)

messages = [
    (
        "system",
        "あなたは優秀なPythonの専門家です。",
    ),
    ("human", "Pythonとは何ですか？"),
]
ai_msg = llm.invoke(messages)
print(ai_msg)