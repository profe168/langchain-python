from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
)

# プロンプトテンプレート
template = "あなたは優秀なPythonの専門家です。次の質問に答えてください。{question}"
prompt = PromptTemplate(input_variables=["question"],template=template)

# 質問を埋め込んでプロンプトを作成する
filled_prompt = prompt.format(question="Pythonとは何ですか？")

response = llm.invoke(filled_prompt)

print(response)