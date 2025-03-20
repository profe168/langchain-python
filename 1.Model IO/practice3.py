from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

llm = ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),model="gpt-4o-mini")

# プロンプトテンプレート
template = "あなたは優秀なPythonの専門家です。次の質問に答えてください。{question}"
prompt = PromptTemplate(input_variables=["question"],template=template)

# 質問を埋め込んでプロンプトを作成する
filled_prompt = prompt.format(question="Pythonとは何ですか？")

response = llm.invoke(filled_prompt)

print(response)