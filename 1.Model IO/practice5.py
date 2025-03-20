from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# LLMから応答を取得
response = llm.invoke(filled_prompt)

# 応答をStrOutputParserで解析
output_parser = StrOutputParser()
parsed_response = output_parser.invoke(response)

# 解析前と解析後の応答を表示
print("解析前の応答:")
print(response)
print("\n解析後の応答:")
print(parsed_response)

# パイプラインを使用した例
print("\n--- パイプラインを使用した例 ---")

# チェーンの構築（プロンプト -> LLM -> StrOutputParser）
chain = prompt | llm | StrOutputParser()

# チェーンの実行
result = chain.invoke({"question": "Pythonの特徴を3つ挙げてください"})
print(result) 