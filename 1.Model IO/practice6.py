from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
import json

# モデルのセットアップ
llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-2.0-flash",
)

# 翻訳用のPydanticモデルを定義
class TranslatedWords(BaseModel):
    english: str = Field(description="英語")
    french: str = Field(description="フランス語")
    chinese: str = Field(description="中国語")

# JSONOutputParserの設定
parser = JsonOutputParser(pydantic_object=TranslatedWords)

# プロンプトテンプレートの作成
template = "指定した言語に翻訳してください。\n{format_instructions}\n{query}"
prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# クエリの作成
query = "こんにちは"
prompt_query = prompt.format(query=query)

# モデルに送信
output = llm.invoke(prompt_query)

# パースして結果を取得
result = parser.invoke(output)

# 出力を確認
print("===== 従来の方法での出力 =====")
print("JSONパーサー前の出力:")
print(output)
print("\nJSONパーサー後の出力:")
print(result)
print("\nJSON形式で整形:")
print(json.dumps(result, ensure_ascii=False, indent=2))

# LCELスタイルのパイプラインを使用した例
print("\n\n===== LCELパイプラインを使用した例 =====")
# チェーンの構築（プロンプト -> LLM -> JsonOutputParser）
chain = prompt | llm | parser

# チェーンの実行
pipeline_result = chain.invoke({"query": "おはようございます"})
print("パイプライン結果:")
print(json.dumps(pipeline_result, ensure_ascii=False, indent=2)) 