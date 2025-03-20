from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import config


main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

# PDFを読み込み
loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()

# PDFのテキストを結合
whole_document = "".join([page.page_content for page in documents])

# print(documents)
# print(whole_document)

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY,model_name="gpt-4o-mini")

# プロンプトテンプレート
template = """
あなたはpdfドキュメントに基づいて質問に答えるアシスタントです。以下のドキュメントに基づいて質問に答えてください。

ドキュメント：{document}

質問：{question}

答え：

"""

prompt = PromptTemplate(input_variables=["document","question"],template=template)

# チャットボット
def chatbot(question):
    filled_prompt = prompt.format(document=whole_document,question=question)

    response = llm.invoke(filled_prompt)

    return response

question = "LangChain株式会社の最近の業績は？"
response = chatbot(question)
print(response)