from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
import config

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

# PDFを読み込み
loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()

# OpenAIの埋め込みモデルを設定
embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY,model="text-embedding-3-small")


# Chromaデータベースを作成
db = Chroma.from_documents(documents=documents,embedding=embeddings_model)

# OpenAIのLLM設定
llm = ChatOpenAI(api_key=config.OPENAI_API_KEY,model_name="gpt-4o-mini")

# プロンプトテンプレート
template = """
あなたはpdfドキュメントに基づいて質問に答えるアシスタントです。以下のドキュメントに基づいて質問に答えてください。

ドキュメント：{document_snippet}

質問：{question}

答え：

"""

prompt = PromptTemplate(input_variables=["document_snippet","question"],template=template)

# チャットボット
def chatbot(question):
    question_embedding = embeddings_model.embed_query(question)
    document_snippet = db.similarity_search_by_vector(question_embedding,k=3)
    print(f"document_snippet:{document_snippet}")
    filled_prompt = prompt.format(document_snippet=document_snippet,question=question)
    response = llm.invoke(filled_prompt)

    return response

question = "LangChain株式会社の最近の業績は？"
response = chatbot(question)

print(f"response:{response}")