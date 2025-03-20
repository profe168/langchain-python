from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import os
from dotenv import load_dotenv

# APIキーを環境変数から取得
load_dotenv(dotenv_path="../.env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Google埋め込みモデルを設定
embeddings_model = GoogleGenerativeAIEmbeddings(
    api_key=GOOGLE_API_KEY,
    model="models/embedding-001"
)

# 文書データの設定
documents = [
    Document(page_content="今日は天気が良いですね。公園に行きますか？"),
    Document(page_content="AI技術は近年急速に発展しています。"),
    Document(page_content="コーヒーはとても人気があります。"),
    Document(page_content="pythonはとても便利だ"),
    Document(page_content="javascriptはとても便利だ"),
    Document(page_content="javaはとても便利だ"),
    Document(page_content="c++はとても便利だ"),
    Document(page_content="c#はとても便利だ"),
    Document(page_content="cはとても便利だ"),
    Document(page_content="c++はとても便利だ"),
    Document(page_content="typescriptはとても便利だ"),
]

# FAISSデータベースを作成
db = FAISS.from_documents(documents=documents,embedding=embeddings_model)

query = "AIの発展について教えて"

query_embedding = embeddings_model.embed_query(query)

# 類似度の高い４つのドキュメントを抽出
result = db.similarity_search_by_vector(query_embedding)

for doc in result:
    print(doc)
    doc_embedding = embeddings_model.embed_query(doc.page_content)
    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
    print(f"類似度({doc.page_content}):{similarity}") 
