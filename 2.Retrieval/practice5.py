from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# APIキーを環境変数から取得
load_dotenv(dotenv_path="../.env")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

# PDFを読み込み
loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()

# Googleの埋め込みモデルを設定
embeddings_model = GoogleGenerativeAIEmbeddings(
    api_key=GOOGLE_API_KEY,
    model="models/embedding-001"
)


# FAISSデータベースを作成
db = FAISS.from_documents(documents=documents,embedding=embeddings_model)

# GoogleのLLM設定
llm = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.0-flash"
)

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
    
    # 関連ドキュメントの表示を改善
    print("\n---------- 関連ドキュメント ----------")
    for i, doc in enumerate(document_snippet, 1):
        content = doc.page_content.strip()
        print(f"[文書 {i}] {content[:100]}..." if len(content) > 100 else f"[文書 {i}] {content}")
    print("------------------------------------\n")
    
    filled_prompt = prompt.format(document_snippet=document_snippet,question=question)
    response = llm.invoke(filled_prompt)

    return response

def format_response(text):
    """回答テキストを整形する"""
    # 段落に分割
    paragraphs = text.split('\n')
    # 空の段落を削除
    paragraphs = [p for p in paragraphs if p.strip()]
    # 整形済みテキスト
    formatted = ""
    
    for p in paragraphs:
        # 段落をインデントして追加
        formatted += f"    {p}\n\n"
    
    return formatted.strip()

question = "LangChain株式会社の事業内容は？"
response = chatbot(question)

# 出力を読みやすく修正
print("\n============== 回答 ==============")
formatted_content = format_response(response.content)
print(formatted_content)
print("==================================\n")