from langchain_community.document_loaders import PyPDFLoader
import os

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()
print(documents)