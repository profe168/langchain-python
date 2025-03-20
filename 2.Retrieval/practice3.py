from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import config


# OpenAIの埋め込みモデルを設定
embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY,model="text-embedding-3-small")

# テキストの埋め込みベクトルを取得
embedding1 = embeddings_model.embed_query("AIはどのように機能しますか？")
embedding2 = embeddings_model.embed_query("人工知能の仕組みを教えてください。")
embedding3 = embeddings_model.embed_query("お天気はどうですか？")

# ベクトルの次元数を確認
print("埋め込みベクトルの次元数:", len(embedding1))

# 埋め込みベクトル間のコサイン類似度を計算
similarity_1_2 = cosine_similarity([embedding1], [embedding2])
similarity_1_3 = cosine_similarity([embedding1], [embedding3])

# 結果を表示
print(f"similarity_1_2:{similarity_1_2}") 
print(f"similarity_1_3:{similarity_1_3}") 