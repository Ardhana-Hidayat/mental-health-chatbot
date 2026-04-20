from chromadb.utils import embedding_functions

def embedding():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")