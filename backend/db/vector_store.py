import os
from typing import List
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Persist Qdrant locally in the root project folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QDRANT_PATH = os.path.join(ROOT_DIR, "qdrant_data")
COLLECTION_NAME = "pdf_chat_collection"

# Singleton for embeddings
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a singleton of the HuggingFace embeddings model to save memory."""
    global _embeddings
    if _embeddings is None:
        print("Loading HuggingFace Embeddings: all-MiniLM-L6-v2...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Can update to cuda if desired
            encode_kwargs={'normalize_embeddings': True} # Essential for cosine similarity
        )
    return _embeddings

def init_qdrant_client() -> QdrantClient:
    """Initialize the Qdrant local client and ensure collection exists."""
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists; if not, create it
    collections = [col.name for col in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating new Qdrant collection: {COLLECTION_NAME}")
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    return client

def get_vector_store() -> QdrantVectorStore:
    """Returns the Qdrant vector store instance attached to the local DB."""
    client = init_qdrant_client()
    embeddings = get_embeddings()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

def add_documents_to_db(documents: List[Document]):
    """Adds processed documents to the Qdrant database."""
    print(f"Inserting {len(documents)} logic chunks into Qdrant...")
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    print("Insertion complete.")
