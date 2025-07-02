# db_utils.py — слой данных
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag-documents"
MODEL_NAME = "BAAI/bge-m3"

client = QdrantClient(host="localhost", port=6333)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cuda"})

def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Создание новой коллекции: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    else:
        logger.info(f"Коллекция {COLLECTION_NAME} уже существует")

def add_documents(docs):
    logger.info(f"Начинаем добавление {len(docs)} документов в Qdrant")
    ensure_collection()
    try:
        Qdrant.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url="http://localhost:6333"
        )
        logger.info("Документы успешно добавлены в Qdrant")
    except Exception as e:
        logger.error(f"Ошибка при добавлении документов: {e}")

def get_vectorstore():
    ensure_collection()
    return Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
