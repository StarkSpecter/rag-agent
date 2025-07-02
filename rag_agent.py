# rag_agent.py — слой бизнес-логики (RAG агент)
import time
import logging
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from db_utils import get_vectorstore

logging.basicConfig(
    filename="rag_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query, documents, top_k=3):
        start = time.time()
        pairs = [(query, doc.page_content) for doc in documents]
        inputs = self.tokenizer.batch_encode_plus(pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()
        elapsed = time.time() - start
        logger.info(f"⏱️ Reranking time: {elapsed:.2f} sec")
        return [documents[i] for i in top_indices]

def get_prompt():
    return ChatPromptTemplate.from_template("""
Ты — опытный специалист, ориентирующийся на юридические вопросы. Используй следующие фрагменты документов и изменений и 
ТОЛЬКО их, чтобы дать точный ответ. Отвечай полными предложениями, изложи всю информацию которая есть в контексте и отвечай на русском языке.

Контекст:
{context}

Вопрос:
{question}
""")

def get_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 10})
    reranker = BGEReranker()
    llm = Ollama(model="llama3:8b")
    prompt = get_prompt()

    def rerank_and_format(inputs):
        total_start = time.time()
        query = inputs["question"]

        start = time.time()
        retrieved_docs = retriever.invoke(query)
        elapsed = time.time() - start
        logger.info(f"⏱️ Retrieval time: {elapsed:.2f} sec")

        top_docs = reranker.rerank(query, retrieved_docs)

        context = "\n\n".join(doc.page_content for doc in top_docs)
        formatted_input = {"context": context, "question": query}

        start = time.time()
        response = llm.invoke(prompt.invoke(formatted_input))
        elapsed = time.time() - start
        logger.info(f"⏱️ LLM generation time: {elapsed:.2f} sec")

        total_elapsed = time.time() - total_start
        logger.info(f"⏱️ Total RAG time: {total_elapsed:.2f} sec")

        return f"{response}\n\n---\n📚 Использованный контекст:\n{context}"

    return RunnableLambda(rerank_and_format)