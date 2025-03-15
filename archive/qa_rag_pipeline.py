import os
import hashlib
import pickle
import io
import json
import re
from contextlib import redirect_stderr
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from operator import itemgetter

# ----------------- CONFIGURATION & SETUP -----------------
def load_environment():
    """Loads API keys from .env file and sets environment variables."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_PROJECT'] = "RAG_exam_generator"
    os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# ----------------- DATA HANDLING -----------------
def load_pdfs_from_folder(folder_path):
    """Loads PDFs from a folder while suppressing warnings."""
    print("\n📂 Extracting PDFs...")
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    docs = []

    for pdf in pdf_files:
        with io.StringIO() as err, redirect_stderr(err):  # Suppress stderr output
            loader = PyPDFLoader(pdf)
            docs.extend(loader.load())

    print(f"✅ Loaded {len(docs)} documents from {len(pdf_files)} PDFs.\n")
    return docs

def compute_dataset_hash(docs):
    """Computes a hash for the dataset based on document contents."""
    hasher = hashlib.md5()
    for doc in docs:
        hasher.update(doc.page_content.encode("utf-8"))
    return hasher.hexdigest()

# ----------------- CHROMA INDEXING & RETRIEVAL -----------------
def setup_chroma_index(docs, chroma_cache_path):
    """Sets up ChromaDB index, either loading from cache or rebuilding."""
    print("🔍 Checking dataset integrity...")
    current_hash = compute_dataset_hash(docs)

    hash_file = os.path.join(chroma_cache_path, "dataset_hash.pkl")
    vectorstore_path = os.path.join(chroma_cache_path, "chroma_db")

    # Try to load previous hash
    previous_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "rb") as f:
            previous_hash = pickle.load(f)

    print("🔍 Checking for existing ChromaDB cache...")
    if os.path.exists(vectorstore_path) and current_hash == previous_hash:
        print("✅ Using cached ChromaDB index.\n")
        return Chroma(persist_directory=vectorstore_path, embedding_function=OpenAIEmbeddings())

    # Rebuild index if dataset has changed
    print("🔄 New data detected. Rebuilding ChromaDB index...\n")

    # Split text into smaller chunks
    print("📂 Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(splits)} text chunks.\n")

    # Embed and store in Chroma
    print("⚡ Processing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=vectorstore_path)
    print("✅ ChromaDB index built successfully.\n")

    # Save new dataset hash
    os.makedirs(chroma_cache_path, exist_ok=True)
    with open(hash_file, "wb") as f:
        pickle.dump(current_hash, f)

    return vectorstore

# ----------------- QUERY EXPANSION & RERANKING -----------------
def get_unique_union(documents):
    """Removes duplicate documents from multi-query retrieval."""
    seen_contents = set()
    unique_docs = []

    for sublist in documents:
        for doc in sublist:
            if isinstance(doc, Document) and doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    return unique_docs

def create_multi_query_pipeline(prompt):
    """Creates a query expansion and retrieval pipeline."""

    return (
        prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

def rerank_with_crossencoder(input_data, exam=False):
    """Reranks retrieved documents with Crossencoder"""
    if exam:
        print("🔄 Reranking the retrieved documents...\n")
        query = input_data["topic"]
        documents = [doc.page_content for doc in input_data["documents"]]

        # Load with croessencoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

        pairs = [(query, doc) for doc in documents]
        scores = reranker.predict(pairs)
        sorted_docs = sorted(zip(input_data["documents"], scores), key=lambda x: x[1], reverse=True)

        return {"topic": query, "documents": [doc for doc, _ in sorted_docs[:5]]}
    
    print("🔄 Reranking the retrieved documents...\n")
    query = input_data["question"]
    documents = [doc.page_content for doc in input_data["documents"]]

    # Load with croessencoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(input_data["documents"], scores), key=lambda x: x[1], reverse=True)

    return {"question": query, "documents": [doc for doc, _ in sorted_docs[:5]]}

# ----------------- RAG PIPELINE CREATION -----------------
def create_rag_pipeline(docs):
    """Creates and returns the RAG pipeline."""
    vectorstore = setup_chroma_index(docs, "chroma_cache")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from a vector 
            database. Provide these alternative questions separated by newlines. 
            Original question: {question}"""
        )
    
    query_pipeline = create_multi_query_pipeline(multi_query_prompt)
    rerank_runnable = RunnableLambda(rerank_with_crossencoder)

    retrieval_chain = (
        query_pipeline
        | retriever.map()
        | get_unique_union
        | (lambda x: {"question": x["question"], "documents": x["documents"]} if isinstance(x, dict) else {"question": "", "documents": x})
        | rerank_runnable
        | (lambda x: x["documents"])
    )

    response_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant answering questions based on the provided context. 
        Follow these instructions carefully:
        
        1️⃣ **Use Only the Top-Ranked Retrieved Context**: Prioritize the most relevant content first.
        2️⃣ **Do Not Add Any External Knowledge**: If the answer is not in the context, say: "The provided information does not contain an answer."
        3️⃣ **Be Clear and Concise**: Answer directly while keeping the response easy to understand.
        4️⃣ **Cite the Most Relevant Context**: Explicitly reference key points from the best-ranked documents.
        
        🔹 **Top-Ranked Retrieved Context**:
        {context}
        
        🔹 **Question**: {question}
        
        **Your Answer**:
        """
    )



    llm = ChatOpenAI(temperature=0)

    rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | response_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
