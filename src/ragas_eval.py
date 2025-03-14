# Standard Library Imports
import os
import hashlib
import pickle
import io
from contextlib import redirect_stderr

# Third-Party Imports
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain import hub
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from operator import itemgetter
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness, answer_correctness
from datasets import Dataset

# ----------------- CONFIGURATION & SETUP -----------------
def load_environment():
    """Loads API keys from .env file and sets environment variables."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGSMITH_PROJECT'] = "RAG"
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
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


# ----------------- RETRIEVAL & RESPONSE GENERATION -----------------
def generate_response(rag_chain, user_query, ground_truth_answer, retriever):
    """Handles user queries, generates responses, and evaluates them."""

    print(f"\n🔍 Query received: {user_query}")
    
    response = rag_chain.invoke({"question": user_query})
    
    print("📢 Answer:", response)
    
    # Retrieve documents using the retriever
    retrieved_docs = retriever.invoke(user_query)

    # RAGAS Evaluation
    eval_sample = {
        "question": user_query,
        "ground_truth": ground_truth_answer,
        "response": response,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs] 
    }

    dataset = Dataset.from_list([eval_sample])
    
    # Run RAGAS Evaluation
    results = evaluate(dataset, metrics=[context_precision, answer_relevancy, faithfulness, answer_correctness])
    
    print("\n📊 RAGAS Evaluation Results:")
    print(results)


# ----------------- MAIN EXECUTION -----------------
def main():
    """Main function to initialize and run the pipeline."""
    load_environment()

    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    chroma_cache_path = os.path.join(base_path, "chroma_cache")

    docs = load_pdfs_from_folder(data_folder)
    vectorstore = setup_chroma_index(docs, chroma_cache_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

    response_prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on this context:

        {context}

        Question: {question}"""
    )

    llm = ChatOpenAI(temperature=0)

    rag_chain = (
        RunnableLambda(lambda x: {"context": retriever.invoke(x["question"]), "question": x["question"]})
        | response_prompt
        | llm
        | StrOutputParser()
    )

    user_query = "What is machine learning?"
    ground_truth_answer = "Machine learning is a method of data analysis that automates analytical model building."

    generate_response(rag_chain, user_query, ground_truth_answer, retriever)


if __name__ == "__main__":
    main()
