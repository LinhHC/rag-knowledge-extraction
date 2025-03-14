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


# ----------------- MULTI-QUERY GENERATION -----------------
def get_unique_union(documents: list[list[Document]]):
    """Removes duplicate documents from multi-query retrieval."""
    seen_contents = set()
    unique_docs = []

    for sublist in documents:
        for doc in sublist:
            if isinstance(doc, Document) and doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

    return unique_docs


def create_query_pipeline():
    """Creates a query expansion and retrieval pipeline."""
    query_expansion_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant. Your task is to generate three
        different versions of the given user question to retrieve relevant documents from a vector 
        database. Provide these alternative questions separated by newlines. 
        Original question: {question}"""
    )

    return (
        query_expansion_prompt
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )


# ----------------- RETRIEVAL & RESPONSE GENERATION -----------------
def generate_response(rag_chain, user_query):
    """Handles user queries and generates responses."""
    print(f"\n🔍 Query received: {user_query}")
    print("🔄 Generating alternative queries...")
    print("📂 Retrieving relevant documents...\n")

    response = rag_chain.invoke({"question": user_query})

    print("⚡ Processing response...\n")
    print(f"📢 Answer: {response}\n")


# ----------------- MAIN EXECUTION -----------------
def main():
    """Main function to initialize and run the pipeline."""
    load_environment()

    # Define dataset paths
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    chroma_cache_path = os.path.join(base_path, "chroma_cache")

    # Load PDFs
    docs = load_pdfs_from_folder(data_folder)

    # Setup ChromaDB
    vectorstore = setup_chroma_index(docs, chroma_cache_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create query expansion and retrieval pipeline
    query_pipeline = create_query_pipeline()
    retrieval_chain = query_pipeline | retriever.map() | get_unique_union

    # Answer Generation
    response_prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on this context:

        {context}

        Question: {question}"""
    )

    llm = ChatOpenAI(temperature=0)

    rag_chain = (
        {"context": retrieval_chain, "question": itemgetter("question")}
        | response_prompt
        | llm
        | StrOutputParser()
    )

    # Process User Query
    user_query = "What is machine learning?"
    generate_response(rag_chain, user_query)


if __name__ == "__main__":
    main()
