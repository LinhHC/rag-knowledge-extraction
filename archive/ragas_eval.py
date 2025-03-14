# Standard Library Imports
import os
import json
import hashlib
import pickle
import io
from contextlib import redirect_stderr

# Third-Party Imports
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
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


# ----------------- LOAD EVALUATION DATA -----------------
def load_evaluation_data(json_path):
    """Loads queries and ground truth answers from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_queries = [entry["question"] for entry in data]
    ground_truth_answers = [entry["answer"] for entry in data]
    
    print(f"\n📂 Loaded {len(user_queries)} evaluation queries from JSON.\n")
    
    return user_queries, ground_truth_answers


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


# ----------------- CHROMA INDEXING & RETRIEVAL -----------------
def setup_chroma_index(docs, chroma_cache_path):
    """Sets up ChromaDB index, either loading from cache or rebuilding."""
    print("🔍 Checking for existing ChromaDB cache...")
    vectorstore_path = os.path.join(chroma_cache_path, "chroma_db")

    if os.path.exists(vectorstore_path):
        print("✅ Using cached ChromaDB index.\n")
        return Chroma(persist_directory=vectorstore_path, embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"))

    # Rebuild index if dataset has changed
    print("🔄 No cache found. Rebuilding ChromaDB index...\n")

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embed and store in Chroma
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-ada-002"), persist_directory=vectorstore_path)
    print("✅ ChromaDB index built successfully.\n")

    return vectorstore


# ----------------- RETRIEVAL & RESPONSE GENERATION -----------------
def generate_response(rag_chain, user_queries, ground_truth_answers, retriever):
    """Handles batch evaluation for multiple queries."""
    
    eval_samples = []
    
    for user_query, ground_truth_answer in zip(user_queries, ground_truth_answers):

        response = rag_chain.invoke({"question": user_query})

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_query)

        # Collect evaluation sample
        eval_samples.append({
            "question": user_query,
            "ground_truth": ground_truth_answer,
            "response": response,
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs]
        })

    # Convert list to RAGAS-compatible dataset
    dataset = Dataset.from_list(eval_samples)

    # Run RAGAS Evaluation
    results = evaluate(dataset, metrics=[context_precision, answer_relevancy, faithfulness, answer_correctness])

    print("\n📊 RAGAS Evaluation Results (Batch Mode):")
    print(results)


# ----------------- MAIN EXECUTION -----------------
def main():
    """Main function to initialize and run the pipeline."""
    load_environment()

    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    chroma_cache_path = os.path.join(base_path, "chroma_cache")

    # ✅ Load evaluation queries & answers from JSON
    json_path = os.path.join(os.path.dirname(__file__), "../evaluation/rephrased_lecture_gt.json")
    user_queries, ground_truth_answers = load_evaluation_data(json_path)

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

    # Run batch evaluation
    generate_response(rag_chain, user_queries, ground_truth_answers, retriever)


if __name__ == "__main__":
    main()
