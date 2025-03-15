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

    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
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
        
        ✏️ **Your Answer**:
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

# ----------------- EXAM PIPELINE -----------------
def retrieve_docs_for_topic(docs, topic):
    """Retrieves relevant documents based on a given topic using a structured RAG pipeline."""
    print(f"📖 Retrieving documents related to the topic: {topic}...")
    
    vectorstore = setup_chroma_index(docs, "chroma_cache")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Custom retrieval prompt tailored for topics
    topic_retrieval_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant retrieving relevant documents for exam creation. 
        Given the topic below, find and return the **most relevant** documents to generate an exam.
        
        **Topic:** {topic}
        
        Only return documents that are directly related to this topic. Ensure that the retrieved content provides clear 
        and factual information useful for generating exam questions."""
    )
    
    query_pipeline = create_multi_query_pipeline(topic_retrieval_prompt)
    # Apply structured retrieval pipeline (similar to standard RAG)
    rerank_runnable = RunnableLambda(lambda input_data: rerank_with_crossencoder(input_data, exam=True))


    topic_query_pipeline = (
        query_pipeline
        | retriever.map()
        | get_unique_union
        | (lambda x: {"topic": x["topic"], "documents": x["documents"]} if isinstance(x, dict) else {"topic": "", "documents": x})
        | rerank_runnable
        | (lambda x: x["documents"])
    )
    
    return topic_query_pipeline


def generate_exam_from_topic(docs, topic):
    """Generates an exam based on the given topic using retrieved documents."""
    
    print(f"📖 Retrieving documents related to the topic: {topic}...")
        
    vectorstore = setup_chroma_index(docs, "chroma_cache")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    # Custom retrieval prompt tailored for topics
    topic_retrieval_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant retrieving relevant documents for exam creation. 
        Given the topic below, find and return the **most relevant** documents to generate an exam.
        
        **Topic:** {topic}
        
        Only return documents that are directly related to this topic. Ensure that the retrieved content provides clear 
        and factual information useful for generating exam questions."""
    )
    
    query_pipeline = create_multi_query_pipeline(topic_retrieval_prompt)
    # Apply structured retrieval pipeline (similar to standard RAG)
    rerank_runnable = RunnableLambda(lambda input_data: rerank_with_crossencoder(input_data, exam=True))


    topic_query_pipeline = (
        query_pipeline
        | retriever.map()
        | get_unique_union
        | (lambda x: {"topic": x["topic"], "documents": x["documents"]} if isinstance(x, dict) else {"topic": "", "documents": x})
        | rerank_runnable
        | (lambda x: x["documents"])
    )

    exam_prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant creating an exam on the topic: **{topic}**.
        Generate **15 multiple-choice questions** using only the given documents.
        Each question should have **four answer options (A, B, C, D)** and indicate the correct answer.
        Also, provide the **document name** for each question.

        **Important Guidelines:**
        - **Do NOT ask questions about the document itself** (e.g., "What does the document mention?" or "In which document...?").  
        - **Do NOT ask questions that require document metadata as an answer** (e.g., "What is the title of the document?" or "On which page is X mentioned?").  
        - **Focus only on the actual content within the documents** to create meaningful exam questions.  
        - **Ensure the correct answer is factually supported by the retrieved content.**  

        **Format (must be a valid JSON list):**  
        
        [
            {{
                "Question_ID": 1,  
                "Question": "<The generated exam question>",  
                "Answer_Options": {{
                    "A": "<Answer Option A>",
                    "B": "<Answer Option B>",
                    "C": "<Answer Option C>",
                    "D": "<Answer Option D>"
                }},
                "Correct_Answer": {{
                    "<Correct Option Letter>": "<Correct Answer Text>"
                }},
                "Document": "<Source document name>"
            }},
            {{
                "Question_ID": 2,  
                "Question": "<Another generated question>",  
                "Answer_Options": {{
                    "A": "<Answer Option A>",
                    "B": "<Answer Option B>",
                    "C": "<Answer Option C>",
                    "D": "<Answer Option D>"
                }},
                "Correct_Answer": {{
                    "<Correct Option Letter>": "<Correct Answer Text>"
                }},
                "Document": "<Source document name>"
            }}
        ]
        
        **Rules for JSON Formatting:**  
        - `Answer_Options` must be a dictionary containing exactly **four** keys (`A`, `B`, `C`, `D`).  
        - `Correct_Answer` must be a **dictionary with a single key** corresponding to the correct answer letter.  
        - The response must be a **valid JSON list** containing exactly **15 questions**.  
        - Do **NOT** include any additional explanations, text, or formatting.  

        **Context (Use only this for generating questions):**  
        {context}

        Generate a total of **15 questions**, ensuring they are **clear, unbiased, and related to the topic.**"""
    )



    
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)  
    
    # ✅ Fix: Ensure `topic` is passed into the prompt
    exam_chain = (
        {"context": topic_query_pipeline, "topic": itemgetter("topic")}  
        | exam_prompt
        | llm
        | StrOutputParser()
    )

    return exam_chain



def save_exam_to_json(exam_output, topic):
    """Parses the LLM-generated structured JSON output and saves it as a JSON file."""
    
    # Debug: Check if the LLM generated any output
    if not exam_output or exam_output.strip() == "":
        print("❌ Error: The LLM did not generate any output. No file will be saved.")
        return  # Exit early if output is empty
    
    # Ensure the directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated_exams")
    os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    
    # Normalize topic name (replace spaces with underscores, lowercase)
    topic_clean = topic.replace(" ", "_").lower()

    # Get list of existing exam files
    existing_files = [f for f in os.listdir(output_dir) if re.match(r"^\d+_" + re.escape(topic_clean) + r"_exam\.json$", f)]
    
    # Determine the next number in sequence
    exam_number = len(existing_files) + 1
    filename = f"{exam_number}_{topic_clean}_exam.json"


    # Set full file path
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Fix: Ensure `exam_output` is properly formatted as a JSON list
        if isinstance(exam_output, str):
            try:
                exam_data = json.loads(exam_output)  # Convert string to Python dictionary/list
                print("Successfully parsed as JSON string")
            except json.JSONDecodeError as e:
                print(f"❌ JSON Decoding Error: {e}")
                print("Raw output:", exam_output)
                return  
        else:
            exam_data = exam_output  # If already JSON, use as is
            print("Already a JSON object")

        # Ensure it's a list (LLM might return a single dictionary instead of a list)
        if isinstance(exam_data, dict):
            exam_data = [exam_data]  # Convert single dict to list

        # Debug: Print processed data before saving
        print(f"✅ Processed JSON Data:\n{json.dumps(exam_data, indent=4)}")

        # Check if data is valid before writing
        if not exam_data:
            print("Exam data is empty. Nothing to save.")
            return

        # Save JSON file
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(exam_data, file, indent=4, ensure_ascii=False)
        
        print(f"Exam successfully saved as {file_path}")

    except Exception as e:
        print(f"Unexpected error while saving JSON: {e}")