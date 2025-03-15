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
from langchain_core.runnables import RunnablePassthrough
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
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

# ----------------- DOCUMENT FORMATTING -----------------

def get_unique_union(documents):
    """Removes duplicate documents from multi-query retrieval while maintaining original order."""
    seen_contents = set()
    unique_docs = []

    for doc in documents: 
        if isinstance(doc, Document):
            content = doc.page_content.strip() 
            if content and content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)

    return unique_docs

def format_docs(docs):
    """Ensure proper formatting even if input is empty or incorrectly structured."""
    if not docs:  # Handle empty input
        print("❌ Error: No documents to format.")
        return ""

    if isinstance(docs, list) and docs and isinstance(docs[0], tuple):  
        # Extract only the document part from (doc, score) tuples
        docs = [doc for doc, _ in docs]  

    if isinstance(docs, list) and docs and isinstance(docs[0], str):  
        return "\n\n".join(docs)  

    if isinstance(docs, list) and docs and hasattr(docs[0], "page_content"):  
        return "\n\n".join(doc.page_content for doc in docs)  

    print("❌ Error: Unexpected document format. Returning empty context.")
    return ""  # Return empty string instead of failing


# ----------------- QUERY EXPANSION & RERANKING -----------------

def expand_query(topic):
    """Expands a short topic into multiple detailed queries."""
    query_expansion_prompt = ChatPromptTemplate.from_template(
        """Expand the given topic into multiple detailed queries to improve document retrieval. 
    
        **Topic:** {topic}
        
        Generate **3 alternative queries** that provide different perspectives on the topic.
        The queries should be formulated to retrieve documents covering:
        1️⃣ Core concepts of the topic.
        2️⃣ Practical applications of the topic.
        3️⃣ Challenges or limitations related to the topic.
        
        Provide the queries separated by newlines."""
    )

    query_expander = query_expansion_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    
    expanded_queries = query_expander.invoke({"topic": topic})  # Expands input
    return expanded_queries.split("\n")  # Convert into a list of queries

def normalize_scores(scores):
    """Normalizes CrossEncoder scores to a range of 0 to 1."""
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero
    if max_score == min_score:
        return [0.5] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]

def rerank_with_crossencoder(input_data, threshold=0.6):
    """Reranks retrieved documents with Crossencoder and returns only those above the threshold."""
    
    query_key = "topic"  # Change to "question" if using for Q&A
    if query_key not in input_data or "documents" not in input_data:
        print("❌ Error: Missing 'topic/question' or 'documents' in input_data.")
        return {"documents": []}  

    query = input_data[query_key]
    documents = input_data["documents"]

    if not documents:
        print(f"❌ No documents retrieved for query: {query}")
        return {"documents": []}  

    print(f"🔄 Reranking {len(documents)} retrieved documents for query: {query}")

    # Extract text content for ranking
    document_texts = [doc.page_content if isinstance(doc, Document) else str(doc) for doc in documents]

    if not document_texts:
        print("❌ No valid document texts available for reranking.")
        return {"documents": []}

    # Load CrossEncoder model
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    # Prepare query-document pairs
    pairs = [(query, doc) for doc in document_texts]

    if not pairs:
        print("❌ No valid query-document pairs for reranking.")
        return {"documents": []}

    # Compute similarity scores
    scores = reranker.predict(pairs)

    # Normalize scores using the existing function
    normalized_scores = normalize_scores(scores)

    # Sort documents by descending scores and filter by threshold
    sorted_docs = sorted(zip(documents, normalized_scores), key=lambda x: x[1], reverse=True)
    filtered_docs = [doc for doc, score in sorted_docs if score >= threshold] 


    print(f"✅ {len(filtered_docs)} / {len(documents)} documents passed the threshold.")

    return {"documents": filtered_docs} 



# ----------------- EXAM PIPELINE -----------------

def generate_exam_from_topic(docs, topic):
    """Generates an exam based on the given topic using retrieved documents."""
    
    print(f"📖 Retrieving documents related to the topic: {topic}...")
        
    expanded_queries = expand_query(topic)  # Expand query

    vectorstore = setup_chroma_index(docs, "chroma_cache")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    print(f"🔍 Retrieving documents for expanded queries...")

    retrieved_docs = []
    for query in expanded_queries:
        docs = retriever.invoke(query)
        retrieved_docs.extend(docs)
        
    retrieved_docs = get_unique_union(retrieved_docs) 
  

    print(f"✅ Retrieved {len(retrieved_docs)} unique documents after multi-query expansion.")

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

    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  
    
    
    exam_chain = (
        {"context": RunnableLambda(lambda _: retrieved_docs) 
                    | RunnableLambda(lambda x: {"documents": x, "topic": topic}) 
                    | RunnableLambda(rerank_with_crossencoder) 
                    | (lambda x: x["documents"])  
                    | format_docs,
        "topic": RunnablePassthrough()}  
        | exam_prompt
        | llm
        | StrOutputParser()
    )


    return exam_chain



def save_exam_to_json(exam_output, topic):
    """Parses the LLM-generated structured JSON output and saves it as a JSON file."""
    
    # Debug: Check if the LLM generated any output
    if not exam_output or exam_output.strip() == "":
        print("Error: The LLM did not generate any output. No file will be saved.")
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
        # Ensure `exam_output` is properly formatted as a JSON list
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

       
        print(f"Processed JSON Data:\n{json.dumps(exam_data, indent=4)}")

        # Check if data is valid before writing
        if not exam_data:
            print("Exam data is empty. Nothing to save.")
            return

        # Save JSON file
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(exam_data, file, indent=4, ensure_ascii=False)
        
        print(f"✅ Exam successfully saved as {file_path}")

    except Exception as e:
        print(f"Unexpected error while saving JSON: {e}")