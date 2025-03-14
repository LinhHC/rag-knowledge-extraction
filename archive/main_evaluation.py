from dotenv import load_dotenv
import os
import json
import random
from sentence_transformers import CrossEncoder
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from operator import itemgetter
from langchain_core.runnables import RunnableLambda



# Load API keys
# Explicitly load the .env file from the outer directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)  # Ensure the correct path is used
openai_api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")

# Enable Tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_PROJECT'] ="RAG"
os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key


#### INDEXING ####

def load_pdfs_from_folder(folder_path):
    """Load PDFs from a specified folder."""
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    return docs

# Define dataset paths
data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Load PDFs from datasets
docs = load_pdfs_from_folder(data_folder)

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  

#### MULTI-QUERY GENERATION ####
# Multi Query: Different Perspectives
query_expansion_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. 
    Original question: {question}"""
)

generate_queries = (
    query_expansion_prompt 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Remove duplicate documents
def get_unique_union(documents: list[list[Document]]):
    """Unique union of retrieved docs while keeping Document objects."""
    seen_contents = set()
    unique_docs = []
    
    for sublist in documents:
        for doc in sublist:
            if isinstance(doc, Document) and doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
    
    return unique_docs

#### RETRIEVAL and GENERATION ####
# Load SBERT reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_with_sbert(input_data):
    query = input_data["question"]
    documents = [doc.page_content for doc in input_data["documents"]]  # Convert to list of strings

    # Prepare query-document pairs
    pairs = [(query, doc) for doc in documents]

    # Compute relevance scores
    scores = reranker.predict(pairs)

    # Sort documents by score (descending order)
    sorted_docs = sorted(zip(input_data["documents"], scores), key=lambda x: x[1], reverse=True)

    # Keep top 5 ranked documents
    reranked_documents = [doc for doc, _ in sorted_docs[:5]]

    return {"question": query, "documents": reranked_documents}

# Wrap in RunnableLambda for LangChain integration
rerank_runnable = RunnableLambda(rerank_with_sbert)

# Modify retrieval chain to use SBERT Reranking
retrieval_chain = (
    generate_queries  # Expand queries
    | retriever.map()  # Retrieve docs
    | get_unique_union  # Remove duplicates
    | (lambda x: {"question": str(x["question"]), "documents": x} if isinstance(x, dict) else {"question": query, "documents": x})  
    | rerank_runnable  # Apply SBERT reranking
    | (lambda x: x["documents"])  # Extract only reranked documents
)

# Answer Generation Prompt
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

# Question
response = rag_chain.invoke("What types of gestures are there?")
print(response)


#### SAVE RESULTS FOR EVALUATION ####
results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(results_folder, exist_ok=True)

# Load evaluation data
eval_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation/rephrased_lecture_gt.json")
with open(eval_file, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Extract the list of Q&A pairs
eval_questions = [{"question": item["question"], "answers": [item["answer"]]} for item in eval_data]

# Sample a subset of questions for evaluation
num_samples = 100  
eval_questions = random.sample(eval_questions, min(num_samples, len(eval_questions)))
num_questions = len(eval_questions)

# Run and save results evaluation
evaluation_results = []
for idx, sample in enumerate(eval_questions, start=1):
    query = sample.get("question", "No question found")  
    answers = sample.get("answers", [])  

    if isinstance(answers, list) and answers:  
        ground_truth = " ".join(answers)
    else:
        ground_truth = "No answer available"

    print(f"Processing query {idx}/{num_questions}...")
    
    retrieved_docs = retrieval_chain.invoke({"question": query})
    response = rag_chain.invoke({"question": query, "context": "\n".join([doc.page_content for doc in retrieved_docs])})
 
    evaluation_results.append({
    "query": query,
    "retrieved": [doc.page_content for doc in retrieved_docs],
    "generated": response,
    "ground_truth": ground_truth
})


# Save intermediate results for evaluation script
results_file = os.path.join(results_folder, "reprhased_evaluation_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Intermediate results saved to {results_file}. Run evaluation.py for scoring.")