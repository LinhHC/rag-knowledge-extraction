from dotenv import load_dotenv
import os
import json
import random
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_chain = retriever | format_docs

# Chain
rag_chain = (
    {"question": RunnablePassthrough(), "context": RunnablePassthrough()}  # Ensure correct passthrough
    | prompt
    | llm
    | StrOutputParser()
)


# Define the retrieval chain


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


    # Generate response from RAG model
    retrieved_docs = retrieval_chain.invoke(query)



    input_payload = {"question": query, "context": retrieved_docs}
    response = rag_chain.invoke(input_payload)
    assert isinstance(response, str), f"Expected response to be string but got {type(response)}"





    evaluation_results.append({
        "query": query,
        "retrieved": retrieved_docs.split("\n\n"),  # Convert back to list format if needed
        "generated": response,
        "ground_truth": ground_truth
    })


# Save intermediate results for evaluation script
results_file = os.path.join(results_folder, "base_evaluation_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Intermediate results saved to {results_file}. Run evaluation.py for scoring.")