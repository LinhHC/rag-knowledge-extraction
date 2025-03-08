from dotenv import load_dotenv
import os

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader




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

# Path to test_data folder containing PDFs
pdf_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

# Load all PDFs from the folder
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

# Load documents using PyPDFLoader
docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
response = rag_chain.invoke("What types of gestures are there?")
print(response)