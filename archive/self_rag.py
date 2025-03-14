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
from pydantic import BaseModel, Field

from typing import List
from typing_extensions import TypedDict
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from bert_score import score
import warnings
from contextlib import redirect_stderr
import io

# Suppress warnings\
warnings.filterwarnings("ignore")


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
    """Load PDFs from a specified folder and suppress warnings."""
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    docs = []

    for pdf in pdf_files:
        with io.StringIO() as err, redirect_stderr(err):  # Suppress stderr output
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
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_cache"  # Store embeddings
)
vectorstore.persist()
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

# Deduplication Function
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
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# Answer Generation Prompt
response_prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(temperature=0)

rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")}
    | response_prompt
    | llm
    | StrOutputParser()
)



def compute_bertscore(generated_answer, reference_documents):
    """
    Compute BERTScore between generated text and retrieved documents.

    Args:
        generated_answer (str): LLM-generated response.
        reference_documents (list): List of retrieved document texts.

    Returns:
        dict: Precision, Recall, F1 scores.
    """
    if not reference_documents or not generated_answer.strip():
        print("⚠️ No valid references or generation. Skipping BERTScore.")
        return {"precision": 0, "recall": 0, "f1": 0}

    # ✅ Fix: Join all documents into a single reference
    combined_reference = " ".join([doc.page_content for doc in reference_documents]).strip()

    if not combined_reference:
        print("⚠️ Reference documents are empty. Returning zero scores.")
        return {"precision": 0, "recall": 0, "f1": 0}

    try:
        # ✅ Fix: Remove rescale_with_baseline (it causes negative scores sometimes)
        P, R, F1 = score([generated_answer], [combined_reference], lang="en", rescale_with_baseline=False)

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except Exception as e:
        print(f"⚠️ Error computing BERTScore: {e}")
        return {"precision": 0, "recall": 0, "f1": 0}


# Data model for Query Classification
class QueryRelevance(BaseModel):
    """Binary score to assess if a query is relevant to the document corpus."""
    
    binary_score: str = Field(
        description="Is the question relevant to the available documents? Answer 'yes' or 'no'."
    )

# LLM for classification
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
query_relevance_classifier = llm.with_structured_output(QueryRelevance)

# Classification prompt
system = """You are an AI system that determines whether a user's question is relevant to a set of documents.  
Your job is to decide if the user's question can reasonably be answered using the provided document corpus.  
Return 'yes' if the question is relevant and should proceed to retrieval.  
Return 'no' if the question is unrelated to the document contents and should be rejected."""

query_relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)

query_checker = query_relevance_prompt | query_relevance_classifier

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


#####################################################################################################
#################                             GRAPH                   ###############################
#####################################################################################################

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    
### Nodes


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("⚠️ No documents retrieved! Returning empty generation.")
        return {"documents": documents, "question": question, "generation": "", "bert_score": None}

    try:
        generation = rag_chain.invoke({"context": documents, "question": question})
        print(f"🟢 Generated Answer: {generation}")
    except Exception as e:
        print(f"⚠️ Error during generation: {e}")
        generation = ""


    # ✅ Compute BERTScore safely
    bert_scores = compute_bertscore(generation, documents)
    print(f"🔵 BERTScore - Precision: {bert_scores['precision']:.4f}, Recall: {bert_scores['recall']:.4f}, F1: {bert_scores['f1']:.4f}")

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "bert_score": bert_scores,
    }




def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def check_query_relevance(state):
    """
    Checks if the query is relevant to the document corpus.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state containing a decision key.
    """
    print("---CHECK QUERY RELEVANCE---")
    question = state["question"]

    # Invoke query classifier
    score = query_checker.invoke({"question": question})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: QUESTION IS RELEVANT, PROCEEDING TO RETRIEVAL---")
        return {"question": question, "decision": "relevant"}  # Return dict
    else:
        print("---DECISION: QUESTION IS NOT RELEVANT, EXITING WITH RESPONSE---")
        return {
            "question": question,
            "generation": "I don’t know. This question is outside my knowledge base.",
            "documents": [],
            "decision": "not relevant",
        }


### Edges



def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
def handle_irrelevant_query(state):
    """
    Handles the case where the question is not relevant to the document corpus.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Response indicating the query is out of scope.
    """
    print("---HANDLE IRRELEVANT QUERY---")
    return {
        "question": state["question"],
        "generation": "I don’t know. This question is outside my knowledge base.",
        "documents": []
    }


workflow = StateGraph(GraphState)

# Define the new query classifier node
workflow.add_node("check_query_relevance", check_query_relevance)
workflow.add_node("handle_irrelevant_query", handle_irrelevant_query)

# Define existing nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Modify workflow edges
workflow.add_edge(START, "check_query_relevance")  # Start with relevance check
workflow.add_conditional_edges(
    "check_query_relevance",
    lambda state: state["decision"],  # Fetch decision from state dictionary
    {
        "relevant": "retrieve",  # Proceed to retrieval
        "not relevant": "handle_irrelevant_query",  # Exit early
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)


# Get user input from the command line
user_question = input("Enter your question: ")



# Compile
app = workflow.compile()

# Run
inputs = {"question": user_question}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])