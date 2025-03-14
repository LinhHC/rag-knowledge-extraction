import os
import json
from rag_pipeline import load_environment, load_pdfs_from_folder, create_rag_pipeline
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness, answer_correctness
from datasets import Dataset

# ----------------- LOAD EVALUATION DATA -----------------
def load_evaluation_data(json_path):
    """Loads queries and ground truth answers from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_queries = [entry["question"] for entry in data]
    ground_truth_answers = [entry["answer"] for entry in data]
    
    print(f"\n📂 Loaded {len(user_queries)} evaluation queries from JSON.\n")
    
    return user_queries, ground_truth_answers

def run_evaluation(rag_chain, user_queries, ground_truth_answers, retriever):
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

def main():
    
    load_environment()
    # Define dataset paths
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    json_path = os.path.join(os.path.dirname(__file__), "../evaluation/rephrased_lecture_gt.json")
    
    # Load documents
    docs = load_pdfs_from_folder(data_folder)
 
    rag_chain, retriever = create_rag_pipeline(docs)
    
    # Load evaluation data...
    user_queries, ground_truth_answers = load_evaluation_data(json_path)
    
    # Run evaluation...
    run_evaluation(rag_chain, user_queries, ground_truth_answers, retriever)
    
if __name__ == "__main__":
    main()
