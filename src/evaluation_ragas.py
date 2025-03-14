import os
import json
import numpy as np 
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

def compute_retrieval_metrics(eval_samples):
    """Computes Precision, Recall, F1-score, MAP, and MRR for retrieval performance."""
    
    total_precision = []
    total_recall = []
    total_f1 = []
    total_ap = []  # Average Precision per query
    total_rr = []  # Reciprocal Rank per query
    
    for sample in eval_samples:
        ground_truth = sample["ground_truth"]
        retrieved_docs = sample["retrieved_contexts"]

        # Binary relevance (1 if relevant, 0 otherwise)
        relevance = [1 if ground_truth in doc else 0 for doc in retrieved_docs]

        retrieved_relevant = sum(relevance)  # Count how many retrieved docs are relevant
        total_relevant = 1  # Assuming one correct ground truth per query
        
        precision = retrieved_relevant / len(retrieved_docs) if retrieved_docs else 0
        recall = retrieved_relevant / total_relevant
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mean Average Precision (MAP)
        ap = sum((sum(relevance[:i+1]) / (i+1)) for i in range(len(relevance)) if relevance[i]) / total_relevant if total_relevant > 0 else 0
        
        # Mean Reciprocal Rank (MRR)
        rr = next((1 / (i+1) for i, rel in enumerate(relevance) if rel), 0)

        # Store metrics
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)
        total_ap.append(ap)
        total_rr.append(rr)

    # Compute averages
    avg_precision = sum(total_precision) / len(total_precision)
    avg_recall = sum(total_recall) / len(total_recall)
    avg_f1 = sum(total_f1) / len(total_f1)
    map_score = sum(total_ap) / len(total_ap)
    mrr_score = sum(total_rr) / len(total_rr)

    # Print Results
    print("\n📊 Retrieval Metrics:")
    print(f"🔹 Precision: {avg_precision:.4f}")
    print(f"🔹 Recall: {avg_recall:.4f}")
    print(f"🔹 F1-score: {avg_f1:.4f}")
    print(f"🔹 MAP (Mean Average Precision): {map_score:.4f}")
    print(f"🔹 MRR (Mean Reciprocal Rank): {mrr_score:.4f}")

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "MAP": map_score,
        "MRR": mrr_score
    }


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
    ragas_results = evaluate(dataset, metrics=[context_precision, answer_relevancy, faithfulness, answer_correctness])

    # Convert RAGAS result to mean values
    context_precision_score = np.nanmean(ragas_results["context_precision"])  # Convert list to mean
    answer_relevancy_score = np.nanmean(ragas_results["answer_relevancy"])
    faithfulness_score = np.nanmean(ragas_results["faithfulness"])
    answer_correctness_score = np.nanmean(ragas_results["answer_correctness"])

    print("\n📊 RAGAS Evaluation Results (Batch Mode):")
    print(f"🔹 Context Precision: {context_precision_score:.4f}")
    print(f"🔹 Answer Relevancy: {answer_relevancy_score:.4f}")
    print(f"🔹 Faithfulness: {faithfulness_score:.4f}")
    print(f"🔹 Answer Correctness: {answer_correctness_score:.4f}")
    
    # Run Retriever Evaluation
    retrieval_metrics = compute_retrieval_metrics(eval_samples)
    
    return retrieval_metrics, ragas_results

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
