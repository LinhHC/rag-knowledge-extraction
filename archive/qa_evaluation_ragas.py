import os
import json
import numpy as np 
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from rag_pipeline import load_environment, load_pdfs_from_folder, create_rag_pipeline
from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness, answer_correctness
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
# from rankeval.metrics import MAP, MRR
# from rankeval.metrics import Precision, Recall, F1

# ----------------- LOAD EVALUATION DATA -----------------
def load_evaluation_data(json_path):
    """Loads queries and ground truth answers from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_queries = [entry["question"] for entry in data]
    ground_truth_answers = [entry["answer"] for entry in data]
    
    print(f"\n📂 Loaded {len(user_queries)} evaluation queries from JSON.\n")
    
    return user_queries, ground_truth_answers

def compute_retrieval_metrics(eval_samples, similarity_threshold=0.7):
    """Computes Precision, Recall, F1-score, MAP, and MRR using precomputed similarity scores from ChromaDB."""
    
    eval_results = []
    relevance_scores = []
    
    context_qa_evaluator = load_evaluator("context_qa", llm=ChatOpenAI(temperature=0))

    for sample in eval_samples:
        # ground_truth = sample["ground_truth"]
        # retrieved_docs = sample["retrieved_contexts"]  # List of retrieved doc contents
        # response = sample["response"]

        result = context_qa_evaluator.evaluate_strings(
            prediction="\n".join(sample["retrieved_contexts"]),  # Retrieved documents
            reference=sample["ground_truth"],  # Ground truth answer
            input=sample["question"]  # Original question
        ) 
        
        eval_results.append({
            "query": sample["question"],
            "retrieval_score": result["score"],
            "feedback": result.get("feedback", "No feedback provided")
        })
        
        # Store binary relevance for MAP & MRR calculations
        retrieved_scores = [1 if result["score"] >= similarity_threshold else 0 for _ in sample["retrieved_contexts"]]
        relevance_scores.append(retrieved_scores)
        
    ### COMPUTE STANDARD METRICS (Precision, Recall, F1, MRR, MAP) ###

    # Extract true labels and predictions
    y_true = [1 if entry["ground_truth"] else 0 for entry in eval_samples]
    y_pred = [1 if entry["retrieval_score"] > 0 else 0 for entry in eval_results]

    # Compute Precision, Recall, and F1-score
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # rank_eval_precision = Precision()(relevance_scores)
    # rank_eval_recall = Recall()(relevance_scores)
    # rank_eval_f1 = F1()(relevance_scores)
    # map_score = MAP()(relevance_scores)
    # mrr_score = MRR()(relevance_scores)    
        
    # Print Results
    print("\n📊 Retrieval Metrics (Using ChromaDB Similarity Scores):")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")
    print(f"🔹 F1-score: {f1:.4f}")
    
    # print(f"🔹 Rank_Eval_Precision: {rank_eval_precision:.4f}")
    # print(f"🔹 Rank_Eval_Recall: {rank_eval_recall:.4f}")
    # print(f"🔹 Rank_Eval_F1-score: {rank_eval_f1:.4f}")
    
    # print(f"🔹 MAP (Mean Average Precision): {map_score:.4f}")
    # print(f"🔹 MRR (Mean Reciprocal Rank): {mrr_score:.4f}")



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
    compute_retrieval_metrics(eval_samples)


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
