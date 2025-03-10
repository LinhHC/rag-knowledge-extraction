import json
import os
import numpy as np
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

# Load environment variables
load_dotenv()

# Set OpenAI API key (ensure it's correctly set in your .env file)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Please set it in your .env file or environment variables.")

# Set file paths
results_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/evaluation_results.json"
output_evaluation_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/langchain_context_qa_evaluation.json"
output_metrics_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/evaluation_metrics.json"

# Ensure the evaluation results file exists
if not os.path.exists(results_file):
    raise FileNotFoundError(f"❌ Error: File not found at {results_file}")

# Load the evaluation results file
with open(results_file, "r", encoding="utf-8") as f:
    evaluation_results = json.load(f)

# Prepare data for LangChain evaluation
eval_samples = []
for entry in evaluation_results:
    eval_samples.append({
        "input": entry["query"],  # The question asked
        "prediction": "\n".join(entry["retrieved"]),  # The retrieved documents as a single string
        "reference": entry["ground_truth"]  # The expected correct answer
    })

# Load LangChain's `context_qa` evaluator with explicit LLM
context_qa_evaluator = load_evaluator("context_qa", llm=ChatOpenAI(api_key=openai_api_key))

# Evaluate each sample
eval_results = []
for sample in eval_samples:
    result = context_qa_evaluator.evaluate_strings(
        prediction=sample["prediction"],  # Retrieved documents
        reference=sample["reference"],  # Ground truth answer
        input=sample["input"]  # Original question
    )
    eval_results.append({
        "query": sample["input"],
        "retrieval_score": result["score"],
        "feedback": result.get("feedback", "No feedback provided")
    })

# Save LangChain retrieval evaluation results
with open(output_evaluation_file, "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=4)

print(f"\n🔹 LangChain Retrieval Evaluation Completed")
print(f"📂 Results saved to {output_evaluation_file}")

### COMPUTE STANDARD METRICS (Precision, Recall, F1, MRR, MAP) ###

# Extract true labels and predictions
y_true = [1] * len(eval_results)  # All queries expect at least one relevant document
y_pred = [1 if entry["retrieval_score"] > 0 else 0 for entry in eval_results]  # 1 if retrieved relevant document, else 0

# Compute Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Function to compute Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_pred):
    ranks = [1 / (i + 1) for i, val in enumerate(y_pred) if val == 1]
    return np.mean(ranks) if ranks else 0

# Function to compute Mean Average Precision (MAP)
def mean_average_precision(y_true, y_pred):
    ap_scores = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            precision_at_i = sum(y_pred[:i+1]) / (i+1)
            ap_scores.append(precision_at_i)
    return np.mean(ap_scores) if ap_scores else 0

# Compute MRR and MAP
mrr = mean_reciprocal_rank(y_pred)
map_score = mean_average_precision(y_true, y_pred)

# Save final metrics
metrics_output = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "mrr": mrr,
    "map": map_score
}

with open(output_metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, indent=4)

print("\n🔹 Final Evaluation Metrics:")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"✅ Mean Average Precision (MAP): {map_score:.4f}")

print(f"\n📂 Metrics saved to {output_metrics_file}")
