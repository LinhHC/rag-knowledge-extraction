import json
import os
import numpy as np
from langchain.evaluation import load_evaluator
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Set OpenAI API key (ensure it's correctly set in your .env file)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Please set it in your .env file or environment variables.")

# Set file paths
results_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/base_evaluation_results.json"
output_evaluation_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/base_langchain_context_qa_evaluation.json"
output_metrics_file = "C:/Users/Linh/Desktop/rag-knowledge-extraction/results/base_evaluation_metrics.json"

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
context_qa_evaluator = load_evaluator("context_qa", llm=ChatOpenAI(api_key=openai_api_key, temperature=0))

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

print("\n🔍 Debug: Retrieval Scores Per Query")
for entry in eval_results:
    print(f"Query: {entry['query']}, Retrieval Score: {entry['retrieval_score']}")

# Save LangChain retrieval evaluation results
with open(output_evaluation_file, "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=4)

print(f"\n🔹 LangChain Retrieval Evaluation Completed")
print(f"📂 Results saved to {output_evaluation_file}")

### COMPUTE STANDARD METRICS (Precision, Recall, F1, MRR, MAP) ###

# Extract true labels and predictions
y_true = [1 if entry["reference"] else 0 for entry in eval_samples]
y_pred = [1 if entry["retrieval_score"] > 0 else 0 for entry in eval_results]

print("\n🔍 Debugging Precision Calculation")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")

# Debugging precision calculation
false_negatives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])
false_positives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])
true_positives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1])

print("\n🔍 Debugging Precision Calculation")
print(f"Total Queries: {len(y_true)}")
print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")


# Debugging check
print(f"\nTotal Queries: {len(y_true)}")
print(f"Total Retrieved (y_pred=1): {sum(y_pred)}")
print(f"Total Ground Truth Exists (y_true=1): {sum(y_true)}")

# Compute Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

retrieved_texts = [entry["retrieved"][0] if entry["retrieved"] else "" for entry in evaluation_results]
ground_truth_texts = [entry["ground_truth"] for entry in evaluation_results]


# Function to compute Mean Reciprocal Rank (MRR)
# Load SBERT Model for embedding computation
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def semantic_mrr(retrieved_docs, ground_truth, threshold=0.5):
    reciprocal_ranks = []
    
    for docs, gt in zip(retrieved_docs, ground_truth):
        if not docs or not gt:
            reciprocal_ranks.append(0.0)
            continue

        # Compute embeddings
        retrieved_embeddings = sbert_model.encode([str(doc) for doc in docs], convert_to_tensor=True)
        gt_embedding = sbert_model.encode([str(gt)], convert_to_tensor=True)

        # Compute cosine similarity
        similarities = cosine_similarity(retrieved_embeddings.cpu().numpy(), gt_embedding.cpu().numpy())


        for rank, score in enumerate(similarities[:, 0]):
            if score >= threshold:
                reciprocal_ranks.append(1.0 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)  # No relevant document found

    return np.mean(reciprocal_ranks)




# Function to compute Mean Average Precision (MAP)
def mean_average_precision(y_true, y_pred):
    ap_scores = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            precision_at_i = sum(y_pred[:i+1]) / (i+1)
            ap_scores.append(precision_at_i)
    return np.mean(ap_scores) if ap_scores else 0

# Compute MRR and MAP

mrr = semantic_mrr(retrieved_texts, ground_truth_texts)
map_score = mean_average_precision(y_true, y_pred)



metrics_output = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "mrr": mrr,
    "map": map_score,
}

with open(output_metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_output, f, indent=4)

print(f"🔹 Final Evaluation Metrics:")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-score: {f1:.4f}")
print(f"✅ Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"✅ Mean Average Precision (MAP): {map_score:.4f}")
print(f"📂 Metrics saved to {output_metrics_file}")
