import os
from rag_pipeline import load_environment, load_pdfs_from_folder, create_rag_pipeline

def main():
    load_environment()
    
    # Define dataset paths
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    
    docs = load_pdfs_from_folder(data_folder)
    rag_chain, _ = create_rag_pipeline(docs)

    user_query = input("Enter your question: ")
    print(rag_chain.invoke({"question": user_query}))

if __name__ == "__main__":
    main()
