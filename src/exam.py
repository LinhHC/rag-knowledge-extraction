import os
from rag_pipeline import load_environment, load_pdfs_from_folder, generate_exam_from_topic, save_exam_to_json

def main():
    """Main function for generating an exam based on a topic using retrieved documents."""
    load_environment()
    
    # Define dataset paths
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_path, "data")
    
    # Load PDFs from the data folder
    docs = load_pdfs_from_folder(data_folder)
    
    # Get topic input from the user
    topic = input("Enter the topic for the exam: ").strip()

    if not topic:
        print("❌ No topic provided. Exiting...")
        return
    
    print(f"📖 Generating exam on topic: {topic}...")

    exam_chain = generate_exam_from_topic(docs, topic)
    exam_output = exam_chain.invoke(topic)

    save_exam_to_json(exam_output, topic)

    #print(f"✅ Exam saved successfully")

if __name__ == "__main__":
    main()
