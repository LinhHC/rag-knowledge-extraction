# **RAG-Based Exam Generator** 📚🤖  

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to extract learning content from **PDF documents** and generate structured multiple-choice questions  using **LLMs**. It supports **query expansion**, **document retrieval**, **reranking**, and **automated question generation**.

---

## **🚀 Installation & Setup**  

### **1️⃣ Create a Virtual Environment**  
To ensure package compatibility, create a virtual environment:  

```bash
python -m venv venv
```

### **2️⃣ Install Dependencies**  
Once the virtual environment is active, install the required dependencies: 

```bash
pip install -r requirements.txt
```

### **3️⃣ Configure API Keys**  
Create a .env file in the project root directory and add the following: 

```env
OPENAI_API_KEY = [ENTER YOUR OPENAI API KEY HERE]
LANGCHAIN_API_KEY = [ENTER YOUR LANGCHAIN API KEY HERE]
```
This ensures secure authentication with OpenAI and LangChain APIs.

### **4️⃣ Insert Data into the /data Folder**  
Place your PDF lecture slides or any educational materials into the /data directory. The system will automatically process these files.
Example file structure:
```kotlin
📂 project_root/
 ┣ 📂 data/
 ┃ ┣ 📄 lecture1.pdf
 ┃ ┣ 📄 lecture2.pdf
 ┃ ┗ 📄 my_notes.pdf
```

### **5️⃣ Run the Main Pipeline**  
To start the exam generation process, execute:
```bash
python src/main.py
```
## **🛠 Project Structure**  
```graphql
📂 project_root/
 ┣ 📂 data/                   # PDFs for content extraction
 ┣ 📂 evaluation/             # Evaluation results and metrics
 ┣ 📂 generated_exams/        # Output MCQ files (JSON format)
 ┣ 📂 results/                # Additional experiment results
 ┣ 📂 src/                    # Source code
 ┃ ┣ 📂 chroma_cache/        # Cached vector embeddings (ignored in Git)
 ┃ ┣ 📜 evaluation.py         # LLM-based exam evaluation
 ┃ ┣ 📜 exam.py               # Question generation logic
 ┃ ┣ 📜 rag_pipeline.py       # Main retrieval-augmented pipeline
 ┃ ┗ 📜 main.py               # Entry point script
 ┣ 📜 .gitignore              # Files and folders ignored in version control
 ┣ 📜 LICENSE                 # Project licensing
 ┣ 📜 README.md               # Documentation (this file)
 ┣ 📜 requirements.txt         # List of dependencies
 ┣ 📜 TODO.txt                # Development notes (ignored in Git)
 ┗ 📜 pipeline.txt             # Pipeline configuration (ignored in Git)

```