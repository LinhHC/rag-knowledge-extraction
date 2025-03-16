# **RAG-Based Exam Generator** 📚🤖  

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to extract learning content from **PDF documents** and generate structured multiple-choice questions  using **LLMs**. It supports **query expansion**, **document retrieval**, **reranking**, and **automated question generation**.

---

## **🚀 Installation & Setup**  

### **1️⃣ Create a Virtual Environment**  
- Run the following command to create a virtual environment:  

    ```bash
    python -m venv venv
    ```

- Activate the environment:  
    - **Windows (CMD / PowerShell):**  
        ```bash
        venv\Scripts\activate
        ```
    - **MacOS/Linux:**  
        ```bash
        source venv/bin/activate
        ```

---

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
 ┣ 📂 src/                    # Source code
 ┃ ┣ 📂 chroma_cache/        # Cached vector embeddings (ignored in Git)
 ┃ ┣ 📜 evaluation.py         # LLM-based exam evaluation
 ┃ ┣ 📜 rag_pipeline.py       # Main retrieval-augmented pipeline
 ┃ ┗ 📜 main.py               # Entry point script
 ┣ 📜 .gitignore              # Files and folders ignored in version control
 ┣ 📜 LICENSE                 # Project licensing
 ┣ 📜 README.md               # Documentation (this file)
 ┗ 📜 requirements.txt         # List of dependencies

```

## **📊 Output Format**  
The generated exams are saved as JSON files in /generated_exams/, structured as follows:
```json
[
  {
    "Question_ID": 1,
    "Question": "What is the main goal of Machine Learning according to Mitchell (1997)?",
    "Answer_Options": {
      "A": "To create static program instructions",
      "B": "To generalize experience to improve performance",
      "C": "To follow empirical data without learning",
      "D": "To rely solely on supervised learning"
    },
    "Correct_Answer": { "B": "To generalize experience to improve performance" },
    "Source": "lecture1.pdf"
  }
]

```

## **📜 License**  
This project is open-source under the Apache 2.0 License.
