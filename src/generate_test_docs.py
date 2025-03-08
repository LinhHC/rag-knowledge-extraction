from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
import os

# Ensure test_data directory exists
os.makedirs("test_data", exist_ok=True)

# Function to create a sample PDF document
def create_pdf(filename, title, content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use a font that supports Unicode (you may need to install 'DejaVuSans' or another TTF font)
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(200, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, content.encode("utf-8").decode("utf-8"))  # Ensure proper encoding
    pdf.output(os.path.join("test_data", filename), "F")

# Function to extract text from a web article
def extract_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n\n".join([p.get_text() for p in paragraphs])
    return text if text else "Content could not be retrieved."


# Extract content from web pages
ml_content = extract_web_content("https://en.wikipedia.org/wiki/Machine_learning")
gesture_content = extract_web_content("https://en.wikipedia.org/wiki/Gesture_recognition")
deeplearning_content = extract_web_content("https://en.wikipedia.org/wiki/Deep_learning")
computer_vision_content = extract_web_content("https://en.wikipedia.org/wiki/Computer_vision")

# Create PDFs with extended content
create_pdf("machine_learning.pdf", "Machine Learning", ml_content)
create_pdf("gesture_recognition.pdf", "esture Recognition", gesture_content)
create_pdf("deep_learning.pdf", "Deep Learning", deeplearning_content)
create_pdf("computer_vision.pdf", "Computer Vision", computer_vision_content)

print("Extended sample PDFs generated in 'test_data' folder: 'machine_learning.pdf', 'gesture_recognition.pdf', 'deep_learning.pdf', and 'computer_vision.pdf'")
