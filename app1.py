from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Flask app
app = Flask(__name__)

# Add CORS support
CORS(app, resources={r"/*": {"origins": "*"}})

# Folder for storing PDFs
PDF_FOLDER = "ncertpdfs"

def get_pdf_text(pdf_folder):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_vector_store():
    print("Processing PDFs from ncertpdfs...")
    raw_text = get_pdf_text(PDF_FOLDER)
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store initialized successfully.")

def create_retrieval_chain(retriever, question_answer_chain):
    def rag_chain(input_data):
        docs = retriever.similarity_search(input_data["input"])
        response = question_answer_chain({"input_documents": docs, "question": input_data["input"]}, return_only_outputs=True)
        return {"answer": response["output_text"]}
    return rag_chain

def create_stuff_documents_chain(llm, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

@app.route("/generate_questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    # Load retriever and chain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    retriever = FAISS.load_local("faiss_index", embeddings)

    system_prompt = (
        "You are an assistant specialized in creating multiple-choice questions (MCQs). "
        "Using the provided context, follow these instructions to generate questions:\n\n"
        "1. **Question Requirements:**\n"
        "   - Create 10 multiple-choice questions (MCQs).\n"
        "   - Each question must have 4 options (A, B, C, D).\n"
        "   - Ensure the correct answers are present in the provided context.\n\n"
        "2. **Relevance Check:**\n"
        "   - If the chapter name or topic is not explicitly mentioned in the context, respond with 'Out of syllabus'.\n\n"
        "3. **Output Format:**\n"
        "   - Provide the output in JSON format.\n"
        "   - Each question and its options should be distinct properties within the JSON structure.\n\n"
        "Input Context:\n"
        "{context}"
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    question_answer_chain = create_stuff_documents_chain(llm, system_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain({"input": query})
    return jsonify({"questions": response["answer"]})

if __name__ == "__main__":
    initialize_vector_store()
    app.run(debug=True)
