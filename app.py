import os
from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    version="1.0",
    description="A FastAPI application with RAG and LangChain integration."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load and process all PDF files in "ncert pdfs" folder
pdf_folder = "ncertpdfs"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
    data = loader.load()
    documents.extend(data)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(documents)

# Create vectorstore
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize LLM (Gemini 1.5 Pro)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None
)

# Define prompt and chain
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

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Add routes with LangServe
add_routes(
    app,
    rag_chain,
    path="/generate_questions"
)

# Render.com entry point
# Render.com entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT env variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

