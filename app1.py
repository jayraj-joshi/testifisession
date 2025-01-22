from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS configuration to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class QueryRequest(BaseModel):
    query: str

# Load and process PDF data
loader = PyPDFLoader("plantphisio.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    temperature=0, 
    max_tokens=None, 
    timeout=None
)

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

# API endpoint
@app.post("/generate_questions")
def generate_questions(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    return {"questions": response["answer"]}
