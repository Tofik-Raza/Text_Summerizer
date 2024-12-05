from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import torch
import os
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("Hindi_Abusive.txt", "r", encoding="utf-8") as hindi_file:
    hindi_abusive_words = set(line.strip().lower() for line in hindi_file)

with open("English_Abusive.txt", "r", encoding="utf-8") as english_file:
    english_abusive_words = set(line.strip().lower() for line in english_file)

# Combine abusive words
offensive_words = hindi_abusive_words.union(english_abusive_words)


class TextInput(BaseModel):
    text: str


# Load the model and tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="cpu", torch_dtype=torch.float32)

# File directory for temporary storage
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def pdf_preprocessing(file_path):
    """Preprocesses a PDF file to extract text."""
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join(text.page_content for text in texts)
    return final_texts


def txt_preprocessing(file_path):
    """Preprocesses a .txt file to extract text."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    final_texts = " ".join(texts)
    return final_texts


def llm_pipeline(input_text):
    """Pipeline for generating summaries."""
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    result = pipe_sum(input_text)
    summary = result[0]['summary_text']
    return summary

@app.post("/check_text")
async def check_text(input: TextInput):
    user_text = input.text.lower()
    words = user_text.split()
    detected_abusive = [word for word in words if word in offensive_words]

    if detected_abusive:
        return {"status": "offensive", "abusive_words": detected_abusive}
    return {"status": "safe"}

@app.post("/input/")
async def summarize_text(input_text: str = Form(...)):
    """Endpoint for summarizing text input."""
    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        summary = llm_pipeline(input_text)
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.post("/document/")
async def summarize_document(file: UploadFile = File(...)):
    """Endpoint for summarizing PDF or TXT files."""
    file_type = file.filename.split('.')[-1].lower()

    if file_type not in ["pdf", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or TXT file.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        if file_type == "pdf":
            input_text = pdf_preprocessing(file_path)
        elif file_type == "txt":
            input_text = txt_preprocessing(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    try:
        summary = llm_pipeline(input_text)
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.get("/")
async def home():
    """Homepage with file upload form and text input form."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Document Summarization App</h2>

        <h3>Summarize Text</h3>
        <form action="/input/" method="post">
            <textarea name="input_text" rows="10" cols="50" placeholder="Enter text here..." required></textarea>
            <br><br>
            <input type="submit" value="Summarize Text">
        </form>

        <h3>Summarize Document</h3>
        <form action="/document/" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept=".pdf,.txt" required>
            <br><br>
            <input type="submit" value="Summarize Document">
        </form>
    </body>
    </html>
    """, status_code=200)
