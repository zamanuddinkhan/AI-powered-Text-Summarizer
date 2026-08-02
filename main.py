from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Create FastAPI app
app = FastAPI(title="AI Text Summarizer")

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Request model
class TextRequest(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to AI Text Summarizer API"}

# Summarization route
@app.post("/summarize")
def summarize(request: TextRequest):
    summary = summarizer(
        request.text,
        max_length=100,
        min_length=30,
        do_sample=False
    )

    return {
        "original_text": request.text,
        "summary": summary[0]["summary_text"]
    }