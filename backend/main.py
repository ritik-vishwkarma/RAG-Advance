from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
import uvicorn

from backend.services.document_service import process_and_store_upload

app = FastAPI(
    title="PDF Chat Semantic Engine API",
    description="Decoupled backend driving the CRAG/RAG architectures."
)

class ChatRequest(BaseModel):
    query: str
    github_token: str = ""
    model_name: str = "gpt-4o"

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def health_check():
    return {"status": "Backend running natively on FastAPI"}

@app.post("/api/v1/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    """Ingests PDFs and visual documents, creating processed Qdrant vectors."""
    total_docs = 0
    for file in files:
        try:
            bytes_data = await file.read()
            chunks = process_and_store_upload(bytes_data, file.filename, file.content_type)
            total_docs += chunks
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")
            
    return {"message": f"Successfully processed {len(files)} files and inserted {total_docs} semantic chunks into Qdrant."}

from backend.core.crag_graph import crag_pipeline

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """The central inference hub connecting requests to the LangGraph pipeline."""
    try:
        inputs = {
            "question": request.query,
             # Using local/github based on token presense logic (default groq fallback if expanded)
            "provider": "github" if request.github_token else "local",
            "api_key": request.github_token,
            "model_name": request.model_name
        }
        
        # Invoke Semantic Graph
        result = crag_pipeline.invoke(inputs)
        return {"answer": result["answer"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
