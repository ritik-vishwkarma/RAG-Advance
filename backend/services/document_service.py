import io
from typing import List, Tuple
from PyPDF2 import PdfReader
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.services.image_processing import extract_images_from_pdf, caption_image
from backend.db.vector_store import add_documents_to_db

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> List[Document]:
    """Extracts text and image captions from a PDF stream and returns Document chunks."""
    text_content = f"--- Source Document: {filename} ---\n\n"
    
    # 1. Extract Text
    import fitz # PyMuPDF
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf_doc:
        extracted = page.get_text()
        if extracted:
            text_content += extracted + "\n"
    pdf_doc.close()
            
    # 2. Extract Images & Caption
    print(f"Scanning {filename} for images...")
    image_descriptions = extract_images_from_pdf(pdf_bytes)
    if image_descriptions:
        print(f"Found {len(image_descriptions)} images.")
        for page_num, desc in image_descriptions:
             text_content += f"\n{desc}\n"
             
    # 3. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_text(text_content)
    
    documents = []
    for chunk in chunks:
        # Scrub surrogate unicode pairs mapped exactly from the reference codebase
        try:
            clean_chunk = chunk.encode("utf-8", "ignore").decode("utf-8", "ignore")
        except Exception:
            clean_chunk = chunk
        documents.append(Document(page_content=clean_chunk, metadata={"source": filename}))
        
    return documents

def parse_image_bytes(image_bytes: bytes, filename: str) -> List[Document]:
    """Generates a caption for an uploaded image and wraps it in a Document."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    caption = caption_image(image)
    desc = f"[Uploaded Image {filename}: {caption}]"
    return [Document(page_content=desc, metadata={"source": filename})]

def process_and_store_upload(file_bytes: bytes, filename: str, file_type: str) -> int:
    """End-to-end ingestion pipeline bridging bytes to Qdrant vectors."""
    docs = []
    if file_type == "application/pdf":
        docs = parse_pdf_bytes(file_bytes, filename)
    elif file_type.startswith("image/"):
        docs = parse_image_bytes(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    if docs:
        add_documents_to_db(docs)
        
    return len(docs)
