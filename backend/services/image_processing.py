import io
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Global cache for model and processor
_processor = None
_model = None

def get_image_captioner():
    """Lazy load the BLIP model and processor."""
    global _processor, _model
    if _processor is None or _model is None:
        print("Loading BLIP image captioning model...")
        # Use a small, fast model suitable for CPU/low-end GPU
        model_id = "Salesforce/blip-image-captioning-base"
        _processor = BlipProcessor.from_pretrained(model_id)
        _model = BlipForConditionalGeneration.from_pretrained(model_id)
        
        # Move to GPU if available and decent (though BLIP base is fine on CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        print(f"BLIP model loaded on {device}")
    
    return _processor, _model

def caption_image(image: Image.Image) -> str:
    """Generate a caption for a PIL Image."""
    try:
        processor, model = get_image_captioner()
        device = model.device
        
        # Preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error captioning image: {e}")
        return "An image that could not be described."

def extract_images_from_pdf(pdf_stream) -> list[tuple[int, str]]:
    """
    Extract images from a PDF stream and caption them.
    Returns a list of (page_number, image_description).
    """
    image_descriptions = []
    
    try:
        # Open PDF from stream
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            if image_list:
                print(f"Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    try:
                        # Load image with Pillow
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        
                        # Filter out very small icons/logos (e.g., less than 100x100)
                        if image.width < 100 or image.height < 100:
                            continue
                            
                        # Generate caption
                        caption = caption_image(image)
                        desc = f"[Image on Page {page_num + 1}, Image {img_index + 1}: {caption}]"
                        image_descriptions.append((page_num + 1, desc))
                        print(f"Processed image: {desc}")
                        
                    except Exception as e:
                        print(f"Failed to process image on page {page_num + 1}: {e}")
                        
        pdf_document.close()
        
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        
    return image_descriptions
