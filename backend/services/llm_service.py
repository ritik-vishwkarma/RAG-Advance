import os
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "gemma-2-2b-it.Q4_K_M.gguf")
N_CTX = 2048
N_THREADS = 8

# Singleton to prevent VRAM memory bleeds on repeated queries
_local_llm = None

def get_llm(provider: str = "local", api_key: str = None, model_name: str = "gpt-4o"):
    """
    Factory service to acquire an LLM dependency.
    """
    global _local_llm
    
    if provider == "local":
        if _local_llm is not None:
            return _local_llm
            
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Local model not found at {MODEL_PATH}")

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        print(f"Loading local model into VRAM: {MODEL_PATH}")
        _local_llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=20, # Offload to GPU
            n_batch=512,
            n_ctx=N_CTX,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,
            temperature=0.1,
            n_threads=N_THREADS
        )
        return _local_llm

    elif provider == "github":
        if not api_key:
            raise ValueError("GitHub Token is required for GitHub Models.")
            
        print(f"Connecting to GitHub Models: {model_name}")
        # GitHub Models is OpenAI Compatible
        return ChatOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=api_key,
            model=model_name,
            temperature=0.1
        )
        
    raise ValueError(f"Unknown LLM provider: {provider}")
