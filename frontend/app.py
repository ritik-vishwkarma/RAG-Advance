import streamlit as st
import requests
import os

API_BASE_URL = "http://localhost:8000/api/v1"

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "model_provider" not in st.session_state:
        st.session_state.model_provider = "local"
    if "github_token" not in st.session_state:
        st.session_state.github_token = ""
    if "github_model" not in st.session_state:
        st.session_state.github_model = "gpt-4o"

def process_uploaded_files(uploaded_files):
    """Sends uploaded files to the FastAPI backend for ingestion."""
    st.info("Transmitting documents to the Semantic Engine...")
    files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
    try:
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        data = response.json()
        st.success(data.get("message", "Processing Complete"))
        st.session_state.processed = True
    except requests.exceptions.HTTPError as err:
        st.error(f"Backend Server Error: {err.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to Semantic Engine Backend. Is it running on port 8000?")
    except Exception as e:
        st.warning(f"Failed to process files: {e}")

def handle_user_input(user_question):
    """Sends the user question to the Backend inference hub."""
    st.chat_message("user").write(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing semantics via backend pipeline..."):
             try:
                 payload = {
                     "query": user_question,
                     "github_token": st.session_state.github_token or os.environ.get("GITHUB_TOKEN", ""),
                     "model_name": st.session_state.github_model
                 }
                 response = requests.post(f"{API_BASE_URL}/chat", json=payload)
                 response.raise_for_status()
                 data = response.json()
                 answer = data.get("answer", "No answer received.")
                 st.write(answer)
                 st.session_state.chat_history.append({"role": "assistant", "content": answer})
             except requests.exceptions.HTTPError as err:
                 st.error(f"Generation failure: {err.response.text}")
             except requests.exceptions.ConnectionError:
                 st.error("Semantic Engine API offline (Connection refused).")
             except Exception as e:
                 st.error(f"Unexpected error: {str(e)}")

def main():
    st.set_page_config(page_title="PDF Chat Semantic Engine", page_icon="🏡", layout="wide")
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    .stSidebar { background-color: #1E1E1E; }
    div.stButton > button:first-child { background-color: #3b82f6; color: white; border: none; }
    div.stButton > button:first-child:hover { background-color: #2563eb; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #2d3748; }
    .assistant-message { background-color: #4a5568; }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.title("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs or Images", 
            accept_multiple_files=True, 
            type=["pdf", "png", "jpg", "jpeg"]
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                process_uploaded_files(uploaded_files)
            else:
                st.warning("Please upload files first.")
                
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            
        st.markdown("---")
        st.subheader("Model Configuration")
        
        provider = st.selectbox(
            "Model Provider",
            ["local", "github"],
            index=0 if st.session_state.model_provider == "local" else 1,
            key="provider_select"
        )
        st.session_state.model_provider = provider
        
        if provider == "github":
             # Try environment variable first, then session state
            default_token = os.environ.get("GITHUB_TOKEN", st.session_state.github_token)
            
            token = st.text_input(
                "GitHub Token", 
                type="password", 
                value=default_token,
                help="Requires a 'GITHUB_TOKEN' environment variable or explicit entry."
            )
            st.session_state.github_token = token
            
            model = st.selectbox(
                "Model",
                ["gpt-4o", "meta-llama-3-70b-instruct", "mistral-large"],
                index=0
            )
            st.session_state.github_model = model
            
        st.markdown("---")
        st.markdown("Powered by **FastAPI + LangGraph + Qdrant**")

    # --- Main Content ---
    st.title("Chat with PDF Semantic Engine 🏡🤖")
    st.caption("Decoupled Frontend connected to local Backend.")
    
    # Render chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    user_question = st.chat_input("Ask a question specific to your documents")
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
