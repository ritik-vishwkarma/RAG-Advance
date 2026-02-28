import re
from typing import List, TypedDict
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
import os

from backend.db.vector_store import get_vector_store
from backend.services.llm_service import get_llm

UPPER_TH = 0.7
LOWER_TH = 0.3

def extract_json_lambda(text_or_msg):
    """Fallback Regex to rip JSON out of LLM hallucinations before Pydantic parsing."""
    text = text_or_msg.content if hasattr(text_or_msg, 'content') else str(text_or_msg)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

# --- 1. State Definition ---
class CRAGState(TypedDict):
    question: str
    provider: str
    api_key: str
    model_name: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str
    
    web_docs: List[Document]

    answer: str

# --- 2. Retrieval Node ---
def retrieve_node(state: CRAGState) -> CRAGState:
    print(f"--- [CRAG] Retrieving docs for: {state['question']} ---")
    retriever = get_vector_store().as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    return {"docs": docs}

# --- 3. Evaluation Node ---
class DocEvalScore(BaseModel):
    score: float
    reason: str

def eval_each_doc_node(state: CRAGState) -> CRAGState:
    print("--- [CRAG] Evaluating chunks against threshold ---")
    llm = get_llm(state["provider"], state["api_key"], state["model_name"])
    
    parser = PydanticOutputParser(pydantic_object=DocEvalScore)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
         "You will be given ONE retrieved chunk and a question.\n"
         "Return a relevance score in [0.0, 1.0].\n"
         "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
         "- 0.0: chunk is irrelevant\n"
         "Be conservative with high scores.\n{format_instructions}"),
        ("human", "Question: {question}\n\nChunk:\n{chunk}")
    ])
    
    chain = prompt | llm | RunnableLambda(extract_json_lambda) | parser
    
    scores = []
    good = []
    
    for d in state["docs"]:
        try:
            out = chain.invoke({"question": state["question"], "chunk": d.page_content, "format_instructions": parser.get_format_instructions()})
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)
        except Exception as e:
            print(f"Eval fallback error: {e}")
            
    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT", "reason": "High relevance found."}
        
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT", "reason": "No chunk cleared lower bounds."}
        
    return {"good_docs": good, "verdict": "AMBIGUOUS", "reason": "Mixed relevance signals."}

# --- Routing ---
def route_after_eval(state: CRAGState) -> str:
    v = state.get("verdict", "INCORRECT")
    if v == "CORRECT":
         return "refine"
    else:
         return "rewrite_query"

# --- 4. Query Rewrite & Web Search ---
class WebQuery(BaseModel):
    query: str

def rewrite_query_node(state: CRAGState) -> CRAGState:
    print("--- [CRAG] Rewriting query for search engine ---")
    llm = get_llm(state["provider"], state["api_key"], state["model_name"])
    parser = PydanticOutputParser(pydantic_object=WebQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user question into a web search query (6-14 words).\nDo NOT answer the question.\n{format_instructions}"),
        ("human", "Question: {question}")
    ])
    try:
        chain = prompt | llm | RunnableLambda(extract_json_lambda) | parser
        out = chain.invoke({"question": state["question"], "format_instructions": parser.get_format_instructions()})
        return {"web_query": out.query}
    except Exception:
        return {"web_query": state["question"]}

def web_search_node(state: CRAGState) -> CRAGState:
    print("--- [CRAG] Searching web via Tavily ---")
    try:
         if not os.environ.get("TAVILY_API_KEY"):
             print("Web Search skipped: No TAVILY_API_KEY found.")
             return {"web_docs": []}
             
         tavily = TavilySearch(max_results=3)
         q = state.get("web_query") or state["question"]
         results = tavily.invoke({"query": q})
         web_docs = [Document(page_content=r["content"], metadata={"url": r["url"]}) for r in results]
         return {"web_docs": web_docs}
    except Exception as e:
         print(f"Web Search fallback: {e}")
         return {"web_docs": []}

# --- 5. Sentence-Level Refinement ---
class KeepOrDrop(BaseModel):
    keep: bool

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def refine_node(state: CRAGState) -> CRAGState:
    print("--- [CRAG] Decomposing & Filtering sentences ---")
    q = state["question"]
    llm = get_llm(state["provider"], state["api_key"], state["model_name"])
    
    docs_to_use = state.get("good_docs", []) if state.get("verdict") == "CORRECT" else state.get("good_docs", []) + state.get("web_docs", [])
    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)
    
    parser = PydanticOutputParser(pydantic_object=KeepOrDrop)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Strict relevance filter. Return keep=true if sentence directly helps answer the question.\n{format_instructions}"),
        ("human", "Question: {question}\n\nSentence:\n{sentence}")
    ])
    chain = prompt | llm | RunnableLambda(extract_json_lambda) | parser
    
    kept = []
    for s in strips[:15]: # Cap to prevent endless loops on giant contexts
        try:
             if chain.invoke({"question": q, "sentence": s, "format_instructions": parser.get_format_instructions()}).keep:
                  kept.append(s)
        except Exception:
             pass
             
    return {"strips": strips, "kept_strips": kept, "refined_context": "\n".join(kept).strip()}

# --- 6. Generation ---
def generate_node(state: CRAGState) -> CRAGState:
    print("--- [CRAG] Generating final contextual answer ---")
    llm = get_llm(state["provider"], state["api_key"], state["model_name"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful UI assistant. Answer ONLY using the provided context.\nIf the context is empty or insufficient, say: 'I don't know.'"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    response = llm.invoke(prompt.format(question=state["question"], context=state["refined_context"]))
    out = response.content if hasattr(response, 'content') else str(response)
    return {"answer": out}

# --- Build the Graph ---
workflow = StateGraph(CRAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("eval_each_doc", eval_each_doc_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("refine", refine_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "eval_each_doc")

workflow.add_conditional_edges("eval_each_doc", route_after_eval, {"refine": "refine", "rewrite_query": "rewrite_query"})

workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("web_search", "refine")
workflow.add_edge("refine", "generate")
workflow.add_edge("generate", END)

crag_pipeline = workflow.compile()
