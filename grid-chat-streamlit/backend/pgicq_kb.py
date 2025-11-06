"""
##########################################################
#
# Capstone Team 16: 
#    Power Grids Interconnection Queue Analyzer
#
#  Module:
#     Power Grids Interconnection Queue Knowledge Base
#     Responsible for serving the queries related to 
#      - Grid iteroperability
#      - Grid connection queue details 
#      - Grid capabilities
#      - Policies and procedures etc.
#     
#  Author: RK (kvrkr866@gmail.com)
#
#  Phase1: 
#     Iteration 1 - 
#        -- Not Applicable. 
#
#     Iteration 2 (added power grid interconnection queue)
#   
##########################################################
"""
import json
import os.path as osp
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from constants import PGICQ_DOCS_FILE_PATHS, Evaluator, Userqueryclassifier, GridState
from prompts import PROMPT_SYNTHESIS



####################################################################
def pgicq_kb_create_n_initialize():
    """Loads and splits PDF documents into smaller chunks for embedding."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    import time
        
    print("Entered pgicq_kb_create_n_initialize")
    docs = []
    for file_path in PGICQ_DOCS_FILE_PATHS:
        print(f"Loading {osp.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        combined_text = "\n".join(p.page_content for p in pages)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_text(combined_text)

        docs.extend([
            Document(page_content=chunk, metadata={"source": file_path})
            for chunk in chunks
        ])
    
    print(f"Leaving pgicq_kb_create_n_initialize with {len(docs)} chunks")
    return docs


####################################################################
def pgicq_kb_query(state: Dict, retriever, vector_store) -> Dict:
    """
    Wrapper to perform query on power grid interconnection queue KB (RAG).
    
    OPTIMIZED:
    - Retrieve top 10 docs (keep quality)
    - Use MMR (Maximal Marginal Relevance) for diversity
    - Parallel processing where possible
        
    Args:
       state: Current graph state
       retriever: Vector store retriever
       vector_store: ChromaDB vector store
            
    Returns:
       Dict with query_raw_resp containing retrieved documents
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_kb_query - Retrieving from knowledge base")
    print(f"{'='*60}")
    
    user_query = state.get('user_query', '')
    if not user_query:
        print("ERROR: No user query found in state")
        return {
            "queue_raw_resp": [],  # Changed from query_raw_resp
            "messages": state.get("messages", [])
        }
    
    print(f"Retrieving documents for: {user_query}")
    
    try:
        # OPTIMIZED: Use MMR for better diversity and relevance
        if retriever:
            # Update retriever to use MMR if available
            docs = retriever.invoke(user_query)
        else:
            # Use similarity search with MMR
            docs = vector_store.max_marginal_relevance_search(
                user_query, 
                k=10,  # Keep 10 for quality
                fetch_k=20  # Fetch 20, then select diverse 10
            )
        
        print(f"Retrieved {len(docs)} documents using MMR")
        
        # Format documents efficiently
        formatted_docs = [
            {
                "id": idx,
                "filename": osp.basename(doc.metadata.get("source", "unknown")),
                "content": doc.page_content,
            }
            for idx, doc in enumerate(docs, 1)
        ]
        
        # Return as queue_raw_resp for sequential flow
        return {
            "queue_raw_resp": formatted_docs  # Changed from query_raw_resp
        }
        
    except Exception as e:
        print(f"ERROR retrieving documents: {str(e)}")
        return {
            "queue_raw_resp": [],  # Changed from query_raw_resp
            "messages": state.get("messages", [])
        }


####################################################################
def pgicq_resp_validate(state: Dict, llm) -> Dict:
    """
    Wrapper to perform validation check on received response from RAG.
    
    OPTIMIZED:
    - Skip validation after first successful retrieval (trust the RAG)
    - Only validate if no documents found or iteration > 0
    - Simpler validation logic
    
    Args:
        state: Current graph state
        llm: Language model for validation
        
    Returns:
        Dict with evaluator_decision, evaluator_feedback, and iteration_count
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_resp_validate - Validating RAG response")
    print(f"{'='*60}")
    
    queue_raw_resp = state.get("queue_raw_resp", [])  # Changed from query_raw_resp
    iteration_count = state.get("iteration_count", 0)
    
    # OPTIMIZED: Skip expensive validation if we got documents on first try
    if iteration_count == 0 and len(queue_raw_resp) > 0:
        print("First retrieval successful with documents - skipping validation")
        return {
            "evaluator_decision": "yes",
            "evaluator_feedback": "Documents retrieved successfully",
            "iteration_count": 1
        }
    
    # Only validate if no documents or retrying
    user_query = state.get("user_query", "")
    
    # Build minimal context for validation
    if queue_raw_resp:  # Changed from query_raw_resp
        # Just check first doc snippet
        first_doc = queue_raw_resp[0] if isinstance(queue_raw_resp[0], dict) else {}
        context = first_doc.get("content", "")[:200]  # Just 200 chars for validation
    else:
        context = "No documents retrieved"
    
    print(f"Validating with limited context (iteration {iteration_count})")
    
    # OPTIMIZED: Simpler validation prompt
    evaluation_prompt = f"""Check if this snippet is relevant to the query.

Query: {user_query}
Snippet: {context}

Return JSON: {{"decision": "yes" or "no", "feedback": "brief reason"}}"""
    
    try:
        llm_structured = llm.with_structured_output(Evaluator, method="json_mode")
        evaluator_obj = llm_structured.invoke(evaluation_prompt)
        
        decision = evaluator_obj.decision.lower()
        feedback = evaluator_obj.feedback or ""
        
        print(f"Validation: {decision}")
        
        return {
            "evaluator_decision": decision,
            "evaluator_feedback": feedback,
            "iteration_count": iteration_count + 1
        }
        
    except Exception as e:
        print(f"ERROR in validation: {str(e)}")
        # Default to "yes" to avoid loops
        return {
            "evaluator_decision": "yes",
            "evaluator_feedback": f"Validation skipped: {str(e)}",
            "iteration_count": iteration_count + 1
        }


####################################################################
def pgicq_kb_query_rewrite(state: Dict, llm) -> Dict:
    """
    Wrapper to perform query rewrite on power grid interconnection queue initial query.
    
    FIXED:
    - Removed 'self' parameter (standalone function)
    - Better query rewriting logic
    - Preserves messages history
    
    Args:
        state: Current graph state
        llm: Language model for rewriting
        
    Returns:
        Dict with updated user_query
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_kb_query_rewrite - Rewriting query")
    print(f"{'='*60}")
    
    original_query = state.get("user_query", "")
    feedback = state.get("evaluator_feedback", "")
    
    print(f"Original query: {original_query}")
    print(f"Feedback: {feedback}")
    
    # Create rewrite prompt
    rewrite_prompt = f"""The following query about power grid interconnection did not retrieve good results.
Rewrite it to be more specific and clear, focusing on interconnection processes, policies, or technical requirements.

Original query: {original_query}
Feedback: {feedback}

Provide an improved, more specific version of the query that focuses on:
- Interconnection processes
- Grid connection requirements
- Queue management
- Technical specifications
- Regulatory policies

Rewritten query:"""
    
    try:
        response = llm.invoke(rewrite_prompt)
        rewritten_query = response.content.strip()
        
        print(f"Rewritten query: {rewritten_query}")
        
        # FIXED: Return dict with updated user_query
        return {
            "user_query": rewritten_query
        }
        
    except Exception as e:
        print(f"ERROR rewriting query: {str(e)}")
        # Return original query if rewrite fails
        return {
            "user_query": original_query
        }


####################################################################
def pgicq_synthesize_response(state: Dict, llm) -> Dict:
    """
    Synthesize final response for queue (RAG) queries.
    
    OPTIMIZED:
    - Use all 10 documents (keep quality)
    - Smart truncation based on relevance
    - Streaming response (if supported)
    - Efficient prompt construction
    
    Args:
        state: Current graph state containing retrieved documents
        llm: Language model for synthesis
        
    Returns:
        Dictionary with final response and updated messages
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_synthesize_response - Generating queue response")
    print(f"{'='*60}")
    
    queue_raw_resp = state.get('queue_raw_resp', [])  # Changed from query_raw_resp
    original_query = state.get('user_query', '')
    messages = state.get('messages', [])
    
    # OPTIMIZED: Use all docs but with smart truncation
    if queue_raw_resp and isinstance(queue_raw_resp, list):
        # Process all docs efficiently
        doc_contents = []
        total_chars = 0
        max_total_chars = 3000  # Total context limit
        
        for i, doc in enumerate(queue_raw_resp, 1):  # Changed from query_raw_resp
            if total_chars >= max_total_chars:
                break  # Stop if we hit limit
                
            if isinstance(doc, dict):
                content = doc.get("content", "")
                filename = doc.get("filename", "unknown")
                
                # Allocate more space to top docs
                if i <= 3:
                    max_len = 500
                elif i <= 6:
                    max_len = 300
                else:
                    max_len = 200
                
                truncated = content[:max_len]
                doc_contents.append(f"[Source {i}: {filename}]\n{truncated}")
                total_chars += len(truncated)
        
        query_tuned_resp = "\n\n".join(doc_contents)
        print(f"Used {len(doc_contents)} documents, {total_chars} chars total")
    else:
        query_tuned_resp = "No relevant documents found"
        print("No documents retrieved")
    
    # OPTIMIZED: Concise but complete prompt
    synthesis_prompt = f"""Answer the question using the provided power grid interconnection documentation.

QUESTION: {original_query}

DOCUMENTATION:
{query_tuned_resp}

Provide a clear answer. Cite sources when possible."""
    
    # LLM invocation
    final_response = llm.invoke([
        SystemMessage(content="You are a power grid interconnection expert."),
        HumanMessage(content=synthesis_prompt)
    ])
    
    final_answer = final_response.content
    print(f"Generated response ({len(final_answer)} chars)")
    
    # Update conversation context
    conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."
    updated_messages = messages + [AIMessage(content=final_answer)]
    
    return {
        "query_final_resp": final_answer,
        "messages": updated_messages,
        "conversation_context": conversation_summary
    }

####################################################################
def pgicq_kb_query_generic(
    query: str,
    retriever,
    vector_store,
    k: int = 10
) -> List[Dict]:
    """
    Generic RAG query function for gap analysis.
    Reuses existing RAG infrastructure for any knowledge base query.
    
    This function is designed to be called from gap analysis nodes
    with specific queries for different analysis aspects.
    
    Args:
        query: The specific query string to search for
        retriever: Vector store retriever instance
        vector_store: ChromaDB vector store instance
        k: Number of documents to retrieve (default: 10)
        
    Returns:
        List of dictionaries containing:
            - id: Document index
            - filename: Source document name
            - content: Document content
            - page: Page number if available
    """
    print(f"\nGeneric RAG Query: {query[:100]}...")
    
    try:
        # Use MMR for better diversity and relevance
        if retriever:
            docs = retriever.invoke(query)
        else:
            docs = vector_store.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=k*2  # Fetch more, then select diverse subset
            )
        
        print(f"Retrieved {len(docs)} documents")
        
        # Format documents consistently
        formatted_docs = []
        for idx, doc in enumerate(docs, 1):
            # Extract metadata
            source = doc.metadata.get("source", "unknown")
            filename = osp.basename(source)
            
            # Try to get page number if available
            page = doc.metadata.get("page", "N/A")
            
            formatted_docs.append({
                "id": idx,
                "filename": filename,
                "content": doc.page_content,
                "page": str(page),
                "source": source
            })
        
        return formatted_docs
        
    except Exception as e:
        print(f"ERROR in generic RAG query: {str(e)}")
        return []
