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
    
    FIXED:
    - Removed 'self' parameter (standalone function)
    - Returns proper Dict format
    - Better error handling
        
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
            "query_raw_resp": [],
            "messages": state.get("messages", [])
        }
    
    print(f"Retrieving documents for: {user_query}")
    
    try:
        # Use retriever to get relevant documents
        if retriever:
            docs = retriever.invoke(user_query)
        else:
            # Fallback: use vector_store directly
            docs = vector_store.similarity_search(user_query, k=10)
        
        print(f"Retrieved {len(docs)} documents")
        
        # Format documents for response
        formatted_docs = []
        for idx, doc in enumerate(docs, 1):
            filename = osp.basename(doc.metadata.get("source", "unknown"))
            formatted_docs.append({
                "id": idx,
                "filename": filename,
                "content": doc.page_content,
            })
        
        # Store formatted documents as query_raw_resp
        return {
            "query_raw_resp": formatted_docs
        }
        
    except Exception as e:
        print(f"ERROR retrieving documents: {str(e)}")
        return {
            "query_raw_resp": [],
            "messages": state.get("messages", [])
        }


####################################################################
def pgicq_resp_validate(state: Dict, llm) -> Dict:
    """
    Wrapper to perform validation check on received response from RAG.
    
    FIXED:
    - Removed 'self' parameter (standalone function)
    - Removed unused 'retriever' parameter
    - Returns Dict (not str)
    - Increments iteration_count
    
    Args:
        state: Current graph state
        llm: Language model for validation
        
    Returns:
        Dict with evaluator_decision, evaluator_feedback, and iteration_count
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_resp_validate - Validating RAG response")
    print(f"{'='*60}")
    
    # Get query and retrieved documents
    user_query = state.get("user_query", "")
    query_raw_resp = state.get("query_raw_resp", [])
    
    # Build context from retrieved documents
    if query_raw_resp:
        context_parts = []
        for doc in query_raw_resp[:5]:  # Use top 5 docs
            context_parts.append(doc.get("content", ""))
        context = "\n\n".join(context_parts)
    else:
        context = "No documents retrieved"
    
    print(f"Validating response for query: {user_query}")
    
    # Create evaluation prompt
    evaluation_prompt = f"""You are evaluating if the retrieved documents adequately answer the user's query.

User Query: {user_query}

Retrieved Context:
{context[:2000]}  # Limit context length

Instructions:
- If the context contains relevant information that can answer the query, respond with decision: "yes"
- If the context is incomplete, irrelevant, or doesn't answer the query, respond with decision: "no"
- Provide feedback explaining your decision

Output valid JSON:
{{
  "decision": "yes" or "no",
  "feedback": "explanation here"
}}"""
    
    try:
        # Use structured output for evaluation
        llm_structured = llm.with_structured_output(Evaluator, method="json_mode")
        evaluator_obj = llm_structured.invoke(evaluation_prompt)
        
        decision = evaluator_obj.decision.lower()
        feedback = evaluator_obj.feedback or ""
        
        print(f"Evaluator decision: {decision}")
        print(f"Evaluator feedback: {feedback}")
        
        # FIXED: Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        
        return {
            "evaluator_decision": decision,
            "evaluator_feedback": feedback,
            "iteration_count": iteration_count
        }
        
    except Exception as e:
        print(f"ERROR in validation: {str(e)}")
        # Default to "yes" to avoid infinite loops on error
        return {
            "evaluator_decision": "yes",
            "evaluator_feedback": f"Validation error: {str(e)}",
            "iteration_count": state.get("iteration_count", 0) + 1
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
    
    Args:
        state: Current graph state containing retrieved documents
        llm: Language model for synthesis
        
    Returns:
        Dictionary with final response and updated messages
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_synthesize_response - Generating queue response")
    print(f"{'='*60}")
    
    query_raw_resp = state.get('query_raw_resp', [])
    original_query = state.get('user_query', '')
    messages = state.get('messages', [])
    
    # Format retrieved documents
    if query_raw_resp and isinstance(query_raw_resp, list):
        doc_contents = []
        for i, doc in enumerate(query_raw_resp[:5], 1):  # Use top 5 docs
            if isinstance(doc, dict):
                content = doc.get("content", "")
                filename = doc.get("filename", "unknown")
                doc_contents.append(f"Document {i} ({filename}):\n{content[:500]}...")
            else:
                doc_contents.append(f"Document {i}:\n{str(doc)[:500]}...")
        
        query_tuned_resp = "\n\n".join(doc_contents)
        print(f"Formatting {len(query_raw_resp)} RAG documents")
    else:
        query_tuned_resp = "No relevant documents found in the knowledge base"
        print("No documents retrieved")
    
    # Build synthesis prompt for queue queries
    synthesis_prompt = f"""{PROMPT_SYNTHESIS}

USER'S ORIGINAL QUESTION:
{original_query}

RETRIEVED INFORMATION FROM KNOWLEDGE BASE:
{query_tuned_resp}

INSTRUCTIONS:
Based on the retrieved information above, provide a comprehensive answer to the user's question about power grid interconnection queue. 
Synthesize the information from multiple sources if available.
If the information is insufficient, acknowledge what is known and what cannot be answered.
Focus on interconnection processes, requirements, policies, and procedures.

RESPONSE:"""
    
    # LLM invocation
    final_response = llm.invoke([
        SystemMessage(content="You are a helpful assistant for power grid interconnection queue queries."),
        HumanMessage(content=synthesis_prompt)
    ])
    
    final_answer = final_response.content
    print(f"Generated queue response ({len(final_answer)} characters)")
    
    # Update conversation context
    conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."
    updated_messages = messages + [AIMessage(content=final_answer)]
    
    return {
        "query_final_resp": final_answer,
        "messages": updated_messages,
        "conversation_context": conversation_summary
    }