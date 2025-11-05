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
    
    #vector_store.add_documents(docs)
    print(f"Leaving pgicq_kb_create_n_initialize with {len(docs)} chunks")
    #return vector_store
    return docs

####################################################################
def pgicq_kb_query(self, state: Dict, retriever, vector_store) -> Dict:
    """
    wrapper to perform query on power grid interconnection queue KB (RAG)
        
    Args:
       incoming_query: The search query about interconnection Queue
            
    Returns:
       JSON formatted string containing relevant interconnection Queue document excerpts
    """

    print("Entered tool_rag_document_retrieve")
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    incoming_query = state['user_query']
    docs = retriever.invoke(incoming_query)

    formatted_docs = []
    for idx, doc in enumerate(docs, 1):
        filename = osp.basename(doc.metadata.get("source", "unknown"))
        formatted_docs.append({
            "id": idx,
            "filename": filename,
            "content": doc.page_content,
        })

    formatted_json = "\n\n".join([json.dumps(doc, indent=2) for doc in formatted_docs])
    print("Leaving tool_rag_document_retrieve")
        
    return formatted_json


####################################################################
def pgicq_resp_validate(self, state: Dict, retriever, llm) -> str:
    """wrapper to perform validation check on received response from RAG """
    print("Entered in document_evaluator")
        
    # Get original query and agent's last response
    query = state["messages"][0].content if state["messages"] else ""
    agent_response = state["messages"][-1].content if len(state["messages"]) > 1 else ""

    evaluation_prompt = f"""You are evaluating if the following response adequately answers the user's query.

User Query: {query}

Agent Response: {agent_response}

Instructions:
- If the response contains relevant information that answers the query, respond with decision: "yes"
- If the response is incomplete, irrelevant, or doesn't answer the query, respond with decision: "no"
- Provide feedback explaining your decision

Output valid JSON:
{{
  "decision": "yes" or "no",
  "feedback": "explanation here"
}}"""

    llm_structured = llm.with_structured_output(Evaluator, method="json_mode")
    evaluator_obj = llm_structured.invoke(evaluation_prompt)
        
    print(f"Evaluator decision: {evaluator_obj.decision}")
    print(f"Evaluator feedback: {evaluator_obj.feedback}")
        
    return {
            "evaluator_decision": evaluator_obj.decision,
            "evaluator_feedback": evaluator_obj.feedback
    }


####################################################################
def pgicq_kb_query_rewrite(self, state: Dict, llm) -> Dict:
    """wrapper to perform query rewrite on ower grid interconnection queue initial query """
    
    print("Entered in pgicq_kb_query_rewrite ")
    
    original_query = state["messages"][0].content if state["messages"] else ""
    feedback = state.get("evaluator_feedback", "")
    
    rewrite_prompt = f"""The following query did not get a good answer. Rewrite it to be more specific and clear.

Original query: {original_query}
Feedback: {feedback}

Provide an improved, more specific version of the query:"""
    
    response = llm.invoke(rewrite_prompt)
    rewritten_query = response.content
    
    print(f"Rewritten query: {rewritten_query}")
    
    # Return new user message with rewritten query
    return {"messages": [("user", rewritten_query)]}

