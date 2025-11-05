"""
##########################################################
#
# Capstone Team 16: Power Grids Interconnection Queue Analyzer
#
#  Author: RK (kvrkr866@gmail.com)
#
#  Phase1: 
#     Iteration 1 - 
#        -- View diagnostic events of the power grid
#        -- Query support using natural langauge and UI (streamlit)
#        -- Retrieving of the events from external SQlite DB
#        -- support of tracing through comet opik
#        -- Chat history and context support through Memory feature.
#        -- Tested with minimal set of UT data and generated Evals and measurements.
#
#     Iteration 2 -
#        -- Added query classifier for routing
#        -- Added RAG support for queue queries
#   
##########################################################
"""

from dotenv import load_dotenv
import os
import json
import os.path as osp
from typing import Dict, List, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field

from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from opik.integrations.langchain import OpikTracer

# Import modular components
from diagevents import diagevents_query_prepare, diagevents_query_execute
from pgicq_kb import pgicq_kb_create_n_initialize, pgicq_kb_query, pgicq_resp_validate, pgicq_kb_query_rewrite

from prompts import PROMPT_SYNTHESIS, USERQUERY_CLASSIFIER_PROMPT
from constants import PGICQ_DOCS_FILE_PATHS, Evaluator, Userqueryclassifier, GridState



####################################################################
class GridChat:
    """
    Grid Chat main orchestrator with memory for contextual conversations.
    
    This class coordinates different modules (diagevents, pgicq_kb)
    and maintains the conversation flow.
    """
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.tracer = None
        self.memory = None
        self.retriever = None
        self.vector_store = None
        self.thread_id = "default_session"  # Can be customized per user session
    
    def initialize(self) -> None:
        """Initialize components for Grid chat with memory."""
        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        import time
        from constants import PGICQ_KB_PERSIST_DIRECTORY, PGICQ_KB_COLLECTION_NAME

        
        load_dotenv()
        # model name and parameters
        self.llm = init_chat_model(
            "gpt-5-mini",  
            model_provider="openai", 
            reasoning_effort="minimal"
        )
        print("✓ LLM initialized")
        
        # Initialize memory saver
        self.memory = MemorySaver()
        print("✓ Memory initialized")

        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Embeddings initialized")

        # Initialize vector store
        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=PGICQ_KB_PERSIST_DIRECTORY,
            collection_name=PGICQ_KB_COLLECTION_NAME, 
        )

        # Check if we need to load documents
        has_existing_documents = len(self.vector_store.get(limit=1)['ids']) > 0
        if has_existing_documents:
            print(" PGICQ KB related Chroma Vector DB found - reusing existing embeddings.")
        else:
            print("No PGICQ KB related Chroma Vector ChromaDB found - processing and embedding documents...")
            docs = pgicq_kb_create_n_initialize()
            print(f"Loaded and chunked {len(docs)} document pieces")
            self.vector_store.add_documents(docs)
            print("Embeddings processed and stored in ChromaDB.")
        
        print("✓ PGICQ KB (Chroma Vector DB) initialized")

        
        # Build the graph
        self._build_graph()
        print("✓ Graph built successfully")
        
        # Track the graph structure
        self.tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
    
    ####################################################################
    # Node functions
    ####################################################################
    
    def userquery_classifier(self, state: GridState) -> Dict:
        """
        Determine classification of user query into diagnostics or queue or general.
        
        FIXED: Returns Dict to update state, not the classification directly.
        """
        print(f"\n{'='*60}")
        print("NODE: userquery_classifier - Classifying user query")
        print(f"{'='*60}")
        
        # Get user query from state
        user_query = state.get("user_query", "")
        if not user_query and state.get("messages"):
            last_msg = state["messages"][-1]
            user_query = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        print(f"User query: {user_query}")
        
        # Create classifier chain
        classifier_chain = USERQUERY_CLASSIFIER_PROMPT | self.llm | StrOutputParser()
        userquery_type = classifier_chain.invoke({"question": user_query}).strip().lower()
        
        # Ensure category is valid
        valid_categories = ["diagnostics", "queue", "general"]
        if userquery_type not in valid_categories:
            userquery_type = "general"
        
        print(f"Classified as: {userquery_type}")
        
        # FIXED: Return dict to update state
        return {
            "userquery_type": userquery_type,
            "user_query": user_query
        }
 
    ####################################################################
    # Wrapper methods for modular functions
    ####################################################################

    def w_diagevents_query_prepare(self, state: GridState) -> Dict:
        """
        Wrapper for diagevents_query_prepare.
        This allows easy swapping of modules in the future.
        """
        return diagevents_query_prepare(state, self.llm)
    
    def w_diagevents_query_execute(self, state: GridState) -> Dict:
        """
        Wrapper for diagevents_query_execute.
        This allows easy swapping of modules in the future.
        """
        return diagevents_query_execute(state)

    def w_pgicq_kb_query(self, state: GridState) -> Dict:
        """Wrapper to perform query on power grid interconnection queue KB (RAG)"""
        return pgicq_kb_query(state, self.retriever, self.vector_store)

    def w_pgicq_resp_validate(self, state: GridState) -> Dict:
        """Wrapper to perform validation check on received response from RAG"""
        return pgicq_resp_validate(state, self.llm)

    def w_pgicq_kb_query_rewrite(self, state: GridState) -> Dict:
        """Wrapper to perform query rewrite on power grid interconnection queue initial query"""
        return pgicq_kb_query_rewrite(state, self.llm)

    ####################################################################
    # Routing methods for langgraph conditional edges
    # FIXED: Added self parameter and return string (not dict)
    ####################################################################
    
    def route_userquery(self, state: GridState) -> str:
        """
        Conditional router: decides next node based on userquery_type.
        
        FIXED: 
        - Added self parameter
        - Returns string (route name), not dict
        """
        userquery_type = state.get("userquery_type", "general").lower()
        
        print(f"Routing based on type: {userquery_type}")
        
        if userquery_type == "diagnostics":
            print("→ Routing to diagnostics flow")
            return "w_diagevents_query_prepare"
        elif userquery_type == "queue":
            print("→ Routing to queue (RAG) flow")
            return "w_pgicq_kb_query"
        else:  # general
            print("→ Routing to general response")
            return "prepare_final_response"
        
    def route_pgicq_validation(self, state: GridState) -> str:
        """
        Conditional router: decides next node based on RAG response validation.
        
        FIXED: 
        - Renamed from pgicq_resp_evaluator
        - Added self parameter
        - Returns string (route name), not dict
        - Proper iteration limit check
        """
        # Safety: limit iterations
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= 3:
            print("Max iterations reached (3) - moving to final response")
            return "prepare_final_response"
        
        decision = state.get("evaluator_decision", "no").lower()
        
        print(f"Validation decision: {decision}, iteration: {iteration_count}")
        
        if decision == "yes":
            print("→ Response validated - moving to final response")
            return "prepare_final_response"
        else:
            print("→ Response needs improvement - rewriting query")
            return "w_pgicq_kb_query_rewrite"

    ####################################################################
    def prepare_final_response(self, state: GridState) -> Dict:
        """
        Route to appropriate response preparation based on query type.
        
        This is a lightweight router that delegates to module-specific functions.
        """
        print(f"\n{'='*60}")
        print("NODE: prepare_final_response - Routing to specific synthesizer")
        print(f"{'='*60}")
        
        userquery_type = state.get('userquery_type', 'general')
        print(f"Query type: {userquery_type}")
        
        # Route to appropriate module for response synthesis
        if userquery_type == "diagnostics":
            from diagevents import diagevents_synthesize_response
            return diagevents_synthesize_response(state, self.llm)
        
        elif userquery_type == "queue":
            from pgicq_kb import pgicq_synthesize_response
            return pgicq_synthesize_response(state, self.llm)
        
        else:  # general
            return self._general_response(state)
    
    def _general_response(self, state: GridState) -> Dict:
        """
        Handle general queries with direct LLM response.
        
        This stays in phase1.py as it's simple orchestration logic.
        """
        print("Generating general response")
        
        original_query = state.get('user_query', '')
        messages = state.get('messages', [])
        
        synthesis_prompt = f"""{PROMPT_SYNTHESIS}

USER'S ORIGINAL QUESTION:
{original_query}

INSTRUCTIONS:
This is a general question. Provide a helpful, informative response based on your general knowledge.

RESPONSE:"""
        
        # LLM invocation
        final_response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant who maintains conversation context."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = final_response.content
        print(f"Generated general response ({len(final_answer)} characters)")
        
        # Update conversation context
        conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."
        updated_messages = messages + [AIMessage(content=final_answer)]
        
        return {
            "query_final_resp": final_answer,
            "messages": updated_messages,
            "conversation_context": conversation_summary
        }

    ####################################################################
    def _build_graph(self) -> None:
        """
        Build the LangGraph workflow with memory.
        
        This is the orchestration layer that connects different modules.
        
        FIXED:
        - Corrected conditional edge mappings
        - Fixed routing method references
        - Proper node connections
        """
        
        workflow_builder = StateGraph(GridState)
        
        # Add nodes - using modular functions
        workflow_builder.add_node("userquery_classifier", self.userquery_classifier)
        
        workflow_builder.add_node("w_diagevents_query_prepare", self.w_diagevents_query_prepare)
        workflow_builder.add_node("w_diagevents_query_execute", self.w_diagevents_query_execute)
        
        workflow_builder.add_node("w_pgicq_kb_query", self.w_pgicq_kb_query)
        workflow_builder.add_node("w_pgicq_resp_validate", self.w_pgicq_resp_validate)
        workflow_builder.add_node("w_pgicq_kb_query_rewrite", self.w_pgicq_kb_query_rewrite)

        workflow_builder.add_node("prepare_final_response", self.prepare_final_response)
        
        # Add edges
        workflow_builder.add_edge(START, "userquery_classifier")
        
        # FIXED: Conditional routing from classifier
        workflow_builder.add_conditional_edges(
            "userquery_classifier",
            self.route_userquery,  # This returns: "w_diagevents_query_prepare", "w_pgicq_kb_query", or "prepare_final_response"
        )

        # Diagnostics flow
        workflow_builder.add_edge("w_diagevents_query_prepare", "w_diagevents_query_execute")
        workflow_builder.add_edge("w_diagevents_query_execute", "prepare_final_response")

        # Queue (RAG) flow with validation loop
        workflow_builder.add_edge("w_pgicq_kb_query", "w_pgicq_resp_validate")
        
        # FIXED: Conditional routing from validation
        workflow_builder.add_conditional_edges(
            "w_pgicq_resp_validate",
            self.route_pgicq_validation,  # This returns: "prepare_final_response" or "w_pgicq_kb_query_rewrite"
        )
        
        workflow_builder.add_edge("w_pgicq_kb_query_rewrite", "w_pgicq_kb_query")
        
        # Final response to end
        workflow_builder.add_edge("prepare_final_response", END)

        # Compile the graph WITH memory
        self.graph = workflow_builder.compile(checkpointer=self.memory)
        print("✓ Graph compiled successfully with memory support")

    ####################################################################
    def process_message(
        self, 
        message: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """
        Process a message using the Grid Chat system with memory.
        
        FIXED:
        - Proper user_query initialization
        - Better state initialization
        
        Args:
            message: User's question
            chat_history: Optional chat history (legacy support)
            thread_id: Optional thread ID for session management
        
        Returns:
            Final answer string
        """
        print(f"\n{'#'*60}")
        print(f"# PROCESSING NEW QUERY")
        print(f"{'#'*60}")
        print(f"Query: {message}\n")
        
        # Use custom thread_id if provided, otherwise use default
        session_id = thread_id or self.thread_id
        
        # Configure with thread_id for memory persistence
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 20,  # Increased for RAG validation loops
            "callbacks": [self.tracer] if self.tracer else []
        }
        
        # Get current state from memory if it exists
        try:
            current_state = self.graph.get_state(config)
            if current_state and current_state.values:
                # Continue from existing conversation
                existing_messages = current_state.values.get("messages", [])
                conversation_context = current_state.values.get("conversation_context", "")
            else:
                # New conversation
                existing_messages = []
                conversation_context = ""
        except:
            # Fresh start if no state exists
            existing_messages = []
            conversation_context = ""
        
        # FIXED: Prepare initial state with user_query properly set
        initial_state = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "user_query": message,  # FIXED: Set the actual query
            "sql_query": "",
            "query_raw_resp": [],
            "query_final_resp": "",
            "conversation_context": conversation_context,
            "iteration_count": 0,  # Initialize iteration counter
            "evaluator_decision": "",
            "evaluator_feedback": "",
            "userquery_type": ""
        }
        
        try:
            # Execute the graph
            result = self.graph.invoke(initial_state, config=config)
            
            # Extract final answer properly
            final_answer = result.get("query_final_resp", "")
            
            # Fallback to messages if query_final_resp is empty
            if not final_answer:
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage):
                        final_answer = last_msg.content
                    elif isinstance(last_msg, tuple):
                        final_answer = last_msg[1]
                    elif hasattr(last_msg, 'content'):
                        final_answer = last_msg.content
                    else:
                        final_answer = str(last_msg)
            
            print(f"\n{'#'*60}")
            print(f"# QUERY COMPLETED SUCCESSFULLY")
            print(f"{'#'*60}\n")
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n{'!'*60}")
            print(f"ERROR: {error_msg}")
            print(f"{'!'*60}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    ####################################################################
    # Utility methods for memory management
    ####################################################################
    
    def clear_memory(self, thread_id: Optional[str] = None):
        """Clear conversation memory for a specific thread."""
        session_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            self.graph.update_state(
                config,
                {
                    "messages": [],
                    "user_query": "",
                    "sql_query": "",
                    "query_raw_resp": [],
                    "query_final_resp": "",
                    "conversation_context": "",
                    "iteration_count": 0,
                    "evaluator_decision": "",
                    "evaluator_feedback": "",
                    "userquery_type": ""
                }
            )
            print(f"✓ Memory cleared for thread: {session_id}")
        except Exception as e:
            print(f"Note: Could not clear memory: {str(e)}")
    
    ####################################################################
    def get_conversation_history(self, thread_id: Optional[str] = None) -> List[BaseMessage]:
        """Get conversation history for a specific thread."""
        session_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            state = self.graph.get_state(config)
            if state and state.values:
                return state.values.get("messages", [])
            return []
        except:
            return []


####################################################################
# USAGE EXAMPLE
####################################################################
if __name__ == "__main__":
    # Initialize the chat system
    load_dotenv()
    grid_chat = GridChat()
    grid_chat.initialize()
    
    # Example conversation demonstrating different query types
    print("\n" + "="*60)
    print("TESTING GRID CHAT SYSTEM WITH QUERY ROUTING")
    print("="*60)
    
    # Test diagnostics query
    response1 = grid_chat.process_message("Show me all critical events from the last 24 hours")
    print(f"\nDiagnostics Response:\n{response1}\n")
    
    # Test queue query
    response2 = grid_chat.process_message("What is the interconnection process?")
    print(f"\nQueue Response:\n{response2}\n")
    
    # Test general query
    response3 = grid_chat.process_message("What is global warming?")
    print(f"\nGeneral Response:\n{response3}\n")
    
    # View conversation history
    history = grid_chat.get_conversation_history()
    print(f"\nConversation has {len(history)} messages")