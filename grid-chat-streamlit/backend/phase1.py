"""
##########################################################
#
# Capstone Team 16
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

from griddiagnostics import gd_userquery_execute
from prompts import PROMPT_SQL_GENERATION, PROMPT_SYNTHESIS


####################################################################
# Enhanced TypedDict definition for state with memory
####################################################################
class GridState(TypedDict):
    """State for Grid Chat workflow with conversation history."""
    messages: Annotated[List[BaseMessage], "Conversation history"]
    user_query: str
    sql_query: str
    query_raw_resp: List  # Database results
    query_final_resp: str
    conversation_context: str  # Summary of recent conversation


####################################################################
class GridChat:
    """Grid Chat main implementation with memory for contextual conversations."""
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.tracer = None
        self.memory = None
        self.thread_id = "default_session"  # Can be customized per user session
    
    def initialize(self) -> None:
        """Initialize components for Grid chat with memory."""
        from langchain.chat_models import init_chat_model
        
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
        
        # Build the graph
        self._build_graph()
        print("✓ Graph built successfully")
        
        # Track the graph structure
        self.tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
    
    ####################################################################
    def diagevents_query_prepare(self, state: GridState) -> Dict:
        """
        Generate SQL query from natural language user input with context awareness.
        """
        print(f"\n{'='*60}")
        print("NODE: query_prepare - Generating SQL query with context")
        print(f"{'='*60}")

        # Extract conversation history
        messages = state.get("messages", [])
        
        # Get the latest user message
        if isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, BaseMessage):
                user_input = last_msg.content
            elif isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            else:
                user_input = str(last_msg)
        else:
            user_input = str(messages)

        print(f"User query: {user_input}")
        
        # Get recent message history (last 3 exchanges for context)
        recent_history = ""
        if len(messages) > 1:
            history_messages = messages[-6:-1] if len(messages) > 6 else messages[:-1]
            history_lines = []
            for msg in history_messages:
                if isinstance(msg, HumanMessage):
                    history_lines.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    history_lines.append(f"Assistant: {msg.content[:100]}...")  # Truncate long responses
            if history_lines:
                recent_history = "\n\nRECENT CONVERSATION:\n" + "\n".join(history_lines) + "\n"
        
        # Build context-aware prompt with conversation context
        conversation_context = state.get("conversation_context", "")
        context_section = ""
        if conversation_context:
            context_section = f"\n\nCONVERSATION CONTEXT:\n{conversation_context}\n"
        
        # Build the complete SQL generation prompt
        sql_generation_prompt = f"""{PROMPT_SQL_GENERATION}
{recent_history}
{context_section}
CURRENT USER QUESTION: {user_input}

IMPORTANT: If the current question refers to previous queries (using words like "those", "that", "same", "previous"), 
use the context above to understand what the user is referring to.

SQL Query:"""
        
        # Model invocation with conversation history
        response = self.llm.invoke([
            SystemMessage(content="You are a SQL query generator expert who understands context from previous conversations."),
            HumanMessage(content=sql_generation_prompt)
        ])
        
        # Extract SQL query
        sql_query = response.content.strip()
        
        # Clean up markdown code blocks if present
        if sql_query.startswith("```"):
            lines = sql_query.split("\n")
            sql_query = "\n".join([line for line in lines if not line.startswith("```")])
            sql_query = sql_query.strip()
        
        # Check "sql" or "SQL" keyword if it appears at start
        if sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()
        
        print(f"\nGenerated SQL Query:\n{sql_query}\n")
        
        # Return proper state update
        return {
            "sql_query": sql_query,
            "user_query": user_input
        }

    ####################################################################
    def diagevents_query_execute(self, state: GridState) -> Dict:
        """
        Execute the generated SQL query against the database.
        
        Improvements:
        - Proper error handling
        - Better logging
        - Handle empty results
        """
        print(f"\n{'='*60}")
        print("NODE: query_execute - Executing SQL query")
        print(f"{'='*60}")
        
        sql_query = state.get('sql_query', '')
        
        if not sql_query:
            print("ERROR: No SQL query found in state")
            return {
                "query_raw_resp": [],
                "messages": state["messages"]
            }
        
        print(f"Executing: {sql_query}")
        
        try:
            # Execute the query
            query_raw_resp = gd_userquery_execute(sql_query)
            
            if query_raw_resp:
                print(f"✓ Query executed successfully - {len(query_raw_resp)} rows returned")
            else:
                print("ℹ Query executed successfully - No rows returned")
            
            return {"query_raw_resp": query_raw_resp}
            
        except Exception as e:
            print(f"ERROR executing query: {str(e)}")
            return {
                "query_raw_resp": [],
                "messages": state["messages"]
            }

    ####################################################################
    def prepare_final_response(self, state: GridState) -> Dict:
        """
        Synthesize final answer from the executed query results with context awareness.
        
        Improvements:
        - Better formatting of results
        - Handle empty results gracefully
        - Improved synthesis prompt
        - Proper message construction
        """
        print(f"\n{'='*60}")
        print("NODE: query_results_report - Generating final response")
        print(f"{'='*60}")

        query_raw_resp = state.get('query_raw_resp', [])
        original_query = state.get('user_query', '')
        sql_query = state.get('sql_query', '')
        messages = state.get('messages', [])
        
        # Better formatting of query results
        if query_raw_resp:
            formatted_results = []
            for i, row in enumerate(query_raw_resp, 1):
                formatted_results.append(f"Row {i}: {row}")
            query_tuned_resp = "\n".join(formatted_results)
            print(f"Formatting {len(query_raw_resp)} rows of results")
        else:
            query_tuned_resp = "No results found"
            print("No results to format")
        
        # Build context-aware synthesis prompt
        synthesis_prompt = f"""{PROMPT_SYNTHESIS}

USER'S ORIGINAL QUESTION:
{original_query}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{query_tuned_resp}

RESPONSE:"""
        
        # LLM invocation
        final_response = self.llm.invoke([
            SystemMessage(content="You are a helpful database query assistant who maintains conversation context."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = final_response.content
        print(f"\nGenerated response ({len(final_answer)} characters)")
        
        # Update conversation context for future queries
        conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."  # Truncated summary
        
        # Append AI response to messages
        updated_messages = messages + [AIMessage(content=final_answer)]
        
        # Return proper message format
        return {
            "query_final_resp": final_answer,
            "messages": updated_messages,
            "conversation_context": conversation_summary
        }

    ####################################################################
    def _build_graph(self) -> None:
        """Build the LangGraph workflow with memory."""
        
        workflow_builder = StateGraph(GridState)
        
        # Add nodes
        workflow_builder.add_node("diagevents_query_prepare", self.diagevents_query_prepare)
        workflow_builder.add_node("diagevents_query_execute", self.diagevents_query_execute)
        workflow_builder.add_node("prepare_final_response", self.prepare_final_response)
        
        # Add edges (simple linear flow)
        workflow_builder.add_edge(START, "diagevents_query_prepare")
        workflow_builder.add_edge("diagevents_query_prepare", "diagevents_query_execute")
        workflow_builder.add_edge("diagevents_query_execute", "prepare_final_response")
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
        
        FIXED ISSUES:
        - Proper initial state structure
        - Better error handling
        - Cleaner result extraction
        - Memory persistence across queries
        
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
            "recursion_limit": 10,
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
        
        # Prepare initial state with conversation history
        initial_state = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "user_query": "",
            "sql_query": "",
            "query_raw_resp": [],
            "query_final_resp": "",
            "conversation_context": conversation_context
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
            return error_msg
    
    ####################################################################
    def clear_memory(self, thread_id: Optional[str] = None):
        """Clear conversation memory for a specific thread."""
        session_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        # Clear the checkpoint
        try:
            # Reset to empty state
            self.graph.update_state(
                config,
                {
                    "messages": [],
                    "user_query": "",
                    "sql_query": "",
                    "query_raw_resp": [],
                    "query_final_resp": "",
                    "conversation_context": ""
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
    
    # Example conversation demonstrating memory
    print("\n" + "="*60)
    print("TESTING GRID CHAT SYSTEM WITH MEMORY")
    print("="*60)
    
    # First question
    response1 = grid_chat.process_message("Show me all critical events from the last 24 hours")
    print(f"\nResponse 1:\n{response1}\n")
    
    # Follow-up question using context
    response2 = grid_chat.process_message("How many of those were related to power outages?")
    print(f"\nResponse 2:\n{response2}\n")
    
    # Another follow-up
    response3 = grid_chat.process_message("What about in the last week?")
    print(f"\nResponse 3:\n{response3}\n")
    
    # View conversation history
    history = grid_chat.get_conversation_history()
    print(f"\nConversation has {len(history)} messages")
    
    # Clear memory if needed
    # grid_chat.clear_memory()