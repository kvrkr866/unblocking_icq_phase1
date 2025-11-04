"""
##########################################################
#
# Capstone Team 16
#
#  Iteration - 1 
#
##########################################################
"""

from dotenv import load_dotenv
import os
import json
import os.path as osp
from typing import Dict, List, Optional, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from opik.integrations.langchain import OpikTracer

from griddiagnostics import gd_userquery_execute


####################################################################
# FIXED: Proper TypedDict definition for state
####################################################################
class GridState(TypedDict):
    """State for Grid Chat workflow."""
    messages: List  # Message history
    user_query: str
    sql_query: str
    query_raw_resp: List  # Database results
    query_final_resp: str


####################################################################
class GridChat:
    """Grid Chat main implementation for SQL query generation and execution."""
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.tracer = None
    
    def initialize(self) -> None:
        """Initialize components for Grid chat."""
        from langchain.chat_models import init_chat_model
        
        load_dotenv()
        # model name and parameters
        self.llm = init_chat_model(
            "gpt-5-mini",  
            model_provider="openai", 
            reasoning_effort="minimal"
        )
        print("✓ LLM initialized")
        
        # Build the graph
        self._build_graph()
        print("✓ Graph built successfully")
        # Track the graph structure:
        self.tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
    
    ####################################################################
    def query_prepare(self, state: GridState) -> Dict:
        """
        Generate SQL query from natural language user input.        
        """
        print(f"\n{'='*60}")
        print("NODE: query_prepare - Generating SQL query")
        print(f"{'='*60}")

        # Proper message extraction
        messages = state.get("messages", [])
        if isinstance(messages, list) and len(messages) > 0:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            elif hasattr(last_msg, 'content'):
                user_input = last_msg.content
            else:
                user_input = str(last_msg)
        else:
            user_input = str(messages)

        print(f"User query: {user_input}")
        
        # Improved prompt for SQL generation
        sql_generation_prompt = f"""You are a SQL expert. Generate a SQLite-compatible SQL query based on the user's question.

DATABASE SCHEMA:
- Table: severity
  Columns: severity_id (INTEGER PRIMARY KEY), severity_name (TEXT), severity_desc (TEXT)

- Table: event
  Columns: event_id (INTEGER PRIMARY KEY), event_name (TEXT), event_type (TEXT), event_desc (TEXT)

- Table: eventslog
  Columns: event_id (INTEGER), severity_id (INTEGER), timestamp (DATETIME), status (TEXT)

USER QUESTION: {user_input}

INSTRUCTIONS:
1. Generate ONLY the SQL query, no explanations
2. Use proper JOINs when multiple tables are needed
3. For "last 24 hours", use: WHERE timestamp >= datetime('now', '-1 day')
4. Include relevant columns based on the question
5. Ensure SQLite3 compatibility

OUTPUT FORMAT:
Return ONLY the SQL query, starting with SELECT and ending with semicolon.

SQL Query:"""
        
        # model invocation
        response = self.llm.invoke([
            SystemMessage(content="You are a SQL query generator expert."),
            HumanMessage(content=sql_generation_prompt)
        ])
        
        #  SQL query 
        sql_query = response.content.strip()
        
        # Check markdown code blocks if present
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
    def query_execute(self, state: GridState) -> Dict:
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
                "messages": state["messages"] + [("assistant", "Error: No SQL query generated")]
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
                "messages": state["messages"] + [("assistant", f"Error executing query: {str(e)}")]
            }

    ####################################################################
    def query_results_report(self, state: GridState) -> Dict:
        """
        Synthesize final answer from the executed query results.
        
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
        
        # Improved synthesis prompt
        synthesis_prompt = f"""You are a helpful assistant that explains database query results in plain language.

USER'S ORIGINAL QUESTION:
{original_query}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{query_tuned_resp}

INSTRUCTIONS:
1. Provide a clear, natural language answer to the user's question
2. If results were found, summarize the key information
3. If no results were found, explain this clearly
4. Use a friendly, professional tone
5. Include specific details from the results when available. 
6. If QUERY RESULTS are not relevant to QUESTION mention clearly and do not gets the results by yourself. 

RESPONSE:"""
        
        #  LLM invocation
        final_response = self.llm.invoke([
            SystemMessage(content="You are a helpful database query assistant."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = final_response.content
        print(f"\nGenerated response ({len(final_answer)} characters)")
        
        # Return proper message format
        return {
            "query_final_resp": final_answer,
            "messages": state["messages"] + [("assistant", final_answer)]
        }

    ####################################################################
    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        
        workflow_builder = StateGraph(GridState)
        
        #  nodes
        workflow_builder.add_node("query_prepare", self.query_prepare)
        workflow_builder.add_node("query_execute", self.query_execute)
        workflow_builder.add_node("query_results_report", self.query_results_report)
        
        #  edges (simple linear flow)
        workflow_builder.add_edge(START, "query_prepare")
        workflow_builder.add_edge("query_prepare", "query_execute")
        workflow_builder.add_edge("query_execute", "query_results_report")
        workflow_builder.add_edge("query_results_report", END)
        
        # Compile the graph
        self.graph = workflow_builder.compile()
        print("✓ Graph compiled successfully")

    ####################################################################
    def process_message(
        self, 
        message: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a message using the Grid Chat system.
        
        FIXED ISSUES:
        - Proper initial state structure
        - Better error handling
        - Cleaner result extraction
        """
        print(f"\n{'#'*60}")
        print(f"# PROCESSING NEW QUERY")
        print(f"{'#'*60}")
        print(f"Query: {message}\n")
        
        # FIXED: Proper initial state with correct structure
        initial_state = {
            "messages": [("user", message)],
            "user_query": "",
            "sql_query": "",
            "query_raw_resp": [],
            "query_final_resp": ""
        }
        
        try:
            # Execute the graph
            result = self.graph.invoke(
                initial_state, 
                config={
                    "recursion_limit": 10,
                    "callbacks": [self.tracer] if self.tracer else []
                }
            )
            
            # Extract final answer properly
            final_answer = result.get("query_final_resp", "")
            
            # Fallback to messages if query_final_resp is empty
            if not final_answer:
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, tuple):
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
# USAGE EXAMPLE
####################################################################
if __name__ == "__main__":
    # Initialize the chat system
    load_dotenv()
    grid_chat = GridChat()
    grid_chat.initialize()
    
    # Example queries
    test_queries = [
        "Show me all events from the last 24 hours",
        "What are the different types of events in the system?",
        "List all events with high severity",
    ]
    
    # Test with first query
    print("\n" + "="*60)
    print("TESTING GRID CHAT SYSTEM")
    print("="*60)
    
    response = grid_chat.process_message(test_queries[0])
    print(f"\nFinal Response:\n{response}")