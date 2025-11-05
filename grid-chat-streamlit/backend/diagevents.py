"""
##########################################################
#
# Capstone Team 16
#
# Diagnostic Events Module
# Contains query preparation and execution logic
#   Author: RK (kvrkr866@gmail.com)
#
##########################################################
"""

from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from prompts import PROMPT_SQL_GENERATION
from diagevents_dbif import gd_userquery_execute


####################################################################
def diagevents_query_prepare(state: Dict, llm) -> Dict:
    """
    Generate SQL query from natural language user input with context awareness.
    
    Args:
        state: Current graph state containing messages and context
        llm: Language model instance for SQL generation
        
    Returns:
        Dictionary with updated sql_query and user_query
    """
    print(f"\n{'='*60}")
    print("NODE: diagevents_query_prepare - Generating SQL query with context")
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
    response = llm.invoke([
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
def diagevents_query_execute(state: Dict) -> Dict:
    """
    Execute the generated SQL query against the database.
    
    Args:
        state: Current graph state containing the SQL query
        
    Returns:
        Dictionary with query results
    """
    print(f"\n{'='*60}")
    print("NODE: diagevents_query_execute - Executing SQL query")
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