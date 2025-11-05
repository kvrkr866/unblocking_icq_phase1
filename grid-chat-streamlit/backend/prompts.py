##########################################################
#
# Capstone Team 16
#  List of prompts used across the project.  
# Author: RK (kvrkr866@gmail.com)
#

#######################################################################
from langchain_core.prompts import ChatPromptTemplate

# Classifier prompt
USERQUERY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
Classify the given user question into one of the specified categories based on its nature.

- diagnostics Questions: Questions related diagnostics events consists of severity or event or events list etc.  like "What are the different types of severities..?" or "List of events by severity...?" or "count of events" should be classified as 'diagnostics'.
- queue Questions: Questions related to power grid interconnection queue like "What is interconnection process...?" or "requests at queue ...?" or "to connect to the 110 kV level, what are feasible points above that leve..?" should be classified as 'queue'.

If the question does not fit into any of these categories, return 'general'.

# Steps

1. Analyze the user question.
2. Determine which category the question fits into based on its structure and keywords.
3. Return the corresponding category or 'general' if none apply.

# Output Format

- Return only the category word: 'diagnostics', 'queue', 'general'.
- Do not include any extra text or quotes in the output.

# Examples

- **Example 1**
* Question: I want to connect to CAISO, what is the interconnection process that I should follow?  
* Response: queue

- **Example 2**  
* Question: What are the expected load and generation levels in 2 years and in 5 years in [name of a specific station]?  
* Response: queue


- **Example 3**  
* Question: List all events with high severity?  
* Response: diagnostics

                                                               
- **Example 4**  
* Question: Show me critical events in the last week?  
* Response: diagnostics

                                                               
- **Example 5**  
* Question: What is impact of gloabl warming in next 2 to 5 years?  
* Response: general

User question: {question}
""")


##########################################################################
# SQL Generation Prompt Template
# NOTE: This is a regular string (not f-string) to be formatted later
PROMPT_SQL_GENERATION = """You are a SQL expert. Generate a SQLite-compatible SQL query based on the user's question.

DATABASE SCHEMA:
- Table: severity
  Columns: severity_id (INTEGER PRIMARY KEY), severity_name (TEXT), severity_desc (TEXT)

- Table: event
  Columns: event_id (INTEGER PRIMARY KEY), event_name (TEXT), event_type (TEXT), event_desc (TEXT)

- Table: eventslog
  Columns: event_id (INTEGER), severity_id (INTEGER), timestamp (DATETIME), status (TEXT)

INSTRUCTIONS:
1. Generate ONLY the SQL query, no explanations
2. Use proper JOINs when multiple tables are needed
3. For "last 24 hours", use: WHERE timestamp >= datetime('now', '-1 day')
4. Include relevant columns based on the question
5. Ensure SQLite3 compatibility

OUTPUT FORMAT:
Return ONLY the SQL query, starting with SELECT and ending with semicolon."""


##########################################################################
# Response Synthesis Prompt Template
# NOTE: This is a regular string (not f-string) to be formatted later
PROMPT_SYNTHESIS = """You are a helpful assistant that explains database query results in plain language.

INSTRUCTIONS:
1. Provide a clear, natural language answer to the user's question
2. If results were found, summarize the key information
3. If no results were found, explain this clearly
4. Use a friendly, professional tone
5. Include specific details from the results when available
6. If this question relates to previous queries in the conversation, acknowledge that context"""


##########################################################
# Usage example: create the tables in 'events_database.db'
if __name__ == "__main__":
    print(" ... NOTHING TODO, Bye ....  ")