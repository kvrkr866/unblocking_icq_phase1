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


##########################################################################
##########################################################################
# GAP ANALYSIS PROMPTS - START
##########################################################################
##########################################################################

##########################################################################
# Section Extraction and Identification
PROMPT_EXTRACT_SECTIONS = """You are an expert at analyzing technical interconnection request documents.

Analyze the following document and extract all sections with their content.

DOCUMENT TEXT:
{document_text}

INSTRUCTIONS:
1. Identify all major sections in the document (e.g., Executive Summary, Technical Specifications, Compliance, etc.)
2. Extract the content of each section
3. Identify document metadata if available (station name, region, country, request type)
4. Return structured JSON format

OUTPUT FORMAT (JSON):
{{
    "sections": [
        {{
            "section_name": "Executive Summary",
            "content": "Full content of this section...",
            "page_range": "1-2"
        }},
        {{
            "section_name": "Technical Specifications",
            "content": "Full content of this section...",
            "page_range": "3-5"
        }}
    ],
    "metadata": {{
        "station_name": "Station name if found",
        "region": "Region if found",
        "country": "Country if found",
        "request_type": "Type of request (e.g., new connection, modification)"
    }}
}}

RESPOND ONLY WITH VALID JSON."""


##########################################################################
# Section Validation
PROMPT_VALIDATE_SECTION = """You are an interconnection compliance expert reviewing document sections.

SECTION TO VALIDATE:
Section Name: {section_name}
Section Content: {section_content}

KNOWLEDGE BASE REQUIREMENTS:
{kb_requirements}

TASK:
Determine if this section meets the requirements based on the knowledge base information.

INSTRUCTIONS:
1. Compare the section content against KB requirements
2. Check for completeness, accuracy, and compliance
3. Identify what is correct
4. Cite specific KB sources (document name + page number)

OUTPUT FORMAT (JSON):
{{
    "is_correct": true/false,
    "correctness_details": "What aspects are correct",
    "references": [
        {{
            "document": "Document name",
            "page": "Page number or section",
            "relevant_requirement": "What requirement it satisfies"
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON."""


##########################################################################
# Modification Identification
PROMPT_FIND_MODIFICATIONS = """You are an interconnection compliance expert identifying needed modifications.

SECTION TO REVIEW:
Section Name: {section_name}
Section Content: {section_content}

KNOWLEDGE BASE REQUIREMENTS:
{kb_requirements}

TASK:
Identify what modifications are needed for this section to comply with requirements.

INSTRUCTIONS:
1. Compare section against KB requirements
2. Identify gaps, errors, or missing information
3. Provide specific modification recommendations
4. Cite KB sources

OUTPUT FORMAT (JSON):
{{
    "needs_modification": true/false,
    "issues_found": [
        "Issue 1: Description",
        "Issue 2: Description"
    ],
    "recommendations": [
        "Recommendation 1: Specific change needed",
        "Recommendation 2: Specific change needed"
    ],
    "references": [
        {{
            "document": "Document name",
            "page": "Page number",
            "requirement": "Requirement that is not met"
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON."""


##########################################################################
# Missing Sections Identification
PROMPT_FIND_MISSING_SECTIONS = """You are an interconnection compliance expert identifying missing required sections.

DOCUMENT SECTIONS PRESENT:
{present_sections}

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Country: {country}
Request Type: {request_type}

KNOWLEDGE BASE REQUIREMENTS:
{kb_requirements}

TASK:
Identify required sections that are missing from the document.

INSTRUCTIONS:
1. Review KB for all mandatory sections for this type of request
2. Compare against sections present in the document
3. List missing sections with their requirements
4. Cite KB sources

OUTPUT FORMAT (JSON):
{{
    "missing_sections": [
        {{
            "section_name": "Name of missing section",
            "requirement_description": "Why this section is required",
            "mandatory": true/false,
            "references": [
                {{
                    "document": "Document name",
                    "page": "Page number",
                    "requirement": "Specific requirement text"
                }}
            ]
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON. If no sections are missing, return {{"missing_sections": []}}."""


##########################################################################
# Compliance Requirements
PROMPT_COMPLIANCE_REQUIREMENTS = """You are an interconnection compliance expert extracting mandatory requirements.

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Country: {country}
Request Type: {request_type}

KNOWLEDGE BASE INFORMATION:
{kb_info}

TASK:
Extract all mandatory compliance requirements for this interconnection request.

INSTRUCTIONS:
1. List all mandatory technical compliance requirements
2. Include testing requirements, certifications, standards
3. Specify deadlines or milestones if mentioned
4. Cite KB sources with document name + page number

OUTPUT FORMAT (JSON):
{{
    "compliance_requirements": [
        {{
            "requirement_name": "Name/title of requirement",
            "description": "Detailed description",
            "category": "technical/regulatory/environmental/safety",
            "mandatory": true/false,
            "deadline": "If applicable",
            "references": [
                {{
                    "document": "Document name",
                    "page": "Page number"
                }}
            ]
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON. If not found, return: {{"compliance_requirements": [], "status": "Not able to find"}}."""


##########################################################################
# Technical Tests Requirements
PROMPT_TECHNICAL_TESTS = """You are an interconnection technical expert extracting testing requirements.

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Country: {country}
Request Type: {request_type}

KNOWLEDGE BASE INFORMATION:
{kb_info}

TASK:
Extract all required technical tests and validation procedures.

INSTRUCTIONS:
1. List all mandatory technical tests
2. Include test procedures, acceptance criteria
3. Specify equipment or system testing requirements
4. Cite KB sources

OUTPUT FORMAT (JSON):
{{
    "technical_tests": [
        {{
            "test_name": "Name of test",
            "description": "What is tested and how",
            "test_type": "commissioning/performance/safety/integration",
            "acceptance_criteria": "Pass/fail criteria",
            "timing": "When test must be performed",
            "references": [
                {{
                    "document": "Document name",
                    "page": "Page number"
                }}
            ]
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON. If not found, return: {{"technical_tests": [], "status": "Not able to find"}}."""


##########################################################################
# Queue Wait Time Analysis
PROMPT_QUEUE_WAIT_TIME = """You are an interconnection queue analyst extracting wait time information.

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Request Type: {request_type}

KNOWLEDGE BASE INFORMATION (Latest Documents):
{kb_info}

TASK:
Extract current queue wait time information and open requests.

INSTRUCTIONS:
1. Find current queue length and wait times
2. Identify open requests at the station if available
3. Find average processing times
4. Use only the LATEST document version available
5. Cite sources with document name + page number

OUTPUT FORMAT (JSON):
{{
    "queue_info": {{
        "average_wait_time": "Time duration (e.g., 12-18 months)",
        "current_queue_length": "Number of requests ahead",
        "open_requests": [
            {{
                "request_id": "ID if available",
                "capacity": "MW if available",
                "status": "Status if available"
            }}
        ],
        "last_updated": "Date of information",
        "references": [
            {{
                "document": "Document name (with version/year)",
                "page": "Page number"
            }}
        ]
    }},
    "status": "found"
}}

If information not found, return: {{"queue_info": {{}}, "status": "Not able to find", "reason": "Brief explanation"}}

RESPOND ONLY WITH VALID JSON."""


##########################################################################
# Risk Assessment
PROMPT_RISK_ASSESSMENT = """You are an interconnection risk assessment expert.

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Country: {country}
Request Type: {request_type}

KNOWLEDGE BASE INFORMATION:
{kb_info}

TASK:
Identify all relevant risks for this interconnection request.

INSTRUCTIONS:
1. Extract technical risks (grid stability, capacity, etc.)
2. Extract regulatory/compliance risks
3. Extract operational risks
4. Extract financial risks
5. Include risk mitigation strategies if available
6. Cite KB sources

OUTPUT FORMAT (JSON):
{{
    "risks": [
        {{
            "risk_name": "Name/title of risk",
            "risk_type": "technical/regulatory/operational/financial",
            "description": "Detailed risk description",
            "severity": "high/medium/low",
            "mitigation_strategy": "How to mitigate if available",
            "references": [
                {{
                    "document": "Document name",
                    "page": "Page number"
                }}
            ]
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON. Include both general risks and station-specific risks if available."""


##########################################################################
# Environmental Challenges
PROMPT_ENVIRONMENTAL_CHALLENGES = """You are an environmental compliance expert for power grid interconnections.

DOCUMENT METADATA:
Station: {station_name}
Region: {region}
Country: {country}
Request Type: {request_type}

KNOWLEDGE BASE INFORMATION:
{kb_info}

TASK:
Identify all environmental requirements and challenges.

INSTRUCTIONS:
1. Extract environmental impact assessment requirements
2. Identify environmental permits needed
3. List environmental compliance standards
4. Note seasonal or geographic challenges
5. Include mitigation measures if available
6. Cite KB sources

OUTPUT FORMAT (JSON):
{{
    "environmental_challenges": [
        {{
            "challenge_name": "Name/title",
            "category": "impact_assessment/permitting/compliance/seasonal",
            "description": "Detailed description",
            "requirements": "What must be done",
            "timeline": "If applicable",
            "references": [
                {{
                    "document": "Document name",
                    "page": "Page number"
                }}
            ]
        }}
    ]
}}

RESPOND ONLY WITH VALID JSON."""


##########################################################################
# Final Gap Report Synthesis
PROMPT_GAP_REPORT_SYNTHESIS = """You are an expert technical writer creating comprehensive gap analysis reports for interconnection requests.

DOCUMENT INFORMATION:
Document Sections: {document_sections}
Station: {station_name}
Region: {region}
Request Type: {request_type}

ANALYSIS FINDINGS:

1. CORRECT SECTIONS:
{correct_sections}

2. SECTIONS REQUIRING MODIFICATIONS:
{sections_to_modify}

3. MISSING SECTIONS:
{missing_sections}

4. MANDATORY COMPLIANCE REQUIREMENTS:
{compliance_requirements}

5. REQUIRED TECHNICAL TESTS:
{technical_tests}

6. QUEUE WAIT TIME ANALYSIS:
{queue_wait_analysis}

7. RISK ASSESSMENT:
{risk_assessment}

8. ENVIRONMENTAL CHALLENGES:
{environmental_challenges}

9. RECENT DIAGNOSTIC EVENTS (from station database):
{station_diagnostics}

TASK:
Create a comprehensive, professional gap analysis report.

REPORT STRUCTURE:

# EXECUTIVE SUMMARY
- Brief overview of document reviewed
- Key findings summary (2-3 paragraphs)
- Critical items requiring immediate attention
- Overall compliance status

# 1. DOCUMENT OVERVIEW
- Document sections present
- Metadata (station, region, type)
- Scope of analysis

# 2. SECTION-BY-SECTION ANALYSIS

## 2.1 Correct Sections
- List sections that meet requirements
- For each: what is correct and why
- Include references (Document name, Page X)

## 2.2 Sections Requiring Modifications
- List sections needing changes
- For each:
  * Current issues
  * Specific recommendations
  * References to requirements
- Use clear, actionable language

## 2.3 Missing Required Sections
- List missing sections
- For each:
  * Why it's required
  * What should be included
  * References to requirements
- Prioritize by importance (mandatory vs. recommended)

# 3. COMPLIANCE REQUIREMENTS
- Comprehensive list of mandatory requirements
- Organized by category (technical, regulatory, environmental, safety)
- Include deadlines/milestones
- All with references

# 4. TECHNICAL TESTING REQUIREMENTS
- List all required tests
- Test procedures and acceptance criteria
- Timeline for testing
- All with references

# 5. QUEUE ANALYSIS & WAIT TIME
- Current queue status
- Expected wait time
- Open requests at station (if available)
- Strategic considerations

# 6. RISK ASSESSMENT
- Identified risks by category
- Risk severity ratings
- Mitigation strategies
- All with references

# 7. ENVIRONMENTAL CHALLENGES & REQUIREMENTS
- Environmental impact considerations
- Required permits and assessments
- Compliance standards
- Timeline considerations
- All with references

# 8. RECENT STATION DIAGNOSTIC EVENTS
- Recent events from station database
- Event details (no analysis required - just list them)
- Note: This is historical data for reference

# 9. RECOMMENDATIONS & NEXT STEPS
- Prioritized action items
- Timeline suggestions
- Critical path items
- Resources needed

# 10. REFERENCES
- Complete list of all referenced documents
- Organized alphabetically

FORMATTING GUIDELINES:
- Use clear headers and subheaders
- Use bullet points for lists
- Use numbered lists for sequential steps
- Bold important terms and findings
- Include page references: [Doc Name, p. X]
- Professional, technical tone
- Clear, actionable recommendations
- No jargon without explanation

CRITICAL: If any section has "Not able to find" status, clearly state this and explain what information is missing.

Generate the complete report now:"""

##########################################################################
##########################################################################
# GAP ANALYSIS PROMPTS - END
##########################################################################
##########################################################################


##########################################################
# Usage example: create the tables in 'events_database.db'
if __name__ == "__main__":
    print(" ... NOTHING TODO, Bye ....  ")
