###############################################################################
#
# Capstone Team 16
#  List of constants used across the project.  
# Author: RK (kvrkr866@gmail.com)

###############################################################################

import json
import os.path as osp

from typing import Dict, List, Optional, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field

from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage


###############################################################################
#### Constants of Grid Diagnostics  START #####################################
DB_NAME = 'gridevents.db'
SQL_DBNAME_WITH_PATH='./databases/gridevents.db'
TABLE_EVENT = 'event'
TABLE_SEVERITY  = 'severity'
TABLE_EVENTSLOG = 'eventslog'
#### Constants of Grid Diagnostics  END #######################################


#### constants of Power grid interconnection Queue RAG Input data - START #####
BASE_DIR = "../grid_docs"
PGICQ_DOCS_FILE_PATHS = [
    osp.join(BASE_DIR, "100517decisiononinterconnectionrequirementsreform_renewableresources-attacha-gecomments.pdf"),
    osp.join(BASE_DIR, "Creating-an-INR-for-a-Resource-Smaller-than-10MW.pdf"),
    osp.join(BASE_DIR, "Creating-an-INR-for-Resource-Larger-than-10MW.pdf"),
    osp.join(BASE_DIR, "Managing-INRs-as-a-TSP-or-DSP.pdf"),
    osp.join(BASE_DIR, "Managing-Your-INR-as-an-IE-or-RE.pdf"),
    osp.join(BASE_DIR, "new-resource-implementation-guide.pdf"),
    osp.join(BASE_DIR, "revised-draft-2024-2025-transmission-plan.pdf"),
    osp.join(BASE_DIR, "Nov_2023_Changes_to_RIOO_IS.pdf"),
    osp.join(BASE_DIR, "Resource_Registration_Guide_v5.5_110119.pdf"),
    osp.join(BASE_DIR, "RIOO_Processes_with_RIOO-Create.pdf"),
]
#### constants of Power grid interconnection Queue RAG Input data - END #####

#### constants of Power grid interconnection Queue - START #####################
PGICQ_KB_PERSIST_DIRECTORY="./databases/pgicq_kb"
PGICQ_KB_COLLECTION_NAME="pgicq_collection_1"
#### constants of Power grid interconnection Queue - END ######################

#### constants of Gap Analysis - START #########################################
GAP_ANALYSIS_UPLOAD_DIR = "./uploads_gap_analysis"
GAP_ANALYSIS_REPORTS_DIR = "./reports_gap_analysis"
#### constants of Gap Analysis - END ###########################################


####################################################################
# Class declarations - START 
####################################################################

#### Class for GridState to use with Langgraph ######################
class GridState(TypedDict):
    """State for Grid Chat workflow with conversation history and sequential flow support."""
    messages: Annotated[List[BaseMessage], "Conversation history"]
    user_query: str
    sql_query: str
    
    # Diagnostics results
    query_raw_resp: List  # Database results from diagnostics queries
    
    # Queue results (for sequential flow)
    queue_raw_resp: List  # RAG results from queue queries
    
    # Final response
    query_final_resp: str
    conversation_context: str  # Summary of recent conversation
    
    # Iteration control
    iteration_count: int
    evaluator_decision: str
    evaluator_feedback: str
    
    # Query classification
    userquery_type: str
    
    # Station filtering (for future use)
    station_filter: Optional[List[str]]  # Station IDs/names for filtering diagnostics
    has_queue_info: bool  # Flag indicating if queue returned results
    
    # NEW: Gap Analysis fields
    is_gap_analysis: bool  # Flag to indicate gap analysis mode
    uploaded_pdf_path: Optional[str]  # Path to uploaded PDF
    extracted_text: Optional[str]  # Full text from PDF
    document_sections: Optional[Dict[str, str]]  # section_name: content mapping
    document_metadata: Optional[Dict[str, str]]  # station, region, country, type
    
    # Gap analysis findings (from RAG queries)
    correct_sections: Optional[List[Dict]]  # Sections that are correct with references
    sections_to_modify: Optional[List[Dict]]  # Sections needing changes with details
    missing_sections: Optional[List[Dict]]  # Required sections not in document
    compliance_requirements: Optional[List[Dict]]  # Mandatory compliance list
    technical_tests: Optional[List[Dict]]  # Required technical tests
    queue_wait_analysis: Optional[Dict]  # Queue wait time info
    risk_assessment: Optional[List[Dict]]  # Risk information
    environmental_challenges: Optional[List[Dict]]  # Environmental requirements
    
    # Diagnostics events (no analysis, just raw data)
    station_diagnostics: Optional[List[Dict]]  # Recent events from DB
    
    # Final gap analysis report
    gap_report_content: Optional[str]  # Report content (markdown/text)
    gap_report_path: Optional[str]  # Path to generated DOCX/PDF
    
    # Progress tracking
    current_step: Optional[str]  # Current processing step
    progress_percent: Optional[int]  # Progress percentage (0-100)

#### Class for userquery classification  ######################
class Userqueryclassifier(BaseModel):
    userquery_type: str = Field(
        description="Relevane score: 'diagnostics' or 'queue' or 'general' "
    ) 

#### Evaluator Class for RAG response relevance check  ######################
class Evaluator(BaseModel):
    decision: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
    feedback: Optional[str] = Field(
        description="Reason for 'no' and possible suggestion for rewriting the query",
        default=""
    )

##########################################################
# Usage example: create the tables in 'events_database.db'
if __name__ == "__main__":
    print(" ... NOTHING TODO, Bye ....  ")
