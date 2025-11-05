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
    osp.join(BASE_DIR, "RIOO Processes with RIOO-Create.pdf"),
]
#### constants of Power grid interconnection Queue RAG Input data - END #####

#### constants of Power grid interconnection Queue - START #####################
PGICQ_KB_PERSIST_DIRECTORY="./databases/pgicq_kb"
PGICQ_KB_COLLECTION_NAME="pgicq_collection_1"
#### constants of Power grid interconnection Queue - END ######################


####################################################################
# Class declarations - START 
####################################################################

#### Class for GridState to use with Langgraph ######################
class GridState(TypedDict):
    """State for Grid Chat workflow with conversation history."""
    messages: Annotated[List[BaseMessage], "Conversation history"]
    user_query: str
    sql_query: str
    query_raw_resp: List  # Database results
    query_final_resp: str
    conversation_context: str  # Summary of recent conversation
    iteration_count: int
    evaluator_decision: str
    evaluator_feedback: str
    userquery_type: str

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

