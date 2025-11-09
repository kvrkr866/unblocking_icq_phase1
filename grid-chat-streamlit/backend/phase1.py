"""
##########################################################
#
# Capstone Team 16: Power Grids Interconnection Queue Analyzer
#
#  Author: RK (kvrkr866@gmail.com)
#
#  Phase1: 
#     Iteration 1 - Basic functionality
#     Iteration 2 - Query routing and RAG
#     Iteration 3 - Sequential Queueâ†’Diagnostics flow
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


from prompts import PROMPT_SYNTHESIS, USERQUERY_CLASSIFIER_PROMPT
from constants import (
    PGICQ_DOCS_FILE_PATHS, 
    Evaluator, 
    Userqueryclassifier, 
    GridState,
    GAP_ANALYSIS_REPORTS_DIR,
    GAP_ANALYSIS_UPLOAD_DIR
)


# Import modular components
from diagevents import diagevents_query_prepare, diagevents_query_execute
from gap_analysis_docgen import generate_gap_analysis_docx

from pgicq_gap_analysis import (
    pgicq_gap_extract_pdf_and_sections,
    pgicq_gap_validate_sections,
    pgicq_gap_identify_modifications,
    pgicq_gap_find_missing_sections,
    pgicq_gap_fetch_compliance_requirements,
    pgicq_gap_fetch_technical_tests,
    pgicq_gap_analyze_queue_wait,
    pgicq_gap_assess_risks,
    pgicq_gap_identify_environmental,
    pgicq_gap_fetch_diagnostics,
    pgicq_gap_synthesize_report,
    pgicq_gap_generate_report_file
)

from pgicq_kb import (
    pgicq_rag_initialize,  # NEW
    pgicq_kb_create_n_initialize, 
    pgicq_kb_query, 
    pgicq_resp_validate, 
    pgicq_kb_query_rewrite
)


####################################################################
class GridChat:
    """
    Grid Chat main orchestrator with sequential queueâ†’diagnostics flow.
    
    For 'queue' queries:
    1. First: Fetch queue information (RAG)
    2. Then: Fetch related diagnostic events (SQL) - if queue info found
    3. Finally: Combine both in response
    """
    
    def __init__(self):
        self.llm = None
        self.graph = None
        self.tracer = None
        self.memory = None
        self.retriever = None
        self.vector_store = None
        self.thread_id = "default_session"
    
    def initialize(self) -> None:
        """Initialize components for Grid chat with memory."""
        from langchain.chat_models import init_chat_model
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from constants import PGICQ_KB_PERSIST_DIRECTORY, PGICQ_KB_COLLECTION_NAME

        load_dotenv()
        
        # Initialize LLM
        self.llm = init_chat_model(
            "gpt-5-mini",  
            model_provider="openai", 
            reasoning_effort="minimal"
        )
        print("âœ“ LLM initialized")
        
        # Initialize memory
        self.memory = MemorySaver()
        print("âœ“ Memory initialized")

        # Wrapper for RAG initialization
        self.vector_store, self.retriever, self.embeddings = pgicq_rag_initialize()

        # Build the graph
        self._build_graph()
        print("âœ“ Graph built successfully")
        
        # Initialize tracer
        try:
            self.tracer = OpikTracer(graph=self.graph.get_graph(xray=True))
            print("âœ“ Opik tracer initialized")
        except Exception as e:
            print(f"âš  Opik tracer not available: {e}")
            self.tracer = None
    
    ####################################################################
    # Node functions
    ####################################################################
    
    def userquery_classifier(self, state: GridState) -> Dict:
        """Classify user query into diagnostics, queue, or general."""
        print(f"\n{'='*60}")
        print("NODE: userquery_classifier")
        print(f"{'='*60}")
        
        user_query = state.get("user_query", "")
        if not user_query and state.get("messages"):
            last_msg = state["messages"][-1]
            user_query = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        
        print(f"Classifying: {user_query}")
        
        classifier_chain = USERQUERY_CLASSIFIER_PROMPT | self.llm | StrOutputParser()
        userquery_type = classifier_chain.invoke({"question": user_query}).strip().lower()
        
        valid_categories = ["diagnostics", "queue", "general"]
        if userquery_type not in valid_categories:
            userquery_type = "general"
        
        print(f"â†’ Classified as: {userquery_type}")
        
        return {
            "userquery_type": userquery_type,
            "user_query": user_query
        }
    
    ####################################################################
    # Wrapper methods
    ####################################################################

    def w_diagevents_query_prepare(self, state: GridState) -> Dict:
        """Wrapper for diagnostics query preparation."""
        return diagevents_query_prepare(state, self.llm)
    
    def w_diagevents_query_execute(self, state: GridState) -> Dict:
        """Wrapper for diagnostics query execution."""
        return diagevents_query_execute(state)

    def w_pgicq_kb_query(self, state: GridState) -> Dict:
        """Wrapper for queue KB query."""
        return pgicq_kb_query(state, self.retriever, self.vector_store)

    def w_pgicq_resp_validate(self, state: GridState) -> Dict:
        """Wrapper for queue response validation."""
        return pgicq_resp_validate(state, self.llm)

    def w_pgicq_kb_query_rewrite(self, state: GridState) -> Dict:
        """Wrapper for query rewrite."""
        return pgicq_kb_query_rewrite(state, self.llm)
    
    ####################################################################
    # NEW: Sequential flow nodes for queueâ†’diagnostics
    ####################################################################
    
    def extract_station_info(self, state: GridState) -> Dict:
        """
        Extract station/location identifiers from queue results.
        
        FUTURE: When station info is available in DB, this will extract
        specific station IDs to filter diagnostic events.
        
        CURRENT: Returns all events (no station filter yet).
        """
        print(f"\n{'='*60}")
        print("NODE: extract_station_info")
        print(f"{'='*60}")
        
        queue_raw_resp = state.get("queue_raw_resp", [])
        
        if not queue_raw_resp:
            print("No queue results to extract from")
            return {
                "station_filter": None,
                "has_queue_info": False
            }
        
        # FUTURE: Extract station identifiers from queue results
        # Example logic (to be implemented when DB has station info):
        # station_names = []
        # for doc in queue_raw_resp:
        #     content = doc.get("content", "")
        #     # Use LLM or regex to extract: "Station XYZ", "Location ABC"
        #     stations = extract_stations_from_text(content)
        #     station_names.extend(stations)
        
        # CURRENT: No station filtering - get all events
        print("Station filtering not yet implemented - will fetch all events")
        print("FUTURE: Will extract station IDs from queue results for filtering")
        
        return {
            "station_filter": None,  # FUTURE: Will be ["Station XYZ", "Station ABC"]
            "has_queue_info": True
        }
    
    def fetch_diagnostics_for_queue(self, state: GridState) -> Dict:
        """
        Fetch diagnostic events related to the queue information.
        
        CURRENT: Fetches all recent events
        FUTURE: Filters by station when available
        """
        print(f"\n{'='*60}")
        print("NODE: fetch_diagnostics_for_queue")
        print(f"{'='*60}")
        
        station_filter = state.get("station_filter")
        user_query = state.get("user_query", "")
        
        # Build SQL query based on station filter availability
        if station_filter and len(station_filter) > 0:
            # FUTURE: When station column exists in DB
            print(f"Filtering events for stations: {station_filter}")
            station_conditions = " OR ".join([f"station_name = '{s}'" for s in station_filter])
            sql_query = f"""
SELECT e.event_id, e.event_name, e.event_type, 
       s.severity_name, l.timestamp, l.status
FROM eventslog l
JOIN event e ON l.event_id = e.event_id
LEFT JOIN severity s ON l.severity_id = s.severity_id
WHERE ({station_conditions})
  AND l.timestamp >= datetime('now', '-7 days')
ORDER BY l.timestamp DESC
LIMIT 50;
"""
        else:
            # CURRENT: Get all recent events (no station filter)
            print("No station filter - fetching all recent diagnostic events")
            sql_query = """
SELECT e.event_id, e.event_name, e.event_type, 
       s.severity_name, l.timestamp, l.status
FROM eventslog l
JOIN event e ON l.event_id = e.event_id
LEFT JOIN severity s ON l.severity_id = s.severity_id
WHERE l.timestamp >= datetime('now', '-7 days')
ORDER BY l.timestamp DESC
LIMIT 50;
"""
        
        print(f"Generated SQL:\n{sql_query}")
        
        # Execute query
        from diagevents_dbif import gd_userquery_execute
        
        try:
            query_raw_resp = gd_userquery_execute(sql_query)
            print(f"âœ“ Retrieved {len(query_raw_resp)} diagnostic events")
            
            return {
                "sql_query": sql_query,
                "query_raw_resp": query_raw_resp
            }
        except Exception as e:
            print(f"ERROR executing diagnostics query: {e}")
            return {
                "sql_query": sql_query,
                "query_raw_resp": []
            }

    ####################################################################
    # Routing methods
    ####################################################################
    
    def route_userquery(self, state: GridState) -> str:
        """Route based on classification."""
        userquery_type = state.get("userquery_type", "general").lower()
        
        print(f"\nRouting: {userquery_type}")
        
        if userquery_type == "diagnostics":
            print("â†’ Diagnostics-only flow")
            return "w_diagevents_query_prepare"
        elif userquery_type == "queue":
            print("â†’ Sequential Queueâ†’Diagnostics flow")
            return "w_pgicq_kb_query"
        else:
            print("â†’ General response")
            return "prepare_final_response"
    
    def route_after_queue(self, state: GridState) -> str:
        """
        Route after queue KB query.
        Check if we got queue results to proceed with diagnostics.
        """
        has_queue_info = state.get("has_queue_info", False)
        queue_raw_resp = state.get("queue_raw_resp", [])
        
        # Check if queue returned meaningful results
        if queue_raw_resp and len(queue_raw_resp) > 0:
            print("â†’ Queue info found - proceeding to diagnostics")
            return "extract_station_info"
        else:
            print("â†’ No queue info found - skipping diagnostics")
            return "prepare_final_response"
    
    def route_pgicq_validation(self, state: GridState) -> str:
        """Route based on validation (for queue query rewrite loop)."""
        iteration_count = state.get("iteration_count", 0)
        if iteration_count >= 3:
            print("Max iterations - checking queue results")
            return "check_queue_results"
        
        decision = state.get("evaluator_decision", "no").lower()
        
        if decision == "yes":
            print("â†’ Validation passed - checking queue results")
            return "check_queue_results"
        else:
            print("â†’ Rewriting query")
            return "w_pgicq_kb_query_rewrite"

    ####################################################################
    def prepare_final_response(self, state: GridState) -> Dict:
        """Route to appropriate response synthesis."""
        print(f"\n{'='*60}")
        print("NODE: prepare_final_response")
        print(f"{'='*60}")
        
        userquery_type = state.get('userquery_type', 'general')
        has_queue_info = state.get("has_queue_info", False)
        
        # For queue queries with diagnostics, use combined response
        if userquery_type == "queue" and has_queue_info:
            print("â†’ Combined queue + diagnostics response")
            return self._combined_queue_diagnostics_response(state)
        
        # For diagnostics-only
        elif userquery_type == "diagnostics":
            print("â†’ Diagnostics-only response")
            from diagevents import diagevents_synthesize_response
            return diagevents_synthesize_response(state, self.llm)
        
        # For queue-only (no diagnostics)
        elif userquery_type == "queue":
            print("â†’ Queue-only response")
            from pgicq_kb import pgicq_synthesize_response
            return pgicq_synthesize_response(state, self.llm)
        
        # For general
        else:
            print("â†’ General response")
            return self._general_response(state)
    
    def _combined_queue_diagnostics_response(self, state: GridState) -> Dict:
        """
        Combine queue information with related diagnostic events.
        PRIMARY: Queue information
        SECONDARY: Related diagnostic events
        """
        print("Synthesizing combined response")
        
        queue_raw_resp = state.get('queue_raw_resp', [])
        diagnostics_raw_resp = state.get('query_raw_resp', [])
        original_query = state.get('user_query', '')
        messages = state.get('messages', [])
        
        # Format queue information (PRIMARY)
        if queue_raw_resp:
            queue_docs = []
            for i, doc in enumerate(queue_raw_resp[:5], 1):
                if isinstance(doc, dict):
                    content = doc.get("content", "")[:400]
                    filename = doc.get("filename", "unknown")
                    queue_docs.append(f"[Source {i}: {filename}]\n{content}")
            queue_info = "\n\n".join(queue_docs)
        else:
            queue_info = "No queue information found"
        
        # Format diagnostic events (SECONDARY)
        if diagnostics_raw_resp:
            event_lines = []
            for i, row in enumerate(diagnostics_raw_resp[:20], 1):
                event_lines.append(f"Event {i}: {row}")
            diagnostics_info = "\n".join(event_lines)
        else:
            diagnostics_info = "No related diagnostic events found"
        
        # Build combined synthesis prompt
        synthesis_prompt = f"""{PROMPT_SYNTHESIS}

USER QUESTION:
{original_query}

PRIMARY: POWER GRID INTERCONNECTION QUEUE INFORMATION
{queue_info}

SECONDARY: RELATED DIAGNOSTIC EVENTS (Last 7 days)
{diagnostics_info}

INSTRUCTIONS:
1. Start with the queue/interconnection information (this is the primary focus)
2. Then present related diagnostic events as supporting context
3. Clearly structure the response: Queue Info first, then Events
4. If no events found, mention cleary "No Events found" 

RESPONSE:"""
        
        # LLM invocation
        final_response = self.llm.invoke([
            SystemMessage(content="You are a power grid interconnection expert. Present queue information first, then related events."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = final_response.content
        print(f"Generated combined response ({len(final_answer)} chars)")
        
        # Update conversation context
        conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."
        updated_messages = messages + [AIMessage(content=final_answer)]
        
        return {
            "query_final_resp": final_answer,
            "messages": updated_messages,
            "conversation_context": conversation_summary
        }
    
    def _general_response(self, state: GridState) -> Dict:
        """Handle general queries."""
        print("Generating general response")
        
        original_query = state.get('user_query', '')
        messages = state.get('messages', [])
        
        synthesis_prompt = f"""{PROMPT_SYNTHESIS}

USER QUESTION: {original_query}

INSTRUCTIONS: Provide a helpful response based on general knowledge.

RESPONSE:"""
        
        final_response = self.llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        final_answer = final_response.content
        conversation_summary = f"Q: {original_query}\nA: {final_answer[:200]}..."
        updated_messages = messages + [AIMessage(content=final_answer)]
        
        return {
            "query_final_resp": final_answer,
            "messages": updated_messages,
            "conversation_context": conversation_summary
        }

    ####################################################################
    # GAP ANALYSIS WRAPPER METHODS - START
    ####################################################################
    def extract_pdf_and_sections(self, state: GridState) -> Dict:
        """Wrapper for gap analysis PDF extraction."""
        return pgicq_gap_extract_pdf_and_sections(state, self.llm)
    
    def validate_sections_rag(self, state: GridState) -> Dict:
        """Wrapper for gap analysis section validation."""
        return pgicq_gap_validate_sections(state, self.llm, self.retriever, self.vector_store)
    
    def identify_modifications_rag(self, state: GridState) -> Dict:
        """Wrapper for gap analysis modification identification."""
        return pgicq_gap_identify_modifications(state, self.llm, self.retriever, self.vector_store)
    
    def find_missing_sections_rag(self, state: GridState) -> Dict:
        """Wrapper for gap analysis missing section identification."""
        return pgicq_gap_find_missing_sections(state, self.llm, self.retriever, self.vector_store)
    
    def fetch_compliance_reqs_rag(self, state: GridState) -> Dict:
        """Wrapper for fetching compliance requirements."""
        return pgicq_gap_fetch_compliance_requirements(state, self.retriever, self.vector_store)
    
    def fetch_technical_tests_rag(self, state: GridState) -> Dict:
        """Wrapper for fetching technical tests."""
        return pgicq_gap_fetch_technical_tests(state, self.retriever, self.vector_store)
    
    def analyze_queue_rag(self, state: GridState) -> Dict:
        """Wrapper for queue wait analysis."""
        return pgicq_gap_analyze_queue_wait(state, self.retriever, self.vector_store)
    
    def assess_risks_rag(self, state: GridState) -> Dict:
        """Wrapper for risk assessment."""
        return pgicq_gap_assess_risks(state, self.retriever, self.vector_store)
    
    def identify_environmental_rag(self, state: GridState) -> Dict:
        """Wrapper for environmental challenge identification."""
        return pgicq_gap_identify_environmental(state, self.retriever, self.vector_store)
    
    def fetch_diagnostics_for_gap(self, state: GridState) -> Dict:
        """Wrapper for fetching diagnostic events."""
        return pgicq_gap_fetch_diagnostics(state)
    
    def synthesize_gap_report(self, state: GridState) -> Dict:
        """Wrapper for report synthesis."""
        return pgicq_gap_synthesize_report(state, self.llm)
    
    def generate_gap_report_file(self, state: GridState) -> Dict:
        """Wrapper for report file generation."""
        return pgicq_gap_generate_report_file(state)

    ####################################################################
    # GAP ANALYSIS WRAPPER METHODS - END 
    ####################################################################

    def process_gap_analysis(
        self,
        pdf_path: str,
        thread_id: Optional[str] = None
    ) -> Dict:
        """
        Process gap analysis for uploaded PDF.
        
        Args:
            pdf_path: Path to uploaded PDF file
            thread_id: Optional thread ID for conversation
            
        Returns:
            Dictionary with report_path and status
        """
        print(f"\n{'='*60}")
        print(f"STARTING GAP ANALYSIS")
        print(f"PDF: {pdf_path}")
        print(f"{'='*60}\n")
        
        session_id = thread_id or self.thread_id
        
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 30,  # Increased for gap analysis
        }

        if self.tracer:
            config["callbacks"] = [self.tracer]
            print("✓ Tracing enabled for gap analysis")
        
        # Initialize gap analysis state
        initial_state = {
            "messages": [],
            "user_query": "Gap Analysis Request",
            "sql_query": "",
            "query_raw_resp": [],
            "queue_raw_resp": [],
            "query_final_resp": "",
            "conversation_context": "",
            "iteration_count": 0,
            "evaluator_decision": "",
            "evaluator_feedback": "",
            "userquery_type": "gap_analysis",
            "station_filter": None,
            "has_queue_info": False,
            
            # Gap analysis specific
            "is_gap_analysis": True,
            "uploaded_pdf_path": pdf_path,
            "extracted_text": None,
            "document_sections": None,
            "document_metadata": None,
            "correct_sections": None,
            "sections_to_modify": None,
            "missing_sections": None,
            "compliance_requirements": None,
            "technical_tests": None,
            "queue_wait_analysis": None,
            "risk_assessment": None,
            "environmental_challenges": None,
            "station_diagnostics": None,
            "gap_report_content": None,
            "gap_report_path": None,
            "current_step": "Starting analysis",
            "progress_percent": 0,
        }
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            return {
                "success": True,
                "report_path": result.get("gap_report_path"),
                "status": result.get("current_step"),
                "progress": result.get("progress_percent")
            }
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\nGAP ANALYSIS ERROR: {error_msg}\n")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "report_path": None,
                "status": error_msg,
                "progress": 0
            }

        ####################################################################
        # GAP ANALYSIS NODE FUNCTIONS - END
        ####################################################################

    ####################################################################
    def _build_graph(self) -> None:
        """Build the LangGraph workflow with sequential queueâ†’diagnostics flow."""
        
        workflow_builder = StateGraph(GridState)
        
        # Add all nodes
        workflow_builder.add_node("userquery_classifier", self.userquery_classifier)
        
        # Diagnostics-only flow
        workflow_builder.add_node("w_diagevents_query_prepare", self.w_diagevents_query_prepare)
        workflow_builder.add_node("w_diagevents_query_execute", self.w_diagevents_query_execute)
        
        # Queue flow with validation
        workflow_builder.add_node("w_pgicq_kb_query", self.w_pgicq_kb_query)
        workflow_builder.add_node("w_pgicq_resp_validate", self.w_pgicq_resp_validate)
        workflow_builder.add_node("w_pgicq_kb_query_rewrite", self.w_pgicq_kb_query_rewrite)
        
        # NEW: Sequential queueâ†’diagnostics nodes
        workflow_builder.add_node("extract_station_info", self.extract_station_info)
        workflow_builder.add_node("fetch_diagnostics_for_queue", self.fetch_diagnostics_for_queue)
        
        # Final response
        workflow_builder.add_node("prepare_final_response", self.prepare_final_response)

        # NEW: GAP ANALYSIS NODES
        workflow_builder.add_node("extract_pdf_and_sections", self.extract_pdf_and_sections)
        workflow_builder.add_node("validate_sections_rag", self.validate_sections_rag)
        workflow_builder.add_node("identify_modifications_rag", self.identify_modifications_rag)
        workflow_builder.add_node("find_missing_sections_rag", self.find_missing_sections_rag)
        workflow_builder.add_node("fetch_compliance_reqs_rag", self.fetch_compliance_reqs_rag)
        workflow_builder.add_node("fetch_technical_tests_rag", self.fetch_technical_tests_rag)
        workflow_builder.add_node("analyze_queue_rag", self.analyze_queue_rag)
        workflow_builder.add_node("assess_risks_rag", self.assess_risks_rag)
        workflow_builder.add_node("identify_environmental_rag", self.identify_environmental_rag)
        workflow_builder.add_node("fetch_diagnostics_for_gap", self.fetch_diagnostics_for_gap)
        workflow_builder.add_node("synthesize_gap_report", self.synthesize_gap_report)
        workflow_builder.add_node("generate_gap_report_file", self.generate_gap_report_file)
        
        # NEW: Updated conditional routing to handle gap analysis
        def route_from_classifier(state: GridState) -> str:
            """Route based on query type or gap analysis flag"""
            if state.get("is_gap_analysis", False):
                return "extract_pdf_and_sections"
            else:
                # Existing routing logic
                return self.route_userquery(state)
        
        workflow_builder.add_conditional_edges(
            "userquery_classifier",
            route_from_classifier
        )

        # Build edges
        workflow_builder.add_edge(START, "userquery_classifier")
        
        # Route from classifier
        workflow_builder.add_conditional_edges(
            "userquery_classifier",
            self.route_userquery,
        )

        # Diagnostics-only flow (existing)
        workflow_builder.add_edge("w_diagevents_query_prepare", "w_diagevents_query_execute")
        workflow_builder.add_edge("w_diagevents_query_execute", "prepare_final_response")

        # Queue flow with validation
        workflow_builder.add_edge("w_pgicq_kb_query", "w_pgicq_resp_validate")
        workflow_builder.add_conditional_edges(
            "w_pgicq_resp_validate",
            self.route_pgicq_validation,
        )
        workflow_builder.add_edge("w_pgicq_kb_query_rewrite", "w_pgicq_kb_query")
        
        # NEW: Sequential flow after queue validation passes
        # Create a named node for conditional routing
        def check_queue_results(state: GridState) -> str:
            """Check if queue returned results."""
            return self.route_after_queue(state)
        
        workflow_builder.add_node("check_queue_results", lambda state: state)
        workflow_builder.add_conditional_edges(
            "check_queue_results",
            check_queue_results,
        )
        
        # Connect validation success to check
        workflow_builder.add_edge("w_pgicq_resp_validate", "check_queue_results")
        
        # Queueâ†’Diagnostics sequential flow
        workflow_builder.add_edge("extract_station_info", "fetch_diagnostics_for_queue")
        workflow_builder.add_edge("fetch_diagnostics_for_queue", "prepare_final_response")
        
        # Final
        workflow_builder.add_edge("prepare_final_response", END)

        # NEW: GAP ANALYSIS SEQUENTIAL FLOW
        workflow_builder.add_edge("extract_pdf_and_sections", "validate_sections_rag")
        workflow_builder.add_edge("validate_sections_rag", "identify_modifications_rag")
        workflow_builder.add_edge("identify_modifications_rag", "find_missing_sections_rag")
        workflow_builder.add_edge("find_missing_sections_rag", "fetch_compliance_reqs_rag")
        workflow_builder.add_edge("fetch_compliance_reqs_rag", "fetch_technical_tests_rag")
        workflow_builder.add_edge("fetch_technical_tests_rag", "analyze_queue_rag")
        workflow_builder.add_edge("analyze_queue_rag", "assess_risks_rag")
        workflow_builder.add_edge("assess_risks_rag", "identify_environmental_rag")
        workflow_builder.add_edge("identify_environmental_rag", "fetch_diagnostics_for_gap")
        workflow_builder.add_edge("fetch_diagnostics_for_gap", "synthesize_gap_report")
        workflow_builder.add_edge("synthesize_gap_report", "generate_gap_report_file")
        workflow_builder.add_edge("generate_gap_report_file", END)

        # Compile with memory
        self.graph = workflow_builder.compile(checkpointer=self.memory)
        print("âœ“ Graph with sequential queueâ†’diagnostics flow compiled")

    ####################################################################
    def process_message(
        self, 
        message: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """Process a message using the Grid Chat system."""
        print(f"\n{'#'*60}")
        print(f"# PROCESSING NEW QUERY")
        print(f"{'#'*60}")
        print(f"Query: {message}\n")
        
        session_id = thread_id or self.thread_id
        
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 25,  # Increased for sequential flow
        }

        if self.tracer:
            config["callbacks"] = [self.tracer]
            print("âœ“ Tracing enabled")

        # Get existing state
        try:
            current_state = self.graph.get_state(config)
            if current_state and current_state.values:
                existing_messages = current_state.values.get("messages", [])
                conversation_context = current_state.values.get("conversation_context", "")
            else:
                existing_messages = []
                conversation_context = ""
        except:
            existing_messages = []
            conversation_context = ""
        
        # Initialize state
        initial_state = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "user_query": message,
            "sql_query": "",
            "query_raw_resp": [],
            "queue_raw_resp": [],  # NEW
            "query_final_resp": "",
            "conversation_context": conversation_context,
            "iteration_count": 0,
            "evaluator_decision": "",
            "evaluator_feedback": "",
            "userquery_type": "",
            "station_filter": None,  # NEW
            "has_queue_info": False,  # NEW

            # NEW: Gap analysis fields
            "is_gap_analysis": False,  # Set to True from UI
            "uploaded_pdf_path": None,
            "extracted_text": None,
            "document_sections": None,
            "document_metadata": None,
            "correct_sections": None,
            "sections_to_modify": None,
            "missing_sections": None,
            "compliance_requirements": None,
            "technical_tests": None,
            "queue_wait_analysis": None,
            "risk_assessment": None,
            "environmental_challenges": None,
            "station_diagnostics": None,
            "gap_report_content": None,
            "gap_report_path": None,
            "current_step": None,
            "progress_percent": 0,

        }
        
        try:
            result = self.graph.invoke(initial_state, config=config)
            
            final_answer = result.get("query_final_resp", "")
            
            if not final_answer:
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage):
                        final_answer = last_msg.content
                    elif hasattr(last_msg, 'content'):
                        final_answer = last_msg.content
                    else:
                        final_answer = str(last_msg)
            
            print(f"\n{'#'*60}")
            print(f"# QUERY COMPLETED")
            print(f"{'#'*60}\n")
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n{'!'*60}")
            print(f"ERROR: {error_msg}")
            print(f"{'!'*60}\n")
            import traceback
            traceback.print_exc()
            return error_msg
    
    ####################################################################
    # Utility methods
    ####################################################################
    
    def clear_memory(self, thread_id: Optional[str] = None):
        """Clear conversation memory."""
        session_id = thread_id or self.thread_id
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            self.graph.update_state(config, {
                "messages": [],
                "user_query": "",
                "sql_query": "",
                "query_raw_resp": [],
                "queue_raw_resp": [],
                "query_final_resp": "",
                "conversation_context": "",
                "iteration_count": 0,
                "evaluator_decision": "",
                "evaluator_feedback": "",
                "userquery_type": "",
                "station_filter": None,
                "has_queue_info": False,
            })
            print(f"âœ“ Memory cleared for thread: {session_id}")
        except Exception as e:
            print(f"Note: Could not clear memory: {str(e)}")
    
    def get_conversation_history(self, thread_id: Optional[str] = None) -> List[BaseMessage]:
        """Get conversation history."""
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
    load_dotenv()
    grid_chat = GridChat()
    grid_chat.initialize()
    
    print("\n" + "="*60)
    print("TESTING SEQUENTIAL QUEUEâ†’DIAGNOSTICS FLOW")
    print("="*60)
    
    # Test queue query (should fetch queue + diagnostics)
    print("\n--- Test 1: Queue Query ---")
    response1 = grid_chat.process_message("What is the interconnection process for new resources?")
    print(f"\nResponse:\n{response1}\n")
    
    # Test diagnostics-only query
    print("\n--- Test 2: Diagnostics Query ---")
    response2 = grid_chat.process_message("Show me critical events from last 24 hours")
    print(f"\nResponse:\n{response2}\n")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)