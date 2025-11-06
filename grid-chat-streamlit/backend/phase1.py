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


# Import modular components
from diagevents import diagevents_query_prepare, diagevents_query_execute
from pgicq_kb import pgicq_kb_create_n_initialize, pgicq_kb_query, pgicq_resp_validate, pgicq_kb_query_rewrite

from prompts import PROMPT_SYNTHESIS, USERQUERY_CLASSIFIER_PROMPT
from constants import PGICQ_DOCS_FILE_PATHS, Evaluator, Userqueryclassifier, GridState

from gap_analysis_docgen import generate_gap_analysis_docx
from constants import GAP_ANALYSIS_REPORTS_DIR, GAP_ANALYSIS_UPLOAD_DIR

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

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("âœ“ Embeddings initialized")

        self.vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=PGICQ_KB_PERSIST_DIRECTORY,
            collection_name=PGICQ_KB_COLLECTION_NAME, 
        )

        # Load documents if needed
        has_existing_documents = len(self.vector_store.get(limit=1)['ids']) > 0
        if has_existing_documents:
            print("âœ“ PGICQ KB Vector DB found - reusing existing embeddings")
        else:
            print("  Loading and embedding documents...")
            docs = pgicq_kb_create_n_initialize()
            print(f"  Loaded {len(docs)} document chunks")
            self.vector_store.add_documents(docs)
            print("  Embeddings stored in ChromaDB")
        
        print("âœ“ PGICQ KB (Chroma Vector DB) initialized")


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
4. If no events found, focus on queue information only

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
    # GAP ANALYSIS NODE FUNCTIONS - START
    ####################################################################

    def extract_pdf_and_sections(self, state: GridState) -> Dict:
        """
        Node 1: Extract text from uploaded PDF and identify sections.
        
        Uses pdfplumber for better text extraction quality.
        Calls LLM to identify document sections and metadata.
        """
        print(f"\n{'='*60}")
        print("NODE: extract_pdf_and_sections")
        print(f"{'='*60}")
        
        from prompts import PROMPT_EXTRACT_SECTIONS
        import pdfplumber
        
        pdf_path = state.get("uploaded_pdf_path")
        if not pdf_path:
            print("ERROR: No PDF path provided")
            return {
                "extracted_text": "",
                "document_sections": {},
                "document_metadata": {},
                "progress_percent": 5,
                "current_step": "PDF extraction failed"
            }
        
        try:
            # Extract text from PDF
            print(f"Extracting text from: {pdf_path}")
            full_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        full_text.append(f"[Page {page_num}]\n{text}")
            
            extracted_text = "\n\n".join(full_text)
            print(f"Extracted {len(extracted_text)} characters from {len(full_text)} pages")
            
            # Truncate if too long (keep first 50k chars for analysis)
            if len(extracted_text) > 50000:
                print("Text too long - truncating to 50k characters")
                extracted_text_for_llm = extracted_text[:50000]
            else:
                extracted_text_for_llm = extracted_text
            
            # Call LLM to identify sections
            print("Identifying document sections with LLM...")
            section_prompt = PROMPT_EXTRACT_SECTIONS.format(
                document_text=extracted_text_for_llm
            )
            
            response = self.llm.invoke([
                SystemMessage(content="You are an expert at analyzing technical documents. Always respond with valid JSON."),
                HumanMessage(content=section_prompt)
            ])
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                sections = {s["section_name"]: s["content"] for s in result.get("sections", [])}
                metadata = result.get("metadata", {})
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Fallback: treat entire doc as one section
                sections = {"Full Document": extracted_text_for_llm[:10000]}
                metadata = {}
            
            print(f"Identified {len(sections)} sections")
            print(f"Metadata: {metadata}")
            
            return {
                "extracted_text": extracted_text,  # Store full text
                "document_sections": sections,
                "document_metadata": metadata,
                "progress_percent": 10,
                "current_step": "PDF extracted and sections identified"
            }
            
        except Exception as e:
            print(f"ERROR extracting PDF: {e}")
            import traceback
            traceback.print_exc()
            return {
                "extracted_text": "",
                "document_sections": {},
                "document_metadata": {},
                "progress_percent": 5,
                "current_step": f"PDF extraction error: {str(e)}"
            }


    def validate_sections_rag(self, state: GridState) -> Dict:
        """
        Node 2: Validate document sections against KB requirements.
        Uses pgicq_kb_query_generic for RAG queries.
        """
        print(f"\n{'='*60}")
        print("NODE: validate_sections_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_VALIDATE_SECTION
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        sections = state.get("document_sections", {})
        metadata = state.get("document_metadata", {})
        
        if not sections:
            print("No sections to validate")
            return {
                "correct_sections": [],
                "progress_percent": 20,
                "current_step": "No sections to validate"
            }
        
        correct_sections = []
        
        # Validate each section
        for idx, (section_name, section_content) in enumerate(sections.items(), 1):
            print(f"\nValidating section {idx}/{len(sections)}: {section_name}")
            
            # Query KB for requirements for this section
            query = f"What are the requirements for the '{section_name}' section in interconnection requests?"
            kb_results = pgicq_kb_query_generic(
                query=query,
                retriever=self.retriever,
                vector_store=self.vector_store,
                k=5
            )
            
            if not kb_results:
                print(f"No KB info found for {section_name}")
                continue
            
            # Format KB results
            kb_text = "\n\n".join([
                f"[{doc['filename']}, p. {doc['page']}]\n{doc['content'][:500]}"
                for doc in kb_results[:3]
            ])
            
            # Call LLM to validate
            validation_prompt = PROMPT_VALIDATE_SECTION.format(
                section_name=section_name,
                section_content=section_content[:1000],  # Truncate content
                kb_requirements=kb_text
            )
            
            try:
                response = self.llm.invoke([
                    SystemMessage(content="You are a compliance expert. Respond with valid JSON only."),
                    HumanMessage(content=validation_prompt)
                ])
                
                result = json.loads(response.content)
                
                if result.get("is_correct"):
                    correct_sections.append({
                        "section_name": section_name,
                        "details": result.get("correctness_details", ""),
                        "references": result.get("references", [])
                    })
                    print(f" {section_name} is correct")
                else:
                    print(f"âœ— {section_name} needs review")
                    
            except Exception as e:
                print(f"Error validating {section_name}: {e}")
        
        print(f"\nFound {len(correct_sections)} correct sections")
        
        return {
            "correct_sections": correct_sections,
            "progress_percent": 25,
            "current_step": f"Validated {len(sections)} sections"
        }


    def identify_modifications_rag(self, state: GridState) -> Dict:
        """
        Node 3: Identify sections that need modifications.
        """
        print(f"\n{'='*60}")
        print("NODE: identify_modifications_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_FIND_MODIFICATIONS
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        sections = state.get("document_sections", {})
        correct_sections = state.get("correct_sections", [])
        
        # Get sections that were NOT marked as correct
        correct_section_names = {s["section_name"] for s in correct_sections}
        sections_to_check = {
            name: content for name, content in sections.items()
            if name not in correct_section_names
        }
        
        if not sections_to_check:
            print("All sections are correct - no modifications needed")
            return {
                "sections_to_modify": [],
                "progress_percent": 35,
                "current_step": "No modifications needed"
            }
        
        sections_to_modify = []
        
        for section_name, section_content in sections_to_check.items():
            print(f"\nChecking modifications for: {section_name}")
            
            # Query KB
            query = f"What are the detailed requirements and compliance standards for '{section_name}' in interconnection requests?"
            kb_results = pgicq_kb_query_generic(
                query=query,
                retriever=self.retriever,
                vector_store=self.vector_store,
                k=5
            )
            
            if not kb_results:
                continue
            
            kb_text = "\n\n".join([
                f"[{doc['filename']}, p. {doc['page']}]\n{doc['content'][:500]}"
                for doc in kb_results[:3]
            ])
            
            # Call LLM
            mod_prompt = PROMPT_FIND_MODIFICATIONS.format(
                section_name=section_name,
                section_content=section_content[:1000],
                kb_requirements=kb_text
            )
            
            try:
                response = self.llm.invoke([
                    SystemMessage(content="You are a compliance expert. Respond with valid JSON only."),
                    HumanMessage(content=mod_prompt)
                ])
                
                result = json.loads(response.content)
                
                if result.get("needs_modification"):
                    sections_to_modify.append({
                        "section_name": section_name,
                        "issues": result.get("issues_found", []),
                        "recommendations": result.get("recommendations", []),
                        "references": result.get("references", [])
                    })
                    print(f"⚠️ {section_name} needs modifications")
                    
            except Exception as e:
                print(f"Error analyzing {section_name}: {e}")
        
        print(f"\nFound {len(sections_to_modify)} sections needing modifications")
        
        return {
            "sections_to_modify": sections_to_modify,
            "progress_percent": 40,
            "current_step": f"Identified {len(sections_to_modify)} sections to modify"
        }


    def find_missing_sections_rag(self, state: GridState) -> Dict:
        """
        Node 4: Find required sections that are missing from document.
        """
        print(f"\n{'='*60}")
        print("NODE: find_missing_sections_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_FIND_MISSING_SECTIONS
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        sections = state.get("document_sections", {})
        metadata = state.get("document_metadata", {})
        
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        country = metadata.get("country", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        print(f"Checking for missing sections for {request_type} request")
        
        # Query KB for all required sections
        query = f"What are all mandatory sections required for {request_type} interconnection requests in {region} {country}?"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No KB info about required sections")
            return {
                "missing_sections": [],
                "progress_percent": 50,
                "current_step": "Could not determine required sections"
            }
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content'][:800]}"
            for doc in kb_results[:5]
        ])
        
        # Get list of present sections
        present_sections = list(sections.keys())
        
        # Call LLM
        missing_prompt = PROMPT_FIND_MISSING_SECTIONS.format(
            present_sections=", ".join(present_sections),
            station_name=station,
            region=region,
            country=country,
            request_type=request_type,
            kb_requirements=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a compliance expert. Respond with valid JSON only."),
                HumanMessage(content=missing_prompt)
            ])
            
            result = json.loads(response.content)
            missing_sections = result.get("missing_sections", [])
            
            print(f"Found {len(missing_sections)} missing sections")
            
            return {
                "missing_sections": missing_sections,
                "progress_percent": 55,
                "current_step": f"Identified {len(missing_sections)} missing sections"
            }
            
        except Exception as e:
            print(f"Error finding missing sections: {e}")
            return {
                "missing_sections": [],
                "progress_percent": 55,
                "current_step": "Error finding missing sections"
            }


    def fetch_compliance_reqs_rag(self, state: GridState) -> Dict:
        """
        Node 5: Fetch mandatory compliance requirements from KB.
        """
        print(f"\n{'='*60}")
        print("NODE: fetch_compliance_reqs_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_COMPLIANCE_REQUIREMENTS
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        country = metadata.get("country", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        # Query KB
        query = f"What are the mandatory compliance requirements for {request_type} at {station} in {region}?"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No compliance requirements found in KB")
            return {
                "compliance_requirements": [],
                "progress_percent": 60,
                "current_step": "Not able to find compliance requirements"
            }
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content']}"
            for doc in kb_results[:8]
        ])
        
        # Call LLM
        compliance_prompt = PROMPT_COMPLIANCE_REQUIREMENTS.format(
            station_name=station,
            region=region,
            country=country,
            request_type=request_type,
            kb_info=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a compliance expert. Respond with valid JSON only."),
                HumanMessage(content=compliance_prompt)
            ])
            
            result = json.loads(response.content)
            
            if result.get("status") == "Not able to find":
                print("Not able to find compliance requirements")
                compliance_reqs = []
            else:
                compliance_reqs = result.get("compliance_requirements", [])
            
            print(f"Found {len(compliance_reqs)} compliance requirements")
            
            return {
                "compliance_requirements": compliance_reqs,
                "progress_percent": 65,
                "current_step": f"Found {len(compliance_reqs)} compliance requirements"
            }
            
        except Exception as e:
            print(f"Error fetching compliance: {e}")
            return {
                "compliance_requirements": [],
                "progress_percent": 65,
                "current_step": "Error fetching compliance requirements"
            }


    def fetch_technical_tests_rag(self, state: GridState) -> Dict:
        """
        Node 6: Fetch technical test requirements from KB.
        """
        print(f"\n{'='*60}")
        print("NODE: fetch_technical_tests_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_TECHNICAL_TESTS
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        country = metadata.get("country", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        # Query KB
        query = f"What technical tests and validation procedures are required for {request_type} at {station}?"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No technical tests found in KB")
            return {
                "technical_tests": [],
                "progress_percent": 70,
                "current_step": "Not able to find technical tests"
            }
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content']}"
            for doc in kb_results[:8]
        ])
        
        # Call LLM
        tests_prompt = PROMPT_TECHNICAL_TESTS.format(
            station_name=station,
            region=region,
            country=country,
            request_type=request_type,
            kb_info=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a technical expert. Respond with valid JSON only."),
                HumanMessage(content=tests_prompt)
            ])
            
            result = json.loads(response.content)
            
            if result.get("status") == "Not able to find":
                print("Not able to find technical tests")
                tests = []
            else:
                tests = result.get("technical_tests", [])
            
            print(f"Found {len(tests)} technical tests")
            
            return {
                "technical_tests": tests,
                "progress_percent": 75,
                "current_step": f"Found {len(tests)} technical tests"
            }
            
        except Exception as e:
            print(f"Error fetching tests: {e}")
            return {
                "technical_tests": [],
                "progress_percent": 75,
                "current_step": "Error fetching technical tests"
            }

    def analyze_queue_rag(self, state: GridState) -> Dict:
        """
        Node 7: Analyze queue wait time and open requests.
        Prioritize LATEST document versions.
        """
        print(f"\n{'='*60}")
        print("NODE: analyze_queue_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_QUEUE_WAIT_TIME
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        # Query KB with emphasis on latest data
        query = f"Current queue wait time and open interconnection requests at {station} in {region} latest 2024 2025"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No queue information found in KB")
            return {
                "queue_wait_analysis": {
                    "status": "Not able to find",
                    "reason": "No queue data available in knowledge base"
                },
                "progress_percent": 80,
                "current_step": "Not able to find queue information"
            }
        
        # Prioritize documents with recent dates in filename
        kb_results_sorted = sorted(
            kb_results,
            key=lambda x: ("2025" in x['filename'] or "2024" in x['filename']),
            reverse=True
        )
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content']}"
            for doc in kb_results_sorted[:5]
        ])
        
        # Call LLM
        queue_prompt = PROMPT_QUEUE_WAIT_TIME.format(
            station_name=station,
            region=region,
            request_type=request_type,
            kb_info=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a queue analyst. Respond with valid JSON only. Prioritize latest information."),
                HumanMessage(content=queue_prompt)
            ])
            
            result = json.loads(response.content)
            
            if result.get("status") == "Not able to find":
                print("Not able to find queue wait time")
                queue_analysis = {
                    "status": "Not able to find",
                    "reason": result.get("reason", "Information not available")
                }
            else:
                queue_analysis = result.get("queue_info", {})
                queue_analysis["status"] = "found"
            
            print(f"Queue analysis: {queue_analysis.get('status')}")
            
            return {
                "queue_wait_analysis": queue_analysis,
                "progress_percent": 82,
                "current_step": "Queue analysis completed"
            }
            
        except Exception as e:
            print(f"Error analyzing queue: {e}")
            return {
                "queue_wait_analysis": {
                    "status": "Not able to find",
                    "reason": f"Error: {str(e)}"
                },
                "progress_percent": 82,
                "current_step": "Error analyzing queue"
            }


    def assess_risks_rag(self, state: GridState) -> Dict:
        """
        Node 8: Assess risks from KB.
        """
        print(f"\n{'='*60}")
        print("NODE: assess_risks_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_RISK_ASSESSMENT
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        country = metadata.get("country", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        # Query KB for risks
        query = f"What are the risks and challenges for {request_type} interconnection at {station} in {region}?"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No risk information found in KB")
            return {
                "risk_assessment": [],
                "progress_percent": 85,
                "current_step": "No risk information found"
            }
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content']}"
            for doc in kb_results[:8]
        ])
        
        # Call LLM
        risk_prompt = PROMPT_RISK_ASSESSMENT.format(
            station_name=station,
            region=region,
            country=country,
            request_type=request_type,
            kb_info=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a risk assessment expert. Respond with valid JSON only."),
                HumanMessage(content=risk_prompt)
            ])
            
            result = json.loads(response.content)
            risks = result.get("risks", [])
            
            print(f"Found {len(risks)} risks")
            
            return {
                "risk_assessment": risks,
                "progress_percent": 87,
                "current_step": f"Identified {len(risks)} risks"
            }
            
        except Exception as e:
            print(f"Error assessing risks: {e}")
            return {
                "risk_assessment": [],
                "progress_percent": 87,
                "current_step": "Error assessing risks"
            }


    def identify_environmental_rag(self, state: GridState) -> Dict:
        """
        Node 9: Identify environmental challenges and requirements.
        """
        print(f"\n{'='*60}")
        print("NODE: identify_environmental_rag")
        print(f"{'='*60}")
        
        from prompts import PROMPT_ENVIRONMENTAL_CHALLENGES
        from pgicq_kb import pgicq_kb_query_generic
        import json
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "unspecified")
        region = metadata.get("region", "unspecified")
        country = metadata.get("country", "unspecified")
        request_type = metadata.get("request_type", "interconnection")
        
        # Query KB
        query = f"What are the environmental requirements and challenges for {request_type} at {station} in {region}?"
        kb_results = pgicq_kb_query_generic(
            query=query,
            retriever=self.retriever,
            vector_store=self.vector_store,
            k=10
        )
        
        if not kb_results:
            print("No environmental information found in KB")
            return {
                "environmental_challenges": [],
                "progress_percent": 90,
                "current_step": "No environmental information found"
            }
        
        kb_text = "\n\n".join([
            f"[{doc['filename']}, p. {doc['page']}]\n{doc['content']}"
            for doc in kb_results[:8]
        ])
        
        # Call LLM
        env_prompt = PROMPT_ENVIRONMENTAL_CHALLENGES.format(
            station_name=station,
            region=region,
            country=country,
            request_type=request_type,
            kb_info=kb_text
        )
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an environmental compliance expert. Respond with valid JSON only."),
                HumanMessage(content=env_prompt)
            ])
            
            result = json.loads(response.content)
            challenges = result.get("environmental_challenges", [])
            
            print(f"Found {len(challenges)} environmental challenges")
            
            return {
                "environmental_challenges": challenges,
                "progress_percent": 92,
                "current_step": f"Identified {len(challenges)} environmental challenges"
            }
            
        except Exception as e:
            print(f"Error identifying environmental: {e}")
            return {
                "environmental_challenges": [],
                "progress_percent": 92,
                "current_step": "Error identifying environmental challenges"
            }


    def fetch_diagnostics_for_gap(self, state: GridState) -> Dict:
        """
        Node 10: Fetch diagnostic events from database.
        NO ANALYSIS - just retrieve and list events.
        """
        print(f"\n{'='*60}")
        print("NODE: fetch_diagnostics_for_gap")
        print(f"{'='*60}")
        
        from diagevents_dbif import gd_userquery_execute
        
        metadata = state.get("document_metadata", {})
        station = metadata.get("station_name", "")
        
        # FUTURE: Filter by station when station column available in DB
        # For now: Get all recent events
        
        if station and station != "unspecified":
            print(f"FUTURE: Will filter by station: {station}")
            print("CURRENT: Fetching all recent events (station filter not yet implemented)")
        
        sql_query = """
    SELECT e.event_id, e.event_name, e.event_type, 
        s.severity_name, l.timestamp, l.status
    FROM eventslog l
    JOIN event e ON l.event_id = e.event_id
    LEFT JOIN severity s ON l.severity_id = s.severity_id
    WHERE l.timestamp >= datetime('now', '-6 months')
    ORDER BY l.timestamp DESC
    LIMIT 50;
    """
        
        try:
            diagnostics_results = gd_userquery_execute(sql_query)
            
            # Format for report (no analysis)
            diagnostics_list = []
            for row in diagnostics_results:
                diagnostics_list.append({
                    "event_id": row[0],
                    "event_name": row[1],
                    "event_type": row[2],
                    "severity": row[3],
                    "timestamp": row[4],
                    "status": row[5]
                })
            
            print(f"Retrieved {len(diagnostics_list)} diagnostic events")
            
            return {
                "station_diagnostics": diagnostics_list,
                "progress_percent": 95,
                "current_step": f"Retrieved {len(diagnostics_list)} diagnostic events"
            }
            
        except Exception as e:
            print(f"Error fetching diagnostics: {e}")
            return {
                "station_diagnostics": [],
                "progress_percent": 95,
                "current_step": "Error fetching diagnostics"
            }


    def synthesize_gap_report(self, state: GridState) -> Dict:
        """
        Node 11: Synthesize all findings into comprehensive report content.
        """
        print(f"\n{'='*60}")
        print("NODE: synthesize_gap_report")
        print(f"{'='*60}")
        
        from prompts import PROMPT_GAP_REPORT_SYNTHESIS
        import json
        
        # Gather all findings
        metadata = state.get("document_metadata", {})
        sections = state.get("document_sections", {})
        correct_sections = state.get("correct_sections", [])
        sections_to_modify = state.get("sections_to_modify", [])
        missing_sections = state.get("missing_sections", [])
        compliance_reqs = state.get("compliance_requirements", [])
        technical_tests = state.get("technical_tests", [])
        queue_analysis = state.get("queue_wait_analysis", {})
        risks = state.get("risk_assessment", [])
        env_challenges = state.get("environmental_challenges", [])
        diagnostics = state.get("station_diagnostics", [])
        
        # Format for prompt
        def format_list(items):
            if not items:
                return "None found"
            return json.dumps(items, indent=2)
        
        # Create comprehensive prompt
        report_prompt = PROMPT_GAP_REPORT_SYNTHESIS.format(
            document_sections=", ".join(sections.keys()) if sections else "None",
            station_name=metadata.get("station_name", "Unspecified"),
            region=metadata.get("region", "Unspecified"),
            request_type=metadata.get("request_type", "Unspecified"),
            correct_sections=format_list(correct_sections),
            sections_to_modify=format_list(sections_to_modify),
            missing_sections=format_list(missing_sections),
            compliance_requirements=format_list(compliance_reqs),
            technical_tests=format_list(technical_tests),
            queue_wait_analysis=json.dumps(queue_analysis, indent=2),
            risk_assessment=format_list(risks),
            environmental_challenges=format_list(env_challenges),
            station_diagnostics=f"{len(diagnostics)} events retrieved (details in report)"
        )
        
        try:
            print("Generating comprehensive gap analysis report...")
            
            response = self.llm.invoke([
                SystemMessage(content="You are an expert technical writer specializing in interconnection compliance reports."),
                HumanMessage(content=report_prompt)
            ])
            
            report_content = response.content
            
            print(f"Generated report: {len(report_content)} characters")
            
            return {
                "gap_report_content": report_content,
                "progress_percent": 98,
                "current_step": "Report synthesized"
            }
            
        except Exception as e:
            print(f"Error synthesizing report: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback simple report
            fallback_report = f"""# GAP ANALYSIS REPORT
            
    ## ERROR
    Report synthesis failed: {str(e)}

    ## Raw Findings:
    - Correct sections: {len(correct_sections)}
    - Sections to modify: {len(sections_to_modify)}
    - Missing sections: {len(missing_sections)}
    - Compliance requirements: {len(compliance_reqs)}
    - Technical tests: {len(technical_tests)}
    - Risks: {len(risks)}
    - Environmental: {len(env_challenges)}
    - Diagnostic events: {len(diagnostics)}
    """
            
            return {
                "gap_report_content": fallback_report,
                "progress_percent": 98,
                "current_step": "Report synthesis error - fallback generated"
            }


    def generate_gap_report_file(self, state: GridState) -> Dict:
        """
        Node 12: Generate DOCX file from report content.
        """
        print(f"\n{'='*60}")
        print("NODE: generate_gap_report_file")
        print(f"{'='*60}")
        
        from gap_analysis_docgen import generate_gap_analysis_docx
        from constants import GAP_ANALYSIS_REPORTS_DIR
        
        report_content = state.get("gap_report_content", "")
        metadata = state.get("document_metadata", {})
        
        if not report_content:
            print("ERROR: No report content to generate file")
            return {
                "gap_report_path": None,
                "progress_percent": 100,
                "current_step": "Error: No report content"
            }
        
        try:
            # Generate DOCX
            docx_path = generate_gap_analysis_docx(
                gap_report_content=report_content,
                document_metadata=metadata,
                output_dir=GAP_ANALYSIS_REPORTS_DIR
            )
            
            print(f"Report generated: {docx_path}")
            
            return {
                "gap_report_path": docx_path,
                "progress_percent": 100,
                "current_step": "Report generated successfully"
            }
            
        except Exception as e:
            print(f"Error generating report file: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "gap_report_path": None,
                "progress_percent": 100,
                "current_step": f"Error generating file: {str(e)}"
            }

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
        
        # ROUTING: Update from START
        workflow_builder.add_edge(START, "userquery_classifier")
        
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