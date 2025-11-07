"""
##########################################################
#
# Capstone Team 16: 
#    Power Grids Interconnection Queue Analyzer
#
#  Module:
#     Power Grids Interconnection Queue Gap Analysis
#     Contains all gap analysis node functions and logic.
#     
#  Author: RK (kvrkr866@gmail.com)
#
#  FIXED: All prompt parameter names corrected to match prompts.py
#
##########################################################
"""

import json
import os.path as osp
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from constants import GridState
from pgicq_kb import pgicq_kb_query_generic
from gap_analysis_docgen import generate_gap_analysis_docx


####################################################################
def pgicq_gap_extract_pdf_and_sections(state: GridState, llm) -> Dict:
    """
    Extract text from uploaded PDF and identify sections.
    
    Uses pdfplumber for better text extraction quality.
    Calls LLM to identify document sections and metadata.
    
    Args:
        state: Current graph state with uploaded_pdf_path
        llm: Language model for section identification
        
    Returns:
        Dict with extracted_text, document_sections, document_metadata, progress
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_extract_pdf_and_sections")
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
        
        response = llm.invoke([
            SystemMessage(content="You are an expert at analyzing technical documents. Always respond with valid JSON."),
            HumanMessage(content=section_prompt)
        ])
        
        # Parse JSON response
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
        print(f"ERROR extracting PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "extracted_text": "",
            "document_sections": {},
            "document_metadata": {},
            "progress_percent": 5,
            "current_step": f"PDF extraction error: {str(e)}"
        }


####################################################################
def pgicq_gap_validate_sections(state: GridState, llm, retriever, vector_store) -> Dict:
    """
    Validate document sections against CAISO requirements using RAG.
    
    Args:
        state: Current graph state with document_sections
        llm: Language model for validation
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with correct_sections list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_validate_sections")
    print(f"{'='*60}")
    
    from prompts import PROMPT_VALIDATE_SECTIONS
    
    sections = state.get("document_sections", {})
    if not sections:
        print("WARNING: No sections to validate")
        return {
            "correct_sections": [],
            "progress_percent": 20,
            "current_step": "Section validation skipped - no sections found"
        }
    
    correct_sections = []
    
    try:
        # For each section, check if it meets requirements
        for section_name, section_content in sections.items():
            print(f"\nValidating section: {section_name}")
            
            # Query RAG for requirements related to this section
            rag_query = f"What are the CAISO requirements for {section_name} in interconnection requests?"
            rag_docs = pgicq_kb_query_generic(
                query=rag_query,
                retriever=retriever,
                vector_store=vector_store,
                k=5
            )
            
            if not rag_docs:
                print(f"  No requirements found for {section_name}")
                continue
            
            # Format RAG results - FIXED: using kb_requirements parameter
            kb_requirements = "\n\n".join([
                f"[{doc['filename']}, p. {doc['page']}]\n{doc['content'][:300]}"
                for doc in rag_docs[:3]
            ])
            
            # Ask LLM to validate - FIXED: correct parameter names
            validation_prompt = PROMPT_VALIDATE_SECTIONS.format(
                section_name=section_name,
                section_content=section_content[:1000],  # Truncate long sections
                kb_requirements=kb_requirements  # FIXED: was 'requirements'
            )
            
            response = llm.invoke([
                SystemMessage(content="You are an interconnection compliance expert. Respond with JSON."),
                HumanMessage(content=validation_prompt)
            ])
            
            # Parse response
            try:
                result = json.loads(response.content)
                if result.get("is_correct", False):
                    correct_sections.append({
                        "section_name": section_name,
                        "status": "Correct",
                        "details": result.get("correctness_details", ""),
                        "references": result.get("references", [])
                    })
                    print(f"  ✓ {section_name} is correct")
            except json.JSONDecodeError:
                print(f"  ERROR: Could not parse validation response for {section_name}")
        
        print(f"\nFound {len(correct_sections)} correct sections")
        
        return {
            "correct_sections": correct_sections,
            "progress_percent": 25,
            "current_step": f"Validated {len(sections)} sections - {len(correct_sections)} correct"
        }
        
    except Exception as e:
        print(f"ERROR in section validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "correct_sections": [],
            "progress_percent": 20,
            "current_step": f"Section validation error: {str(e)}"
        }


####################################################################
def pgicq_gap_identify_modifications(state: GridState, llm, retriever, vector_store) -> Dict:
    """
    Identify sections that need modifications using RAG.
    
    Args:
        state: Current graph state with document_sections
        llm: Language model
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with sections_to_modify list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_identify_modifications")
    print(f"{'='*60}")
    
    from prompts import PROMPT_FIND_MODIFICATIONS  # FIXED: correct prompt name
    
    sections = state.get("document_sections", {})
    correct_sections = state.get("correct_sections", [])
    correct_section_names = {s["section_name"] for s in correct_sections}
    
    sections_to_modify = []
    
    try:
        # Check sections that aren't already marked as correct
        for section_name, section_content in sections.items():
            if section_name in correct_section_names:
                continue  # Skip already validated sections
            
            print(f"\nAnalyzing modifications needed for: {section_name}")
            
            # Query RAG for what should be in this section
            rag_query = f"What must be included in {section_name} for CAISO interconnection requests?"
            rag_docs = pgicq_kb_query_generic(
                query=rag_query,
                retriever=retriever,
                vector_store=vector_store,
                k=5
            )
            
            if not rag_docs:
                continue
            
            # Format requirements - FIXED: using kb_requirements parameter
            kb_requirements = "\n\n".join([
                f"[{doc['filename']}]\n{doc['content'][:300]}"
                for doc in rag_docs[:3]
            ])
            
            # Ask LLM to identify gaps - FIXED: correct parameter names
            modification_prompt = PROMPT_FIND_MODIFICATIONS.format(
                section_name=section_name,
                section_content=section_content[:1000],
                kb_requirements=kb_requirements  # FIXED: was 'requirements'
            )
            
            response = llm.invoke([
                SystemMessage(content="You are an interconnection technical reviewer."),
                HumanMessage(content=modification_prompt)
            ])
            
            # Parse response
            try:
                result = json.loads(response.content)
                if result.get("needs_modification", False):
                    sections_to_modify.append({
                        "section_name": section_name,
                        "issues": result.get("issues_found", []),
                        "recommendations": result.get("recommendations", []),
                        "references": result.get("references", [])
                    })
                    print(f"  ⚠ {section_name} needs modifications")
            except json.JSONDecodeError:
                print(f"  ERROR: Could not parse modification analysis for {section_name}")
        
        print(f"\nFound {len(sections_to_modify)} sections needing modifications")
        
        return {
            "sections_to_modify": sections_to_modify,
            "progress_percent": 35,
            "current_step": f"Identified {len(sections_to_modify)} sections needing changes"
        }
        
    except Exception as e:
        print(f"ERROR identifying modifications: {str(e)}")
        return {
            "sections_to_modify": [],
            "progress_percent": 30,
            "current_step": f"Modification analysis error: {str(e)}"
        }


####################################################################
def pgicq_gap_find_missing_sections(state: GridState, llm, retriever, vector_store) -> Dict:
    """
    Identify required sections that are missing from the document using RAG.
    
    Args:
        state: Current graph state
        llm: Language model
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with missing_sections list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_find_missing_sections")
    print(f"{'='*60}")
    
    from prompts import PROMPT_FIND_MISSING_SECTIONS
    
    sections = state.get("document_sections", {})
    present_sections = list(sections.keys())
    document_metadata = state.get("document_metadata", {})
    
    try:
        # Query RAG for list of required sections
        rag_query = "What are all the mandatory sections required in a CAISO interconnection request document?"
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=8
        )
        
        if not rag_docs:
            print("No requirement docs found")
            return {
                "missing_sections": [],
                "progress_percent": 45,
                "current_step": "Could not determine missing sections"
            }
        
        # Format requirements - FIXED: using kb_requirements parameter
        kb_requirements = "\n\n".join([
            f"[{doc['filename']}]\n{doc['content']}"
            for doc in rag_docs
        ])
        
        # Ask LLM to identify missing sections - FIXED: ALL required parameters
        missing_prompt = PROMPT_FIND_MISSING_SECTIONS.format(
            present_sections=", ".join(present_sections),
            station_name=document_metadata.get("station_name", "Not specified"),
            region=document_metadata.get("region", "Not specified"),
            country=document_metadata.get("country", "Not specified"),
            request_type=document_metadata.get("request_type", "Not specified"),
            kb_requirements=kb_requirements  # FIXED: was 'requirements'
        )
        
        response = llm.invoke([
            SystemMessage(content="You are an interconnection documentation expert. Respond with JSON."),
            HumanMessage(content=missing_prompt)
        ])
        
        # Parse response
        try:
            result = json.loads(response.content)
            missing_sections = result.get("missing_sections", [])
            
            # Enhance with references
            for section in missing_sections:
                if "references" not in section:
                    section["references"] = [{"document": doc['filename'], "page": doc['page']} for doc in rag_docs[:2]]
            
            print(f"Found {len(missing_sections)} missing mandatory sections")
            
            return {
                "missing_sections": missing_sections,
                "progress_percent": 45,
                "current_step": f"Found {len(missing_sections)} missing sections"
            }
            
        except json.JSONDecodeError:
            print("ERROR: Could not parse missing sections response")
            return {
                "missing_sections": [],
                "progress_percent": 40,
                "current_step": "Missing section analysis incomplete"
            }
            
    except Exception as e:
        print(f"ERROR finding missing sections: {str(e)}")
        return {
            "missing_sections": [],
            "progress_percent": 40,
            "current_step": f"Missing section error: {str(e)}"
        }


####################################################################
def pgicq_gap_fetch_compliance_requirements(state: GridState, retriever, vector_store) -> Dict:
    """
    Fetch mandatory compliance requirements using RAG.
    
    Args:
        state: Current graph state
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with compliance_requirements list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_fetch_compliance_requirements")
    print(f"{'='*60}")
    
    try:
        # Query for compliance requirements
        rag_query = "What are the mandatory compliance requirements for CAISO interconnection requests? Include certifications, standards, and regulatory requirements."
        
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=10
        )
        
        if not rag_docs:
            print("No compliance requirement docs found")
            return {
                "compliance_requirements": [],
                "progress_percent": 50,
                "current_step": "Compliance requirements unavailable"
            }
        
        # Extract requirements from docs
        compliance_requirements = []
        for doc in rag_docs:
            compliance_requirements.append({
                "requirement": doc["content"][:300],  # First 300 chars
                "source": doc["filename"],
                "page": doc["page"]
            })
        
        print(f"Found {len(compliance_requirements)} compliance requirements")
        
        return {
            "compliance_requirements": compliance_requirements,
            "progress_percent": 55,
            "current_step": f"Found {len(compliance_requirements)} compliance requirements"
        }
        
    except Exception as e:
        print(f"ERROR fetching compliance requirements: {str(e)}")
        return {
            "compliance_requirements": [],
            "progress_percent": 50,
            "current_step": f"Compliance fetch error: {str(e)}"
        }


####################################################################
def pgicq_gap_fetch_technical_tests(state: GridState, retriever, vector_store) -> Dict:
    """
    Fetch required technical tests using RAG.
    
    Args:
        state: Current graph state
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with technical_tests list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_fetch_technical_tests")
    print(f"{'='*60}")
    
    try:
        # Query for technical testing requirements
        rag_query = "What technical tests and studies are required for CAISO interconnection requests? Include electrical tests, protection studies, and validation requirements."
        
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=10
        )
        
        if not rag_docs:
            print("No technical test docs found")
            return {
                "technical_tests": [],
                "progress_percent": 60,
                "current_step": "Technical test info unavailable"
            }
        
        # Extract tests from docs
        technical_tests = []
        for doc in rag_docs:
            technical_tests.append({
                "test_name": "Technical Test",  # Could extract with LLM if needed
                "description": doc["content"][:300],
                "source": doc["filename"],
                "page": doc["page"]
            })
        
        print(f"Found {len(technical_tests)} technical tests")
        
        return {
            "technical_tests": technical_tests,
            "progress_percent": 65,
            "current_step": f"Found {len(technical_tests)} required tests"
        }
        
    except Exception as e:
        print(f"ERROR fetching technical tests: {str(e)}")
        return {
            "technical_tests": [],
            "progress_percent": 60,
            "current_step": f"Technical test fetch error: {str(e)}"
        }


####################################################################
def pgicq_gap_analyze_queue_wait(state: GridState, retriever, vector_store) -> Dict:
    """
    Analyze queue wait times using RAG.
    
    Args:
        state: Current graph state
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with queue_wait_analysis
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_analyze_queue_wait")
    print(f"{'='*60}")
    
    try:
        # Query for queue information
        rag_query = "What are typical queue wait times and interconnection queue statistics for CAISO? Include timeline information and processing duration."
        
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=5
        )
        
        if not rag_docs:
            print("No queue wait info found")
            return {
                "queue_wait_analysis": {
                    "summary": "Queue wait information not available",
                    "references": []
                },
                "progress_percent": 70,
                "current_step": "Queue analysis unavailable"
            }
        
        # Summarize queue info
        queue_info = "\n".join([doc["content"][:200] for doc in rag_docs])
        references = [f"{doc['filename']}, p. {doc['page']}" for doc in rag_docs]
        
        queue_wait_analysis = {
            "summary": queue_info[:500],  # Truncate summary
            "references": references
        }
        
        print("Queue wait analysis completed")
        
        return {
            "queue_wait_analysis": queue_wait_analysis,
            "progress_percent": 75,
            "current_step": "Queue wait analysis completed"
        }
        
    except Exception as e:
        print(f"ERROR analyzing queue wait: {str(e)}")
        return {
            "queue_wait_analysis": {},
            "progress_percent": 70,
            "current_step": f"Queue analysis error: {str(e)}"
        }


####################################################################
def pgicq_gap_assess_risks(state: GridState, retriever, vector_store) -> Dict:
    """
    Assess interconnection risks using RAG.
    
    Args:
        state: Current graph state
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with risk_assessment list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_assess_risks")
    print(f"{'='*60}")
    
    try:
        # Query for risk information
        rag_query = "What are common risks and challenges in CAISO interconnection requests? Include technical risks, regulatory risks, and timeline risks."
        
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=8
        )
        
        if not rag_docs:
            print("No risk info found")
            return {
                "risk_assessment": [],
                "progress_percent": 80,
                "current_step": "Risk assessment unavailable"
            }
        
        # Extract risks
        risk_assessment = []
        for doc in rag_docs:
            risk_assessment.append({
                "risk": doc["content"][:250],
                "source": doc["filename"],
                "page": doc["page"]
            })
        
        print(f"Identified {len(risk_assessment)} risk factors")
        
        return {
            "risk_assessment": risk_assessment,
            "progress_percent": 85,
            "current_step": f"Identified {len(risk_assessment)} risk factors"
        }
        
    except Exception as e:
        print(f"ERROR assessing risks: {str(e)}")
        return {
            "risk_assessment": [],
            "progress_percent": 80,
            "current_step": f"Risk assessment error: {str(e)}"
        }


####################################################################
def pgicq_gap_identify_environmental(state: GridState, retriever, vector_store) -> Dict:
    """
    Identify environmental challenges using RAG.
    
    Args:
        state: Current graph state
        retriever: Vector store retriever
        vector_store: ChromaDB vector store
        
    Returns:
        Dict with environmental_challenges list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_identify_environmental")
    print(f"{'='*60}")
    
    try:
        # Query for environmental requirements
        rag_query = "What are the environmental requirements and challenges for CAISO interconnection requests? Include permitting, impact studies, and compliance."
        
        rag_docs = pgicq_kb_query_generic(
            query=rag_query,
            retriever=retriever,
            vector_store=vector_store,
            k=6
        )
        
        if not rag_docs:
            print("No environmental info found")
            return {
                "environmental_challenges": [],
                "progress_percent": 88,
                "current_step": "Environmental analysis unavailable"
            }
        
        # Extract challenges
        environmental_challenges = []
        for doc in rag_docs:
            environmental_challenges.append({
                "challenge": doc["content"][:250],
                "source": doc["filename"],
                "page": doc["page"]
            })
        
        print(f"Identified {len(environmental_challenges)} environmental factors")
        
        return {
            "environmental_challenges": environmental_challenges,
            "progress_percent": 90,
            "current_step": f"Identified {len(environmental_challenges)} environmental factors"
        }
        
    except Exception as e:
        print(f"ERROR identifying environmental challenges: {str(e)}")
        return {
            "environmental_challenges": [],
            "progress_percent": 88,
            "current_step": f"Environmental analysis error: {str(e)}"
        }


####################################################################
def pgicq_gap_fetch_diagnostics(state: GridState) -> Dict:
    """
    Fetch recent diagnostic events from database (no analysis, just data).
    
    Args:
        state: Current graph state
        
    Returns:
        Dict with station_diagnostics list
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_fetch_diagnostics")
    print(f"{'='*60}")
    
    from diagevents_dbif import gd_userquery_execute
    
    try:
        # Get recent events (last 30 days)
        sql_query = """
SELECT e.event_name, e.event_type, s.severity_name, l.timestamp, l.status
FROM eventslog l
JOIN event e ON l.event_id = e.event_id
LEFT JOIN severity s ON l.severity_id = s.severity_id
WHERE l.timestamp >= datetime('now', '-30 days')
ORDER BY l.timestamp DESC
LIMIT 50;
"""
        
        results = gd_userquery_execute(sql_query)
        
        # Format results
        station_diagnostics = []
        for row in results:
            station_diagnostics.append({
                "event_name": row[0],
                "event_type": row[1],
                "severity": row[2],
                "timestamp": row[3],
                "status": row[4]
            })
        
        print(f"Retrieved {len(station_diagnostics)} diagnostic events")
        
        return {
            "station_diagnostics": station_diagnostics,
            "progress_percent": 92,
            "current_step": f"Retrieved {len(station_diagnostics)} diagnostic events"
        }
        
    except Exception as e:
        print(f"ERROR fetching diagnostics: {str(e)}")
        return {
            "station_diagnostics": [],
            "progress_percent": 90,
            "current_step": f"Diagnostics fetch error: {str(e)}"
        }


####################################################################
def pgicq_gap_synthesize_report(state: GridState, llm) -> Dict:
    """
    Synthesize all gap analysis findings into a comprehensive report.
    
    FIXED: Uses correct parameter names from PROMPT_GAP_REPORT_SYNTHESIS
    
    Args:
        state: Current graph state with all analysis results
        llm: Language model for synthesis
        
    Returns:
        Dict with gap_report_content (markdown text)
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_synthesize_report")
    print(f"{'='*60}")
    
    from prompts import PROMPT_GAP_REPORT_SYNTHESIS 
    
    # Gather all analysis results
    document_sections = state.get("document_sections", {})
    correct_sections = state.get("correct_sections", [])
    sections_to_modify = state.get("sections_to_modify", [])
    missing_sections = state.get("missing_sections", [])
    compliance_requirements = state.get("compliance_requirements", [])
    technical_tests = state.get("technical_tests", [])
    queue_wait_analysis = state.get("queue_wait_analysis", {})
    risk_assessment = state.get("risk_assessment", [])
    environmental_challenges = state.get("environmental_challenges", [])
    station_diagnostics = state.get("station_diagnostics", [])
    document_metadata = state.get("document_metadata", {})
    
    try:
        # FIXED: Format parameters to match PROMPT_GAP_REPORT_SYNTHESIS
        synthesis_prompt = PROMPT_GAP_REPORT_SYNTHESIS.format(
            document_sections=", ".join(document_sections.keys()),  # FIXED: was missing
            station_name=document_metadata.get("station_name", "Not specified"),  # FIXED
            region=document_metadata.get("region", "Not specified"),  # FIXED
            request_type=document_metadata.get("request_type", "Not specified"),  # FIXED
            correct_sections=json.dumps(correct_sections, indent=2),
            sections_to_modify=json.dumps(sections_to_modify, indent=2),
            missing_sections=json.dumps(missing_sections, indent=2),
            compliance_requirements=json.dumps(compliance_requirements[:10], indent=2),
            technical_tests=json.dumps(technical_tests[:10], indent=2),
            queue_wait_analysis=json.dumps(queue_wait_analysis, indent=2),
            risk_assessment=json.dumps(risk_assessment[:8], indent=2),
            environmental_challenges=json.dumps(environmental_challenges[:6], indent=2),
            station_diagnostics=json.dumps(station_diagnostics[:15], indent=2)
        )
        
        # Call LLM to synthesize
        print("Synthesizing comprehensive gap analysis report...")
        
        response = llm.invoke([
            SystemMessage(content="You are an expert technical writer creating gap analysis reports."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        gap_report_content = response.content
        
        print(f"Generated report ({len(gap_report_content)} characters)")
        
        return {
            "gap_report_content": gap_report_content,
            "progress_percent": 95,
            "current_step": "Gap analysis report synthesized"
        }
        
    except Exception as e:
        print(f"ERROR synthesizing report: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "gap_report_content": "Error generating report",
            "progress_percent": 93,
            "current_step": f"Report synthesis error: {str(e)}"
        }


####################################################################
def pgicq_gap_generate_report_file(state: GridState) -> Dict:
    """
    Generate DOCX report file from synthesized content.
    
    Args:
        state: Current graph state with gap_report_content
        
    Returns:
        Dict with gap_report_path (file path)
    """
    print(f"\n{'='*60}")
    print("NODE: pgicq_gap_generate_report_file")
    print(f"{'='*60}")
    
    from constants import GAP_ANALYSIS_REPORTS_DIR
    
    gap_report_content = state.get("gap_report_content", "")
    document_metadata = state.get("document_metadata", {})
    
    if not gap_report_content:
        print("ERROR: No report content to generate")
        return {
            "gap_report_path": None,
            "progress_percent": 95,
            "current_step": "Report generation failed - no content"
        }
    
    try:
        # Generate DOCX file
        print("Generating DOCX report file...")
        report_path = generate_gap_analysis_docx(
            gap_report_content=gap_report_content,
            document_metadata=document_metadata,
            output_dir=GAP_ANALYSIS_REPORTS_DIR
        )
        
        print(f"✅ Report file generated: {report_path}")
        
        return {
            "gap_report_path": report_path,
            "progress_percent": 100,
            "current_step": "Gap analysis completed successfully"
        }
        
    except Exception as e:
        print(f"ERROR generating report file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "gap_report_path": None,
            "progress_percent": 95,
            "current_step": f"Report file generation error: {str(e)}"
        }
