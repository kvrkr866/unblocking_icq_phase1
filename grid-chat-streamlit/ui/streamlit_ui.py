"""
##########################################################
# Capstone Team 16
# Streamlit UI Module
#
# This module handles all UI-related functionality.
# It is completely separate from backend logic.
# Author: RK (kvrkr866@gmail.com)
##########################################################
v27
"""

import streamlit as st
import time
from typing import Optional, List, Dict
from datetime import datetime


class GridChatUI:
    """Power Grids Interconnection Queue Analyzer system."""
    
    # Predefined sample questions
    SAMPLE_QUESTIONS = [
        "Show me all events from the last 24 hours",
        "What are the different types of events in the system?",
        "List all events with high severity",
        "Count total events by severity level",
        "Show me critical events in the last week",
        "I want to connect to CAISO, what is the interconnection process that I should follow?",
        "I want to connect to the 110 kV level, what are feasible points above that level?",
        "What are the expected load and generation levels in 2 years and in 5 years in [name of a specific station]?",
        "I am interested in connecting to station [name of a specific station] in CAISO, what are the planned projects and which interconnection requests are in the queue for that point?"
    ]
    
    def __init__(self):
        """Initialize UI components."""
        self.backend = None
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.backend_ready = False
            st.session_state.backend_instance = None
            st.session_state.chat_history = []
            st.session_state.processing = False
            st.session_state.current_query = None
            st.session_state.current_response = None
    
    def _initialize_backend(self):
        """Initialize backend components (lazy loading)."""
        if not st.session_state.backend_ready:
            with st.spinner("🔄 Initializing Power Grids Interconnection Queue Analyzer backend..."):
                try:
                    import sys
                    import os
                    # Add backend directory to path
                    backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
                    if backend_path not in sys.path:
                        sys.path.insert(0, backend_path)
                    
                    from grid_chat import GridChatBackend
                    backend = GridChatBackend()
                    backend.initialize()
                    
                    # Store in session state
                    st.session_state.backend_instance = backend
                    st.session_state.backend_ready = True
                    st.session_state.initialized = True
                    
                    st.success("✅ Backend initialized successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to initialize backend: {str(e)}")
                    st.exception(e)
                    st.stop()
        
        # Set the instance variable from session state
        self.backend = st.session_state.backend_instance
    
    def _render_header(self):
        """Render application header."""
        st.title("⚡ Power Grids Interconnection Queue Analyzer")
        st.caption("Ask questions about your interconnection queue in natural language")
    
    def _render_sidebar(self):
        """Render sidebar with sample questions and info."""
        with st.sidebar:
            st.markdown("""
                <h3 style='font-size: 18px; margin-bottom: 15px; margin-top: 0px;'>
                    📋 Sample Questions
                </h3>
            """, unsafe_allow_html=True)
            
            for idx, question in enumerate(self.SAMPLE_QUESTIONS):
                if st.button(
                    question, 
                    key=f"sample_q_{idx}",
                    use_container_width=True,
                    disabled=st.session_state.processing
                ):
                    return question
            
            st.divider()
            
            # System info
            st.markdown("### ℹ️ System Info")
            if st.session_state.backend_ready:
                st.success("Backend: Ready ✅")
            else:
                st.warning("Backend: Not initialized")
            
            st.info(f"Total queries: {len(st.session_state.chat_history)}")
            
            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_query = None
                st.session_state.current_response = None
                st.rerun()
        
        return None
    
    def _render_current_qa(self):
        """Render current question and answer."""
        if st.session_state.current_query:
            st.markdown("---")
            
            # Current question
            st.info(f"**Question:** {st.session_state.current_query}")
            
            # Current answer
            if st.session_state.current_response:
                st.success(f"**Answer:** {st.session_state.current_response}")
            
            st.markdown("---")
    
    def _render_history(self):
        """Render chat history in a scrollable box at the bottom."""
        if st.session_state.chat_history:
            st.markdown("<br/>", unsafe_allow_html=True)
            
            st.markdown("### 📜 History")
            
            # Create a scrollable container using Streamlit's native container
            with st.container():
                # Use expander or native scrolling with height
                history_container = st.container(height=400)
                
                with history_container:
                    # Reverse to show most recent first
                    for idx, entry in enumerate(reversed(st.session_state.chat_history)):
                        # Question
                        st.markdown(f"**Q:** {entry['query']}")
                        
                        # Answer
                        st.markdown(f"**A:** {entry['response']}")
                        
                        # Metadata
                        st.caption(f"⏱️ {entry['duration']:.2f}s | 🕒 {entry['timestamp']}")
                        
                        # Dotted line separator (except for last item)
                        if idx < len(st.session_state.chat_history) - 1:
                            st.markdown("---")
    
    def _process_query(self, query: str):
        """
        Process user query through backend.
        
        Args:
            query: User's natural language query
        """
        if not query or not query.strip():
            st.warning("⚠️ Please enter a valid question")
            return
        
        st.session_state.processing = True
        st.session_state.current_query = query
        st.session_state.current_response = None
        
        # Rerun to show the question immediately
        st.rerun()
    
    def _execute_query(self):
        """Execute the current query and show completion time."""
        query = st.session_state.current_query
        
        try:
            # Record start time
            start_time = time.time()
            
            # Show processing status
            with st.spinner("🔄 Processing query..."):
                # Call backend (this is the long-running operation)
                response = self.backend.process_query(query)
            
            # Calculate final duration
            duration = time.time() - start_time
            
            # Store response
            st.session_state.current_response = response
            
            # Add to history (most recent first by inserting at beginning)
            st.session_state.chat_history.insert(0, {
                'query': query,
                'response': response,
                'duration': duration,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Show completion message with duration
            st.success(f"✅ Query completed in {duration:.2f} seconds")
            
            # Keep success message visible for a moment
            time.sleep(1.5)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.current_response = f"Error processing query: {str(e)}"
        
        finally:
            st.session_state.processing = False
            st.rerun()
    
    def _render_input_section(self):
        """Render user input section."""
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Enter your question:",
                placeholder="e.g., Show me all critical events from yesterday",
                key="user_input_text",
                disabled=st.session_state.processing,
                label_visibility="collapsed"
            )
        
        with col2:
            # Blue submit button
            submit_clicked = st.button(
                "Submit",
                use_container_width=True,
                disabled=st.session_state.processing,
                type="primary"
            )
        
        # Process submission
        if submit_clicked and user_input:
            return user_input
        
        return None
    
    def run(self):
        """Main run method for the UI."""
        
        # Step 1: Render Header at top (this will be pinned)
        self._render_header()
        
        # Step 2: Auto-initialize backend if not already done
        if not st.session_state.initialized:
            self._initialize_backend()
            return
        
        # Step 3: Ensure backend is loaded in session
        if st.session_state.backend_ready:
            self.backend = st.session_state.backend_instance
        
        # Step 4: Render Sidebar
        selected_sample = self._render_sidebar()
        
        # Step 5: Render Input Section (this stays at top)
        user_query = self._render_input_section()
        
        # Step 6: Determine query to process
        query_to_process = user_query or selected_sample
        
        # Step 7: If there's a new query and not processing, start processing
        if query_to_process and not st.session_state.processing and query_to_process != st.session_state.current_query:
            self._process_query(query_to_process)
            return
        
        # Step 8: If currently processing, execute the query
        if st.session_state.processing and st.session_state.current_query and not st.session_state.current_response:
            self._execute_query()
            return
        
        # Step 9: Render current Q&A (stays visible)
        self._render_current_qa()
        
        # Step 10: Render History in scrollable box at bottom
        self._render_history()
        
        # Footer
        st.markdown("<br/>", unsafe_allow_html=True)
        st.caption("Power Grids Interconnection Queue Analyzer v1.0 | Capstone Team 16")


# Custom CSS for better styling
def inject_custom_css():
    """Inject custom CSS for professional look."""
    st.markdown("""
        <style>
        /* Reduce top padding significantly */
        .block-container {
            padding-top: 0.5rem !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] button {
            text-align: left !important;
            font-size: 14px !important;
            padding: 10px !important;
        }
        
        /* Input field styling */
        .stTextInput input {
            font-size: 15px !important;
        }
        
        /* FORCE Submit Button to BLUE - multiple selectors */
        button[kind="primary"],
        .stButton > button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        .stButton button[type="submit"],
        button[data-testid="baseButton-primary"] {
            background-color: #1f77b4 !important;
            border-color: #1f77b4 !important;
            color: white !important;
        }
        
        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #1557a0 !important;
            border-color: #1557a0 !important;
            color: white !important;
        }
        
        /* Make header sticky */
        [data-testid="stHeader"] {
            position: sticky;
            top: 0;
            background-color: var(--background-color);
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)