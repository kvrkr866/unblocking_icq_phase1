"""
##########################################################
# Capstone Team 16
# Backend Wrapper Module
#
# This module provides a clean interface between UI and
# the core LangGraph implementation (phase1.py).
##########################################################
v4
"""

import sys
import os
from typing import Optional, List, Dict

# Add backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Now import from phase1
from phase1 import GridChat


class GridChatBackend:
    """
    Backend wrapper for Grid Chat system.
    
    This class provides a clean interface for the UI layer,
    isolating it from the internal implementation details.
    """
    
    def __init__(self):
        """Initialize backend wrapper."""
        self.grid_chat = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the Grid Chat system.
        
        This loads all necessary models, builds the graph,
        and prepares the system for processing queries.
        """
        if not self._initialized:
            self.grid_chat = GridChat()
            self.grid_chat.initialize()
            self._initialized = True
    
    def process_query(
        self, 
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a user query and return the response.
        
        Args:
            query: User's natural language question
            chat_history: Optional chat history for context
        
        Returns:
            String response from the system
        
        Raises:
            RuntimeError: If backend is not initialized
            Exception: For any processing errors
        """
        if not self._initialized or self.grid_chat is None:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first."
            )
        
        try:
            response = self.grid_chat.process_message(query, chat_history)
            return response
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if backend is ready to process queries."""
        return self._initialized and self.grid_chat is not None
    
    def get_status(self) -> Dict[str, any]:
        """
        Get current status of the backend.
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self._initialized,
            "ready": self.is_ready(),
            "has_graph": self.grid_chat is not None and self.grid_chat.graph is not None,
            "has_llm": self.grid_chat is not None and self.grid_chat.llm is not None
        }