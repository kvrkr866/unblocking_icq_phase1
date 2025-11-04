"""
##########################################################
# Capstone Team 16
# Streamlit UI Entry Point
# 
# This is the main entry point for the Streamlit application.
# Run with: streamlit run app.py
##########################################################
"""

import streamlit as st
from ui.streamlit_ui import GridChatUI

def main():
    """Main entry point for Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Grid Chat Assistant",
        page_icon="🔌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run UI
    ui = GridChatUI()
    ui.run()

if __name__ == "__main__":
    main()
