# Grid Chat Assistant - Streamlit UI

## Capstone Team 16 - Phase 1 with Streamlit Integration

This project provides a Streamlit-based user interface for the Grid Chat system, which uses LangGraph to process natural language queries about grid diagnostics data.

---

## 📁 Project Structure

```
grid-chat-streamlit/
│
├── app.py                          # Main Streamlit entry point
│
├── backend/
│   ├── __init__.py                 # Backend package initializer
│   ├── phase1.py                   # Core LangGraph implementation (your original file)
│   ├── grid_chat.py                # Backend wrapper for UI isolation
│   └── griddiagnostics.py          # Database interface (your existing file)
│
├── ui/
│   ├── __init__.py                 # UI package initializer
│   └── streamlit_ui.py             # Streamlit UI implementation
│
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (create this)
└── README.md                       # This file
```

---

## 🚀 Setup Instructions

### 1. Create Project Structure

```bash
# Create main directory
mkdir grid-chat-streamlit
cd grid-chat-streamlit

# Create subdirectories
mkdir backend ui

# Create __init__.py files
touch backend/__init__.py
touch ui/__init__.py
```

### 2. Copy Files

1. Copy your **phase1.py** to `backend/phase1.py`
2. Copy your **griddiagnostics.py** to `backend/griddiagnostics.py`
3. Create the other files as provided above

### 3. Create Environment File

Create a `.env` file in the root directory:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # If using Tavily
```

### 4. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🎯 Features

### 1. **Dual Input Methods**
   - **Predefined Questions**: Click sample questions from the sidebar
   - **Custom Input**: Type your own questions in the text box

### 2. **Real-time Processing**
   - Live status updates during query processing
   - Response time tracking (seconds)
   - Processing indicators

### 3. **Chat History**
   - Complete conversation history
   - Expandable responses
   - Timestamp and duration for each query
   - Clear history option

### 4. **Clean Architecture**
   - **Complete separation** between UI and backend
   - Backend changes don't require UI modifications
   - Easy to extend with new features

---

## 🏗️ Architecture

### Initialization Flow

1. **App Entry** (`app.py`)
   - Streamlit configuration
   - UI initialization

2. **UI Initialization** (`ui/streamlit_ui.py`)
   - Session state setup
   - UI rendering

3. **Backend Initialization** (`backend/grid_chat.py`)
   - Lazy loading on first use
   - LLM and graph compilation
   - One-time setup

### Execution Flow

1. **User Input** (UI Layer)
   - Text input or sample question selection
   - Input validation

2. **Backend Processing** (Backend Layer)
   - Query passed to `GridChatBackend.process_query()`
   - Wrapper calls `GridChat.process_message()`
   - LangGraph executes workflow

3. **Response Display** (UI Layer)
   - Duration calculation
   - Response rendering
   - History update

---

## 🔧 Adding New Backend Features

To add new features (RAG, vector DB, tools, etc.):

1. **Modify only backend files**:
   - Update `backend/phase1.py` for core logic
   - Update `backend/grid_chat.py` if new interface methods needed

2. **NO UI changes required** unless you want to:
   - Add new UI elements
   - Change display format

Example - Adding RAG:
```python
# In backend/phase1.py
class GridChat:
    def __init__(self):
        self.llm = None
        self.graph = None
        self.vector_store = None  # NEW
        self.tracer = None
    
    def initialize(self):
        # Existing initialization
        self._setup_vector_db()  # NEW
        
    def _setup_vector_db(self):  # NEW
        # Vector DB setup logic
        pass
```

No changes to `app.py` or `ui/streamlit_ui.py` needed!

---

## 📝 Sample Questions

The UI includes these predefined questions:
- "Show me all events from the last 24 hours"
- "What are the different types of events in the system?"
- "List all events with high severity"
- "Count total events by severity level"
- "Show me critical events in the last week"

You can customize these in `ui/streamlit_ui.py` → `SAMPLE_QUESTIONS` list.

---

## 🐛 Troubleshooting

### Backend initialization fails
- Check your `.env` file has correct API keys
- Verify `griddiagnostics.py` is in the `backend/` folder
- Ensure database file is accessible

### Import errors
- Make sure `__init__.py` files exist in both `backend/` and `ui/` folders
- Run from project root directory
- Check virtual environment is activated

### Streamlit issues
- Clear cache: `streamlit cache clear`
- Restart Streamlit server
- Check terminal for error messages

---

## 📦 File Download Bundle

All files are provided above. Create the directory structure and copy each file to its respective location as shown in the **Project Structure** section.

---

## 🎓 Team

**Capstone Team 16**  
Phase 1 - LangGraph Integration with Streamlit UI

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

To contribute or extend functionality:
1. Backend features → Modify `backend/` files only
2. UI enhancements → Modify `ui/` files only
3. Keep separation of concerns

---

**Happy Querying! 🔌⚡**
