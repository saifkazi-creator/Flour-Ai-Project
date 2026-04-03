import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.agent_controller import stream_agent_response

# --- Page config ---
st.set_page_config(page_title="Flour Mill AI Assistant", page_icon="🏭", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.main-title {
    color: #1f4e79;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 4px;
}
.subtitle {
    color: #666;
    font-size: 14px;
    margin-bottom: 20px;
}
.chat-user {
    background: #e8f0fe;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
}
.chat-bot {
    background: #f0f4f0;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
}
.footer-note {
    font-size: 11px;
    color: #999;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

# --- Sidebar ---
st.sidebar.header("⚙️ System Overview")

st.sidebar.markdown("""
**Capabilities:**
- 🔧 Troubleshooting Assistance
- 📋 Maintenance Log Lookup
- 📅 Schedule Queries
- 📘 Manual Explanation
""")

st.sidebar.markdown("""
**Architecture:**
- 🤖 Tool-based Query Routing
- 🗄 ChromaDB Vector Store
- 🖥 Local LLM via Ollama (Llama3)
- 📊 Confidence Scoring
""")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Document", type=["pdf", "csv", "txt"]
)
if uploaded_file is not None:
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state["last_uploaded_file"] = uploaded_file.name
    st.sidebar.success(f"✅ Uploaded: {uploaded_file.name}")

# Rebuild knowledge base button
if st.sidebar.button("🔄 Rebuild AI Knowledge Base"):
    os.system("python ingestion/build_vector_store.py")
    st.sidebar.success("✅ Knowledge base rebuilt!")

# List uploaded files
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
if os.path.exists(uploads_dir):
    uploaded_files = os.listdir(uploads_dir)
    if uploaded_files:
        st.sidebar.markdown("**Uploaded Files:**")
        for fname in uploaded_files:
            st.sidebar.caption(f"📄 {fname}")

# Clear chat button
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state["messages"] = []

# --- Main Area ---
st.markdown('<div class="main-title">🏭 Flour Mill AI Maintenance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by RAG + Local LLM | Ask anything about maintenance</div>', unsafe_allow_html=True)

# Render prior messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a maintenance question...")

if prompt:
    # Append and display user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in stream_agent_response(prompt):
            full_response += chunk
            placeholder.markdown(full_response + " ▌")
        placeholder.markdown(full_response)
        st.caption("🔍 Tool-based retrieval + Controlled LLM synthesis")

    st.session_state["messages"].append({"role": "assistant", "content": full_response})
