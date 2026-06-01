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
- 🖥 Local LLM via Ollama
- 📊 Confidence Scoring
""")

# --- Model Selection ---
st.sidebar.subheader("🤖 Local LLM Selection")
from rag.llm_engine import get_local_ollama_models, set_current_model

available_models = get_local_ollama_models()
default_model = "mistral:latest"
if default_model not in available_models and available_models:
    mistral_variants = [m for m in available_models if "mistral" in m.lower()]
    if mistral_variants:
        default_model = mistral_variants[0]
    else:
        default_model = available_models[0]

if default_model not in available_models:
    available_models.insert(0, default_model)

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = default_model

selected_model = st.sidebar.selectbox(
    "Active Model",
    options=available_models,
    index=available_models.index(st.session_state["selected_model"]) if st.session_state["selected_model"] in available_models else 0
)

if selected_model != st.session_state["selected_model"]:
    st.session_state["selected_model"] = selected_model
    set_current_model(selected_model)
    st.sidebar.success(f"🤖 Switched to {selected_model}!")
else:
    set_current_model(st.session_state["selected_model"])

# File uploader — key changes after each upload so the widget auto-clears
if "upload_counter" not in st.session_state:
    st.session_state["upload_counter"] = 0
if "ingested_files" not in st.session_state:
    st.session_state["ingested_files"] = set()

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Document", type=["pdf", "csv", "txt"],
    key=f"file_uploader_{st.session_state['upload_counter']}"
)
if uploaded_file is not None:
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, uploaded_file.name)

    if uploaded_file.name not in st.session_state["ingested_files"]:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state["last_uploaded_file"] = uploaded_file.name

        # Auto-ingest: rebuild the vector store with the new file included,
        # then reload the in-memory vectorstore so the agent sees it immediately.
        with st.spinner("🔄 Ingesting uploaded document into knowledge base..."):
            from rag.retriever import reload_vectorstore, get_vectorstore
            reload_vectorstore()
            get_vectorstore()

        st.session_state["ingested_files"].add(uploaded_file.name)
        st.sidebar.success(f"✅ Uploaded & ingested: {uploaded_file.name}")

        # Increment counter to reset the file uploader widget for next upload
        st.session_state["upload_counter"] += 1
        st.rerun()
    else:
        st.sidebar.info(f"📄 Already ingested: {uploaded_file.name}")

# Rebuild knowledge base button
if st.sidebar.button("🔄 Rebuild AI Knowledge Base"):
    with st.spinner("🔄 Rebuilding knowledge base..."):
        from rag.retriever import reload_vectorstore, get_vectorstore
        reload_vectorstore()
        get_vectorstore()
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
    st.rerun()

# --- Save Resolved Query to Logs ---
if "log_success_msg" in st.session_state:
    st.sidebar.success(st.session_state["log_success_msg"])
    del st.session_state["log_success_msg"]

last_user_query = ""
last_ai_response = ""
messages = st.session_state.get("messages", [])
for i in range(len(messages) - 1, -1, -1):
    if messages[i]["role"] == "assistant" and i > 0 and messages[i-1]["role"] == "user":
        last_user_query = messages[i-1]["content"]
        last_ai_response = messages[i]["content"]
        break

if last_ai_response:
    # Clean up Markdown labels/prefixes
    prefixes = [
        "💬 **Assistant:**\n\n",
        "📜 **Historical Maintenance Records:**\n\n",
        "🛠 **Scheduled Maintenance:**\n\n",
        "🔍 **Troubleshooting Analysis:**\n\n",
        "📘 **Equipment Manual Reference:**\n\n"
    ]
    for pref in prefixes:
        if last_ai_response.startswith(pref):
            last_ai_response = last_ai_response[len(pref):]
            break
            
    # Strip confidence rating footer
    footer_idx = last_ai_response.find("\n\n> ")
    if footer_idx != -1:
        last_ai_response = last_ai_response[:footer_idx]

st.sidebar.markdown("---")
with st.sidebar.expander("📝 Save Resolved Issue to Logs", expanded=False):
    st.write("Record this solution in the maintenance log for future troubleshooting lookup.")
    
    machine_options = ["Roller Mill", "Conveyor", "Bucket Elevator", "Purifier", "Other"]
    detected_idx = 0
    if last_user_query:
        for idx, m in enumerate(machine_options[:-1]):
            if m.lower() in last_user_query.lower():
                detected_idx = idx
                break
                
    log_machine = st.selectbox("Machine", options=machine_options, index=detected_idx, key="log_machine_select")
    
    if log_machine == "Other":
        machine_name = st.text_input("Enter Machine Name", key="log_custom_machine")
    else:
        machine_name = log_machine

    log_issue = st.text_input("Issue / Symptom", value=last_user_query[:150] if last_user_query else "", key="log_issue_input")
    log_action = st.text_area("Action Taken / Resolution", value=last_ai_response[:500] if last_ai_response else "", height=150, key="log_action_input")

    if st.button("💾 Save to Logs", key="save_to_logs_btn"):
        if not machine_name.strip():
            st.error("Please specify a machine name.")
        elif not log_issue.strip():
            st.error("Please specify the issue.")
        elif not log_action.strip():
            st.error("Please specify the action taken / resolution.")
        else:
            # Save to CSV
            import csv
            from datetime import datetime
            
            csv_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data", "logs", "maintenance_logs.csv"
            )
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            issue_clean = log_issue.replace("\n", " ").strip()
            action_clean = log_action.replace("\n", " ").strip()
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["date", "machine", "issue", "action_taken"])
                writer.writerow([today_str, machine_name, issue_clean, action_clean])
            
            # Update Vector Store dynamically
            from rag.retriever import get_vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is not None:
                from langchain_core.documents import Document
                content = (
                    f"Date: {today_str}\n"
                    f"Machine: {machine_name}\n"
                    f"Issue: {issue_clean}\n"
                    f"Action Taken: {action_clean}"
                )
                metadata = {
                    "machine": machine_name,
                    "type": "maintenance_log",
                    "date": today_str
                }
                vectorstore.add_documents([Document(page_content=content, metadata=metadata)])
                if hasattr(vectorstore, "persist"):
                    vectorstore.persist()
                st.session_state["log_success_msg"] = "✅ Log entry saved and vector store updated!"
                st.rerun()
            else:
                st.session_state["log_success_msg"] = "⚠️ Saved to CSV, but vector store not loaded."
                st.rerun()
