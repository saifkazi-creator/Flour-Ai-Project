import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.manual_tool import manual_search
from tools.log_tool import log_search
from tools.schedule_tool import schedule_search
from tools.troubleshooting_tool import troubleshooting_search
from rag.llm_engine import stream_llm_answer, stream_llm_direct

# Module-level memory
last_context = ""

# ─────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────

# Keywords that signal the user wants to SOLVE a real problem
# and therefore needs the RAG / maintenance data pipeline
MAINTENANCE_PROBLEM_KEYWORDS = [
    # troubleshooting
    "fix", "repair", "broken", "fault", "failed", "failure", "issue", "problem",
    "error", "vibration", "noise", "leak", "overheat", "overheating", "jam",
    "stuck", "not working", "won't start", "tripping", "seized", "worn",
    "damage", "damaged", "crack", "cracked", "why is", "why does",
    "cause", "reason", "diagnose", "diagnosis",
    # logs
    "history", "previous", "before", "last time", "past", "record", "logged",
    # schedule
    "schedule", "due", "next service", "upcoming", "planned", "when should",
    "how often", "frequency", "maintenance interval",
    # manual lookup for a procedure
    "how to replace", "how to fix", "how to repair", "how to adjust",
    "procedure for", "steps to", "torque", "specification", "spec",
    "reassemble", "disassemble", "install", "remove",
    # document / manual references — queries about uploaded or existing docs
    "manual", "document", "pdf", "uploaded", "according to",
    "from the", "in the", "refer",
    # common technical queries that should use documents
    "temperature", "pressure", "capacity", "speed", "rpm", "voltage",
    "dimension", "weight", "size", "rating", "power", "output",
    "lubrication", "lubricant", "oil", "grease", "belt", "bearing",
    "alignment", "clearance", "gap", "tolerance", "setting",
    "start", "stop", "operate", "operation", "run", "running",
    "safety", "precaution", "warning", "maintenance",
]

# Keywords that are clearly general / informational — go straight to LLM
GENERAL_KEYWORDS = [
    "what is a", "what are", "explain", "describe", "tell me about",
    "how does a", "how do", "definition", "overview", "introduction",
    "hi", "hello", "hey", "thanks", "thank you", "good morning",
    "good afternoon", "who are you", "what can you do",
]


def classify_intent(query: str) -> str:
    """
    Returns one of: "general" | "log" | "schedule" | "troubleshooting" | "manual"

    Order of precedence:
      1. If query matches a maintenance PROBLEM keyword → route to maintenance pipeline
      2. If query matches a general/informational keyword  → route to direct LLM
      3. Default → if there is an uploaded file, route to manual search;
                    otherwise route to direct LLM
    """
    q = query.lower()

    # Check problem / maintenance intent first
    for kw in MAINTENANCE_PROBLEM_KEYWORDS:
        if kw in q:
            return _route_maintenance(q)

    # Check general / informational intent
    for kw in GENERAL_KEYWORDS:
        if kw in q:
            return "general"

    # Default: if user has uploaded a document, assume they want to query it;
    # otherwise fall back to direct LLM for casual conversation.
    try:
        if st.session_state.get("last_uploaded_file"):
            return "manual"
    except Exception:
        pass

    return "general"


def _route_maintenance(q: str) -> str:
    """Sub-router for confirmed maintenance queries — picks the right data source."""
    log_keywords = ["history", "previous", "before", "last time", "past", "record", "logged"]
    schedule_keywords = ["schedule", "due", "next service", "upcoming", "planned",
                         "when should", "how often", "frequency", "maintenance interval"]
    troubleshooting_keywords = [
        "why", "cause", "reason", "fault", "problem", "vibration",
        "noise", "leak", "overheat", "overheating", "fix", "repair",
        "broken", "failed", "failure", "issue", "error", "jam", "stuck",
        "not working", "won't start", "tripping", "seized", "worn",
        "damage", "damaged", "crack", "cracked", "diagnose", "diagnosis"
    ]

    for kw in log_keywords:
        if kw in q:
            return "log"
    for kw in schedule_keywords:
        if kw in q:
            return "schedule"
    for kw in troubleshooting_keywords:
        if kw in q:
            return "troubleshooting"
    return "manual"


def enhance_query_with_memory(query: str) -> str:
    """Prepend last context if query references a prior topic."""
    global last_context
    context_words = ["this", "that", "it", "the issue", "the problem"]
    if last_context and any(word in query.lower() for word in context_words):
        return f"{last_context} {query}"
    return query


# ─────────────────────────────────────────────
# MAIN STREAMING GENERATOR
# ─────────────────────────────────────────────

def stream_agent_response(query: str):
    """
    Generator — yields string chunks to the Streamlit UI.

    Routes to:
      • Direct LLM   — casual chat, general knowledge questions
      • RAG pipeline — maintenance logs, schedules, troubleshooting, manuals
    """
    global last_context

    enhanced = enhance_query_with_memory(query)
    intent = classify_intent(enhanced)

    # ── General / conversational ──────────────────────────────────────────
    if intent == "general":
        yield "💬 **Assistant:**\n\n"
        for chunk in stream_llm_direct(enhanced):
            yield chunk

    # ── Maintenance logs ──────────────────────────────────────────────────
    elif intent == "log":
        yield "📜 **Historical Maintenance Records:**\n\n"
        docs = log_search(enhanced)
        if docs:
            for doc in docs:
                yield doc.page_content + "\n\n---\n\n"
        else:
            yield "_No matching maintenance log records found._\n\n"
        yield "\n\n> 🟢 **Confidence: High** *(From maintenance logs)*"

    # ── Schedule ──────────────────────────────────────────────────────────
    elif intent == "schedule":
        yield "🛠 **Scheduled Maintenance:**\n\n"
        docs = schedule_search(enhanced)
        if docs:
            for doc in docs:
                yield doc.page_content + "\n\n---\n\n"
        else:
            yield "_No scheduled maintenance found for this query._\n\n"
        yield "\n\n> 🟢 **Confidence: High** *(From maintenance schedule)*"

    # ── Troubleshooting ───────────────────────────────────────────────────
    elif intent == "troubleshooting":
        yield "🔍 **Troubleshooting Analysis:**\n\n"
        docs = troubleshooting_search(enhanced)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            for chunk in stream_llm_answer(enhanced, context):
                yield chunk
        else:
            for chunk in stream_llm_answer(enhanced, "No troubleshooting data found."):
                yield chunk
        yield "\n\n> 🟡 **Confidence: Medium** *(From troubleshooting guide)*"

    # ── Manual / procedure ────────────────────────────────────────────────
    else:
        yield "📘 **Equipment Manual Reference:**\n\n"

        source_filter = None
        try:
            source_filter = st.session_state.get("last_uploaded_file", None)
        except Exception:
            pass

        docs = []
        if source_filter:
            docs = manual_search(enhanced, source_filter=source_filter)
        if not docs:
            docs = manual_search(enhanced)

        if not docs:
            fallback_context = (
                "No relevant manual data found. "
                "Answer using general industrial flour mill maintenance knowledge."
            )
            for chunk in stream_llm_answer(enhanced, fallback_context):
                yield chunk
            yield "\n\n> 🔴 **Confidence: Low** *(LLM general knowledge — no manual found)*"
        else:
            context = "\n\n".join([d.page_content for d in docs])
            for chunk in stream_llm_answer(enhanced, context):
                yield chunk
            yield "\n\n> 🟢 **Confidence: High** *(From equipment manuals)*"

    last_context = query
