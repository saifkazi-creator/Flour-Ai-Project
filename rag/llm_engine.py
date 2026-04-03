import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Instantiate Llama3 model once at module level.
# Requires Ollama running locally: ollama serve && ollama pull llama3
llm = ChatOllama(model="llama3", streaming=True)


def _fix_formatting(text: str) -> str:
    """
    Normalises LLM output so it renders cleanly in Streamlit markdown:
    - Converts bullet chars (bullet, middot) to markdown '- '
    - Splits inline bullets onto their own lines
    - Adds blank line before bold section headers
    - Collapses excessive blank lines
    """
    # Split inline bullets separated by bullet or middot chars into separate lines
    text = re.sub(r'\s*[\u2022\u00b7]\s*', '\n- ', text)

    # Replace any remaining leading bullet chars at line start
    text = re.sub(r'(?m)^[\s]*[\u2022\u00b7]\s*', '- ', text)

    # Ensure a blank line before bold headers e.g. **Possible Causes:**
    text = re.sub(r'(?<!\n)\n(\*\*[^\n]+\*\*)', r'\n\n\1', text)

    # Ensure each '- ' bullet starts on its own line
    text = re.sub(r'([^\n])(- )', r'\1\n\2', text)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _stream_and_fix(messages):
    """
    Accumulates the full LLM response, applies formatting fix, then yields it.
    Collecting first ensures markdown renders correctly in Streamlit.
    """
    full = ""
    for chunk in llm.stream(messages):
        if not chunk.content:
            continue
        full += str(chunk.content)
    yield _fix_formatting(full)


def stream_llm_answer(query: str, context: str):
    """
    RAG-grounded answer. Yields a cleanly formatted response from Llama3.
    """
    system_content = f"""You are an expert industrial maintenance assistant for a flour mill.
Answer ONLY using the provided context below.

STRICT FORMATTING RULES - you MUST follow these exactly:
- NEVER put multiple bullet points on one line.
- Each bullet point MUST be on its own separate line.
- Use ONLY this structure:

**Machine:** <machine name>
**Symptom:** <symptom>

**Possible Causes:**
- cause one
- cause two
- cause three

**Recommended Actions:**
- action one
- action two
- action three

Other rules:
- Do NOT invent dates, work orders, or personnel names.
- Do NOT fabricate maintenance schedules or production assumptions.
- If the context does not contain the answer, reply exactly:
  "Not available in the provided maintenance data."
- Be technical, concise, and direct.

CONTEXT:
{context}
"""
    yield from _stream_and_fix([
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ])


def stream_llm_direct(query: str):
    """
    Direct LLM answer with no RAG context - for general questions and chat.
    """
    system_content = """You are a friendly and knowledgeable assistant specialising in
industrial flour mill equipment and maintenance.

FORMATTING RULES:
- Use clear markdown formatting.
- When listing items, put EACH item on its own line starting with '- '.
- Use **bold** for section headers.
- NEVER put multiple points on a single line.
- For greetings or small talk, respond naturally in plain prose (no bullet points needed).
"""
    yield from _stream_and_fix([
        SystemMessage(content=system_content),
        HumanMessage(content=query),
    ])
