import re
from typing import List, Dict, Any, Optional
from utils.db import save_memory, get_user_memories, update_conversation_title
from utils.llm import query_llm

CLINICAL_SYSTEM_PROMPT = """You are Prescription Trend AI — an advanced, empathetic, and precision-driven Clinical Intelligence & Pharmacology AI Assistant.
You specialize in:
1. Prescription pattern interpretation, drug interactions, contraindications, and therapeutic alternatives.
2. Clinical dataset analysis, patient risk factors, and epidemiological trends.
3. Dosage guidelines, renal/hepatic adjustments, and evidence-based clinical reasoning.

CRITICAL INSTRUCTIONS:
- You have access to persistent patient & clinical memories provided below under [CLINICAL MEMORY CONTEXT]. Use this memory seamlessly to personalize your responses and maintain clinical continuity.
- Format responses clearly using markdown headers (###), bullet points, bold key terms, and cautionary callouts for high-risk drug combinations.
- Maintain a professional, rigorous, yet accessible medical tone.
"""


def build_conversation_prompt(
    messages: List[Dict[str, Any]], 
    user_memories: Optional[List[Dict[str, Any]]] = None,
    custom_system_prompt: Optional[str] = None
) -> str:
    """
    Construct a full multi-turn conversational prompt with system persona, 
    stored long-term memory context, and conversation history.
    """
    system_text = custom_system_prompt or CLINICAL_SYSTEM_PROMPT

    # 1. Add Stored Long-Term Memories
    memory_section = ""
    if user_memories:
        mem_items = [f"- {m['content']}" for m in user_memories[:15]]
        memory_section = (
            "\n\n[PERSISTENT CLINICAL MEMORY & CONTEXT]\n"
            "The following facts are remembered from previous interactions with this user/patient context:\n"
            + "\n".join(mem_items)
            + "\n[END OF CLINICAL MEMORY]\n"
        )

    full_system = system_text + memory_section

    # 2. Add Recent Conversation Turns (Sliding window of last 10 messages)
    history_turns = []
    recent_messages = messages[-10:] if len(messages) > 10 else messages

    # If the last message is from user (the current prompt), we format prior turns
    for msg in recent_messages[:-1]:
        role_label = "User" if msg["role"] == "user" else "Clinical Assistant"
        history_turns.append(f"{role_label}: {msg['content']}")

    current_user_msg = recent_messages[-1]["content"] if recent_messages else ""

    if history_turns:
        history_context = "\n\n[CONVERSATION HISTORY]\n" + "\n\n".join(history_turns) + "\n\n"
    else:
        history_context = "\n\n"

    full_prompt = (
        f"{full_system}{history_context}"
        f"User: {current_user_msg}\n\n"
        f"Clinical Assistant:"
    )

    return full_prompt


def generate_chat_title(first_message: str) -> str:
    """
    Generate a concise (3-6 words) title for a new chat session based on the first prompt.
    """
    cleaned = first_message.strip()
    if len(cleaned) < 50 and not any(c in cleaned for c in ["\n", "\r", "?", "!"]):
        return cleaned.title()

    prompt = (
        f"Generate a concise 3-5 word title summarizing this clinical chat prompt. "
        f"Return ONLY the title text, no quotes, no markdown, no explanation.\n\n"
        f"Prompt: {cleaned[:300]}"
    )
    
    title = query_llm(prompt, system="You are an expert at generating short 3-5 word concise titles.", temperature=0.2)
    
    if "ERROR_OLLAMA_DOWN" in title or "ERROR:" in title or len(title.strip()) < 2:
        # Fallback heuristic title
        words = [w for w in cleaned.replace("\n", " ").split(" ") if w.strip()]
        fallback = " ".join(words[:5])
        return (fallback[:35] + "...") if len(fallback) > 35 else (fallback.title() or "Clinical Consultation")

    # Clean any quotes or prefixes
    title = re.sub(r'^[#"\s*]+|[#"\s*]+$', '', title.strip())
    return title[:45] if title else "Clinical Consultation"


def extract_and_save_clinical_memory(user_id: int, user_message: str, assistant_response: str, conv_id: Optional[str] = None):
    """
    Detect and extract clinical key facts from the conversation turn to persist in long-term memory.
    E.g., patient conditions, allergies, medication regimens, or physician preferences.
    """
    text_to_check = f"User: {user_message}\nAssistant: {assistant_response}"

    # Heuristic check to avoid unnecessary LLM calls if message is too trivial
    clinical_keywords = [
        "patient is", "patient has", "diagnosed with", "allergic to", "allergy",
        "taking", "prescribed", "history of", "mg", "dosage", "stage", "ckd",
        "hypertension", "diabetes", "cardiac", "renal", "specialty", "dr.", "doctor"
    ]
    has_clinical_fact = any(kw in user_message.lower() for kw in clinical_keywords)

    if not has_clinical_fact:
        return

    extraction_prompt = (
        "Extract any permanent clinical facts, patient details (age, gender, diagnoses, allergies, current medications), "
        "or physician preferences mentioned in the user statement below. "
        "Return ONLY a bulleted list of 1-2 concise facts (starting with '- '), or return 'NONE' if no persistent clinical fact exists. "
        "Do NOT invent information.\n\n"
        f"User Statement: {user_message}"
    )

    extracted = query_llm(extraction_prompt, system="You are a clinical entity extractor.", temperature=0.1)

    if "ERROR" not in extracted and "NONE" not in extracted.upper() and extracted.strip():
        lines = extracted.strip().split("\n")
        for line in lines:
            line_clean = line.strip().lstrip("-*• ").strip()
            if len(line_clean) > 8 and not line_clean.lower().startswith("none"):
                save_memory(
                    user_id=user_id,
                    content=line_clean,
                    memory_type="clinical_fact",
                    conversation_id=conv_id
                )
