"""
Prompt templates for script analysis.

Prompt Engineering Design Decisions:
────────────────────────────────────

1. /no_think prefix (line 1 of SYSTEM_PROMPT):
   Qwen3 supports a "thinking" mode where it reasons step-by-step inside
   <think>...</think> tags before answering. While useful for math/logic,
   it adds latency without meaningful quality gain on creative analysis tasks.
   The /no_think directive disables this, giving faster responses.
   Remove it if you want deeper (but slower) analysis.

2. Persona priming ("expert script analyst"):
   The system prompt establishes domain expertise in SHORT-FORM content
   specifically — microdramas, web series, OTT shorts. This is not generic
   "writing analysis" but targets Bullet's specific content vertical.

3. Grounding instruction ("reference exact moments"):
   Without this, LLMs default to vague, generic advice ("improve pacing").
   This instruction forces the model to cite specific lines and scenes,
   which is the difference between useful and useless script notes.

4. Separation of concerns:
   The system prompt defines WHO the model is and HOW it should behave.
   The user prompt injects WHAT to analyze and the per-field evaluation rubric.
   This separation makes it easy to swap scripts without re-engineering prompts.

5. No JSON formatting instructions in prompts:
   Unlike typical LLM JSON workflows, we do NOT ask the model to "respond in JSON"
   or provide format examples. Ollama's native format= parameter handles schema
   enforcement at the inference level, so the prompt focuses purely on ANALYSIS
   QUALITY, not output formatting. This is a key architectural advantage.
"""

# ─── System Prompt ──────────────────────────────────────────────────────────────
# The /no_think prefix MUST be the very first line — Qwen3 parses it as a
# top-level directive before processing the rest of the prompt.
SYSTEM_PROMPT = """/no_think
You are an expert script analyst specializing in short-form scripted content:
microdramas, web series, and OTT short films. You deeply understand narrative
structure, emotional storytelling, and what hooks audiences into watching the next episode.

Analyze scripts with the precision of a story editor and the instinct of a showrunner.
Be specific and grounded in craft — reference exact moments from the script, not generic advice."""


# ─── User Prompt Builder ────────────────────────────────────────────────────────
def build_user_prompt(title: str, script_content: str) -> str:
    """Build the analysis prompt with per-field evaluation guidelines.

    The evaluation guidelines section serves as a "rubric" that tells the model
    exactly what each output field should contain. This is NOT about JSON formatting
    (Ollama handles that) — it's about CONTENT QUALITY:

    - "3-4 sentences" prevents single-line summaries or essay-length responses
    - "top 3 emotions" sets quantity expectations
    - "tied to a specific moment in THIS script" forces grounded analysis
    - "not generic tips" explicitly blocks lazy, template responses

    These guidelines work in tandem with the schema constraints: Ollama enforces
    the STRUCTURE (field names, types, nesting), while the prompt enforces the
    QUALITY (specificity, grounding, depth).
    """
    return f"""Analyze the following script:

---
Title: {title}

{script_content}
---

Evaluation guidelines:
- summary: 3-4 sentences covering plot, stakes, and emotional core
- dominant_emotions: top 3 emotions felt by the audience
- emotional_arc: 3-5 beats in chronological order with intensity 1-10
- engagement_score: 0.0-10.0 based on hook strength, conflict clarity,
  pacing, tension, and cliffhanger presence
- engagement_factors: 3-5 named elements (Opening Hook, Character Conflict,
  Dialogue Sharpness, Tension Build, Cliffhanger) each with a score and
  reasoning tied to a specific moment in THIS script
- improvement_suggestions: 4-6 concrete, actionable suggestions referencing
  specific lines or beats from this script, not generic tips
- cliffhanger: the single most suspenseful moment and why it works mechanically"""
