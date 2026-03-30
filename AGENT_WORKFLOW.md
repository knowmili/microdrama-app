# 🎬 Bullet AI Engineer Assignment — Agent Workflow

> **For:** Claude Opus 4.6 Agent  
> **Assignment:** Build a Script Analysis AI System for short-form scripted content  
> **Stack:** Python + Ollama (Qwen3) + Streamlit — **100% local, no API key required**  
> **Deliverables:** GitHub-ready codebase + README

---

## 📋 Assignment Summary

Build an AI system that accepts a short script as input and produces structured storytelling insights:

| Output | Description |
|--------|-------------|
| **Story Summary** | 3–4 line narrative overview |
| **Emotional Tone Analysis** | Dominant emotions + emotional arc evolution |
| **Engagement Score** | Overall score (0–10) + influencing factors |
| **Improvement Suggestions** | Pacing, conflict, dialogue, emotional impact tips |
| **Cliffhanger Detection** *(bonus)* | Most suspenseful moment + explanation |

---

## 🤖 Model Selection: Qwen3 via Ollama

**Recommended model: `qwen3:8b`** (primary) with `qwen3:14b` as the high-quality fallback.

### Why Qwen3?

- **Best-in-class structured JSON output** among all Ollama-available models — Ollama's native `format` parameter constrains the model to emit only valid JSON matching your schema, eliminating parse errors
- **Superior instruction-following and creative reasoning** — explicitly excels at creative writing, role-playing, and multi-turn dialogue, which maps perfectly to script/narrative analysis
- **Thinking mode support** — Qwen3 can toggle between a reasoning ("thinking") mode for deep analysis and a fast non-thinking mode; use **non-thinking mode** (`/no_think`) for this task for speed
- **Apache 2.0 licensed** — fully open, commercially usable
- **Runs on consumer hardware** — `qwen3:8b` fits in ~8GB RAM; `qwen3:14b` needs ~16GB

### Hardware-Based Model Fallback Table

| Available RAM | Recommended Model | Ollama Pull Command |
|---------------|-------------------|---------------------|
| 8 GB | `qwen3:8b` | `ollama pull qwen3:8b` |
| 16 GB | `qwen3:14b` | `ollama pull qwen3:14b` |
| 32 GB+ | `qwen3:32b` | `ollama pull qwen3:32b` |

> **The app should auto-detect which model is available and use the best one found.**

---

## 🗂️ Project Structure to Create

```
bullet-script-analyzer/
├── app.py                  # Streamlit UI entry point
├── analyzer/
│   ├── __init__.py
│   ├── pipeline.py         # Orchestration + Ollama client logic
│   ├── prompts.py          # All prompt templates
│   └── models.py           # Pydantic output schemas
├── requirements.txt
└── README.md
```

> No `.env` file needed — Ollama runs locally with no API key.

---

## ⚙️ Tech Stack

- **Language:** Python 3.11+
- **LLM Runtime:** [Ollama](https://ollama.com) (local server, must be running before app starts)
- **LLM Model:** `qwen3:8b` via the `ollama` Python SDK
- **Structured Output:** Ollama native `format` parameter (JSON Schema mode) — bypasses prompt-based JSON coercion entirely
- **UI:** Streamlit
- **Validation:** Pydantic v2
- **Ollama Python SDK:** `ollama` package

### `requirements.txt` content:
```
ollama>=0.3.0
streamlit>=1.35.0
pydantic>=2.0.0
```

---

## 🧠 Core Design Decisions

### 1. Ollama Native Structured Output (not prompt-based JSON)

Ollama supports a `format` parameter in `ollama.chat()` that accepts a raw JSON Schema. When provided, the model is **constrained at the inference level** to produce output that matches the schema exactly — no prompt tricks needed, no risk of markdown fences or preamble in output.

Implementation:
```python
import ollama
from .models import ScriptAnalysis

response = ollama.chat(
    model="qwen3:8b",
    messages=[...],
    format=ScriptAnalysis.model_json_schema(),  # Pydantic → JSON Schema
    options={"temperature": 0}                  # Deterministic output
)
```

This is the cleanest, most reliable approach: **Pydantic defines the schema → passed directly to Ollama → output guaranteed to match**.

### 2. Non-Thinking Mode for Speed

Qwen3 supports a `<think>` reasoning mode. For script analysis, toggle it **off** by prepending `/no_think` in the system prompt. This gives fast responses without sacrificing quality on this task.

### 3. Temperature = 0 for Schema Adherence

Set `temperature: 0` in Ollama options. This maximizes determinism and minimizes schema violations.

### 4. Single-Call Architecture

One LLM call returns the complete analysis JSON. Cross-dimensional coherence (emotional arc informing engagement score) is maintained naturally since everything is reasoned together.

---

## 📦 Output Schema (`analyzer/models.py`)

```python
from pydantic import BaseModel
from typing import List, Optional


class EmotionBeat(BaseModel):
    moment: str      # e.g., "Opening", "Midpoint", "Climax"
    emotion: str     # e.g., "Anticipation", "Grief", "Shock"
    intensity: int   # 1-10


class EngagementFactor(BaseModel):
    factor: str      # e.g., "Opening Hook", "Character Conflict"
    score: int       # 1-10
    reasoning: str   # Specific reasoning tied to this script


class CliffhangerMoment(BaseModel):
    moment: str       # The exact dialogue/scene beat
    explanation: str  # Why it creates tension


class ScriptAnalysis(BaseModel):
    title: str
    summary: str                              # 3-4 sentences
    dominant_emotions: List[str]              # Top 3 emotions
    emotional_arc: List[EmotionBeat]          # 3-5 chronological beats
    engagement_score: float                   # 0.0-10.0
    engagement_factors: List[EngagementFactor]
    improvement_suggestions: List[str]        # 4-6 actionable tips
    cliffhanger: Optional[CliffhangerMoment]  # Bonus field
```

---

## 📝 Prompt Engineering (`analyzer/prompts.py`)

### System Prompt

```python
SYSTEM_PROMPT = """/no_think
You are an expert script analyst specializing in short-form scripted content:
microdramas, web series, and OTT short films. You deeply understand narrative
structure, emotional storytelling, and what hooks audiences into watching the next episode.

Analyze scripts with the precision of a story editor and the instinct of a showrunner.
Be specific and grounded in craft — reference exact moments from the script, not generic advice.
"""
```

> The `/no_think` prefix at the very start disables Qwen3's chain-of-thought reasoning mode for speed. Remove it if you want deeper (but slower) analysis.

### User Prompt Builder

```python
def build_user_prompt(title: str, script_content: str) -> str:
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
- cliffhanger: the single most suspenseful moment and why it works mechanically
"""
```

---

## 🔄 Pipeline Logic (`analyzer/pipeline.py`)

```python
import ollama
from pydantic import ValidationError
from .models import ScriptAnalysis
from .prompts import SYSTEM_PROMPT, build_user_prompt

# Ordered preference: auto-select best available model
MODEL_PREFERENCE = ["qwen3:32b", "qwen3:14b", "qwen3:8b"]


def get_available_model() -> str:
    """Auto-detect the best Qwen3 model available in local Ollama."""
    try:
        available = [m.model for m in ollama.list().models]
        for model in MODEL_PREFERENCE:
            if model in available:
                return model
        # If none pulled yet, default to 8b
        return "qwen3:8b"
    except Exception:
        return "qwen3:8b"


def analyze_script(title: str, script_content: str) -> ScriptAnalysis:
    """
    Run script analysis using Ollama's native structured output mode.
    
    Key design choices:
    - format=ScriptAnalysis.model_json_schema() enforces schema at inference level
      (no prompt-based JSON coercion, no parse errors from markdown fences)
    - temperature=0 maximizes determinism and schema adherence
    - /no_think in system prompt disables Qwen3 chain-of-thought for speed
    """
    model = get_available_model()
    user_prompt = build_user_prompt(title, script_content)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format=ScriptAnalysis.model_json_schema(),  # Native JSON schema enforcement
        options={"temperature": 0},
    )

    raw = response["message"]["content"]

    try:
        return ScriptAnalysis.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"Model output did not match expected schema: {e}") from e
```

---

## 🖥️ Streamlit UI (`app.py`)

### Layout Design

```
+---------------------------------------------+
|  Sidebar: How it works + active model badge  |
+---------------------------------------------+
|  Script Title Input                          |
|  Script Content Textarea (15 rows tall)      |
|  [Analyze Script Button]                     |
+---------------------------------------------+
|  RESULTS via st.tabs():                      |
|  Summary | Emotions | Engagement             |
|  Suggestions | Cliffhanger                   |
+---------------------------------------------+
```

### Key Implementation Notes

- **Startup health check:** On app load, ping `ollama.list()` to verify Ollama is running. Show a clear `st.error` with setup instructions if not — do not let it crash with a Python traceback.
- **Model badge in sidebar:** Show which model is active (e.g., "Using qwen3:14b").
- **`st.spinner`** during analysis with message: "Analyzing your script with Qwen3..."
- **Tabs for results:** `st.tabs(["Summary", "Emotions", "Engagement", "Suggestions", "Cliffhanger"])`
- **Engagement score:** `st.metric()` showing the number + `st.progress(result.engagement_score / 10)`
- **Emotional arc:** Render each beat showing moment, emotion, and an intensity bar (use `st.progress(beat.intensity / 10)`)
- **Engagement factors:** `st.expander(factor.factor)` containing score + reasoning
- **Pre-fill sample script** on first load so demo is immediately runnable without typing

### Ollama Health Check (add at top of `app.py`)

```python
import ollama
import streamlit as st

def check_ollama():
    try:
        ollama.list()
    except Exception:
        st.error(
            "Ollama is not running. Please start it:\n\n"
            "1. Run `ollama serve` in your terminal\n"
            "2. Run `ollama pull qwen3:8b` to download the model\n"
            "3. Refresh this page"
        )
        st.stop()

check_ollama()
```

### Sample Script for Demo (pre-fill in UI)

```python
SAMPLE_TITLE = "The Last Message"

SAMPLE_SCRIPT = """SCENE
A dimly lit apartment. RIYA (28) sits by the window, staring at her phone.
An unread message notification glows on the screen: "Arjun."
She has not spoken to him in five years.

DIALOGUE
Riya: Why now?
Arjun: Because today I learned the truth.
Riya: What truth?
Arjun: That the accident was not your fault.

Riya's hand trembles. She sets the phone face-down on the table.
Then picks it up again. Types something. Deletes it. Types again.

Riya: (quietly) Then whose was it?

The typing indicator appears on Arjun's side. Then stops.
Then appears again. The screen cuts to black."""
```

---

## 📄 README.md Content

Write the README with these exact sections:

```markdown
# Bullet Script Analyzer

An AI-powered tool that analyzes short-form scripts and generates structured
storytelling insights: emotional arc, engagement score, improvement suggestions,
and cliffhanger detection.

Runs 100% locally via Ollama. No API key required.

## Stack
- LLM: Qwen3 (via Ollama) - local inference, Apache 2.0 licensed
- UI: Streamlit
- Schema validation: Pydantic v2

## Setup

### 1. Install Ollama
Download from https://ollama.com and install for your OS.

### 2. Pull the model
ollama pull qwen3:8b        # ~5GB download, runs on 8GB RAM
# or for better quality:
ollama pull qwen3:14b       # ~9GB download, runs on 16GB RAM

### 3. Install Python dependencies
pip install -r requirements.txt

### 4. Run
# Terminal 1
ollama serve

# Terminal 2
streamlit run app.py

## Approach

Single-call LLM architecture: one call to Qwen3 returns the entire structured
analysis as JSON. This keeps reasoning coherent across all dimensions (emotional
arc informs engagement score, etc.) and minimizes latency.

Structured output uses Ollama's native format= parameter, which enforces the
Pydantic schema at the inference level. This is more reliable than prompt-based
JSON coercion because the model cannot produce output that violates the schema.

The app auto-detects the best available Qwen3 model (32b > 14b > 8b).

## How Prompts Are Designed

System prompt: Establishes an expert script analyst persona. The /no_think prefix
disables Qwen3's chain-of-thought reasoning mode for faster responses on this task.

User prompt: Injects the script + a per-field evaluation rubric that requires the
model to reference specific moments from the script, not give generic advice.

Schema enforcement: Pydantic model_json_schema() is passed directly to ollama.chat()
format= parameter. Temperature is set to 0 for maximum determinism.

## Limitations
- Requires Ollama installed and running locally
- qwen3:8b may miss nuance in very short or ambiguous scenes
- Engagement score is heuristic, not trained on real OTT engagement data
- No multi-language support
- First inference is slow while model loads into memory (~10-30 seconds)

## Possible Improvements
- Revision assistant mode: compare two script versions side-by-side
- PDF/FDX script file upload
- Genre detection with genre-specific scoring rubrics
- Fine-tune a Qwen3 LoRA on real engagement data for calibrated scoring
- Integrate with Bullet's internal content database for benchmark comparisons
```

---

## ✅ Build Checklist (Execution Order)

Execute in this exact sequence:

1. **[ ] Verify Ollama** — run `ollama --version` and `ollama pull qwen3:8b`
2. **[ ] Create project skeleton** — all folders + empty `__init__.py`
3. **[ ] Write `models.py`** — all Pydantic schemas (contract-first approach)
4. **[ ] Write `prompts.py`** — `SYSTEM_PROMPT` constant + `build_user_prompt()` function
5. **[ ] Write `pipeline.py`** — `get_available_model()` + `analyze_script()` using `ollama.chat()` with `format=` and `temperature=0`
6. **[ ] Smoke test pipeline in isolation** — write a throwaway `test.py` that calls `analyze_script("Test", "A man walks into a bar alone.")` and prints the result — verify Pydantic model parses cleanly
7. **[ ] Write `app.py`** — health check at top → sidebar → input form → spinner → tabs with results
8. **[ ] Pre-fill sample script** — hardcode `SAMPLE_TITLE` and `SAMPLE_SCRIPT` constants
9. **[ ] Test full app** — `streamlit run app.py`, click Analyze, verify all 5 tabs render correctly
10. **[ ] Write `README.md`** — all sections above
11. **[ ] Write `requirements.txt`** — `ollama`, `streamlit`, `pydantic`
12. **[ ] Final review** — no secrets hardcoded, no unnecessary files, README covers all assignment deliverable requirements

---

## 🧪 Validation Criteria

Verify every item before submitting:

- [ ] `ollama serve` + `streamlit run app.py` is the complete launch sequence
- [ ] App shows a clear error (not a traceback) if Ollama is not running
- [ ] Script title and content accepted as free text inputs
- [ ] Story summary: 3-4 prose sentences (not bullet points)
- [ ] Emotional arc: minimum 3 beats with intensity values shown visually
- [ ] Engagement score: both a number and a progress bar displayed
- [ ] Engagement factors: each shows factor name, score, and script-specific reasoning
- [ ] At least 4 improvement suggestions rendered
- [ ] Cliffhanger section identifies a specific moment with explanation
- [ ] No API key required anywhere in the codebase
- [ ] README covers: approach, prompt design, model/tools used, limitations, improvements

---

## ⚠️ Critical Notes for Agent

- **The key API call is `ollama.chat(..., format=ScriptAnalysis.model_json_schema(), options={"temperature": 0})`** — this is what makes structured output reliable. Do NOT use raw string prompting to get JSON.
- **`/no_think` goes as the very first line of the system prompt** — this is a Qwen3-specific directive that disables chain-of-thought mode for faster output.
- **Auto model detection is required** — call `ollama.list()` and select the best available model from the preference list `["qwen3:32b", "qwen3:14b", "qwen3:8b"]`. Never hardcode a single model name.
- **The Streamlit app must work on first try for a demo** — pre-fill the sample script, add the health check, make it zero-friction.
- **Write inline comments explaining design choices** — evaluators will read the code and judge prompt engineering thinking. Explain WHY `/no_think`, WHY `format=` over prompt-based JSON, WHY `temperature=0`.
- **Keep all functions small and single-responsibility** — `get_available_model()`, `analyze_script()`, `build_user_prompt()` are all separate.

---

*Workflow prepared by Senior AI Engineer. Stack: Ollama + Qwen3. Local-first, zero API friction, demo-ready.*
