"""
Pipeline orchestration — Ollama client logic + model auto-detection.

Architecture Decision: Single-Call Design
─────────────────────────────────────────
This pipeline makes ONE LLM call per analysis. Why not multiple calls
(one per dimension)?

1. Cross-dimensional coherence: When the LLM reasons about emotional arc,
   engagement score, and suggestions in one pass, the analysis is internally
   consistent. A multi-call approach risks contradictions (e.g., "high tension"
   in emotions but "low tension" in engagement factors).

2. Latency: One call ≈ 15-30 seconds. Five calls ≈ 75-150 seconds. For a
   demo, single-call is dramatically better UX.

3. Simplicity: No orchestration logic, no partial failure handling, no
   result merging. The entire output is one validated Pydantic object.

Schema Enforcement Strategy
───────────────────────────
The format= parameter in ollama.chat() accepts a JSON Schema dict. When
provided, Ollama constrains the model's token generation to ONLY produce
valid JSON matching that schema. This is fundamentally different from
prompt-based JSON coercion:

- Prompt-based: "Please respond in JSON with these fields..." → model
  can still emit markdown fences, preamble text, or malformed JSON
- format= based: Model is physically unable to emit non-conforming tokens

We generate the JSON Schema from Pydantic using model_json_schema(),
creating a single source of truth: Pydantic model → JSON Schema → LLM
constraint → Pydantic validation. No schema drift possible.
"""

import ollama
from pydantic import ValidationError
from .models import ScriptAnalysis
from .prompts import SYSTEM_PROMPT, build_user_prompt

# Ordered by quality (best first). The auto-detection logic picks the best
# model that the user has already pulled into their local Ollama instance.
# This means the app works on any hardware without config changes:
# - 32GB+ RAM users get qwen3:32b (highest quality)
# - 16GB RAM users get qwen3:14b (good balance)
# - 8GB RAM users get qwen3:8b (fast, still capable)
MODEL_PREFERENCE = ["qwen3:32b", "qwen3:14b", "qwen3:8b"]


def get_available_model() -> str:
    """Auto-detect the best Qwen3 model available in local Ollama.

    Queries ollama.list() for all downloaded models, then returns the first
    match from MODEL_PREFERENCE (best quality first). Falls back to qwen3:8b
    if none are found (Ollama will auto-pull it on first use).

    This design means the user never needs to configure a model name —
    the app adapts to whatever hardware they have.
    """
    try:
        # ollama.list() returns all models currently downloaded
        available = [m.model for m in ollama.list().models]
        for model in MODEL_PREFERENCE:
            if model in available:
                return model
        # If no Qwen3 model is pulled yet, default to the smallest.
        # Ollama will auto-download it on first inference call.
        return "qwen3:8b"
    except Exception:
        # If Ollama isn't running, return default. The health check in
        # app.py will catch this before we ever reach analyze_script().
        return "qwen3:8b"


def analyze_script(title: str, script_content: str) -> ScriptAnalysis:
    """Run script analysis using Ollama's native structured output mode.

    This is the core function. Key design choices explained:

    1. format=ScriptAnalysis.model_json_schema():
       Passes the Pydantic-generated JSON Schema directly to Ollama.
       The model is constrained at the INFERENCE level to produce only
       valid JSON matching this schema. No prompt-based JSON coercion,
       no risk of markdown fences or preamble in output.

    2. options={"temperature": 0}:
       Maximizes determinism. With temperature > 0, the model might
       occasionally produce edge-case values (engagement_score: 10.5)
       that violate schema constraints. Temperature 0 picks the
       highest-probability token every time, which aligns best with
       structured output requirements.

    3. Single message pair (system + user):
       No multi-turn conversation needed. The system prompt establishes
       the persona, the user prompt provides the script + rubric.
       One call, one complete analysis.

    Args:
        title: Script title (used in the prompt and echoed in output)
        script_content: Full script text to analyze

    Returns:
        ScriptAnalysis: Fully validated Pydantic model with all analysis fields

    Raises:
        ValueError: If model output doesn't match the expected schema
                    (extremely rare with format= enforcement)
        Exception: If Ollama is not running or model is unavailable
    """
    model = get_available_model()
    user_prompt = build_user_prompt(title, script_content)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # ── This is the critical line ──
        # model_json_schema() converts our Pydantic model to a JSON Schema dict.
        # Ollama uses this to constrain token generation at inference time.
        # The model CANNOT produce output that violates this schema.
        format=ScriptAnalysis.model_json_schema(),
        # ── Deterministic output ──
        # temperature=0 ensures the model always picks the highest-probability
        # token, maximizing consistency and schema adherence.
        options={"temperature": 0},
    )

    # Extract the raw JSON string from Ollama's response
    raw = response["message"]["content"]

    # Validate + parse using Pydantic. This is a safety net — with format=
    # enforcement, the JSON should always be valid. But Pydantic validation
    # also enforces constraints like ge=1, le=10 on intensity fields.
    try:
        return ScriptAnalysis.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"Model output did not match expected schema: {e}") from e
