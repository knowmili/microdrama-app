"""
Pydantic v2 output schemas for script analysis.

These models define the EXACT structure that Ollama must produce. The key design
decision here is contract-first development: we define the output schema BEFORE
writing the pipeline or prompts. This ensures:

1. The Pydantic schema is passed to ollama.chat(format=...) as a JSON Schema,
   which constrains the model at the INFERENCE LEVEL — not via prompt tricks.
2. The same schema validates the raw JSON response, giving us type-safe Python
   objects with zero parsing boilerplate.
3. Prompt engineers can read these models to understand exactly what the LLM
   must produce, making the prompt ↔ schema contract explicit.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EmotionBeat(BaseModel):
    """A single emotional moment in the script's arc.

    Captures the chronological progression of emotions — the LLM maps each
    story beat to an emotion and intensity, creating a visual emotional arc.
    """
    moment: str = Field(description="Story beat label, e.g. 'Opening', 'Midpoint', 'Climax'")
    emotion: str = Field(description="Dominant emotion at this beat, e.g. 'Anticipation', 'Grief'")
    intensity: int = Field(description="Emotional intensity from 1 (subtle) to 10 (overwhelming)", ge=1, le=10)


class EngagementFactor(BaseModel):
    """A named engagement dimension with a score and script-specific reasoning.

    Each factor must reference a SPECIFIC moment from the analyzed script —
    this forces the LLM to ground its analysis in the text rather than giving
    generic advice.
    """
    factor: str = Field(description="Factor name, e.g. 'Opening Hook', 'Character Conflict'")
    score: int = Field(description="Score from 1 (weak) to 10 (excellent)", ge=1, le=10)
    reasoning: str = Field(description="Specific reasoning tied to moments in THIS script")


class CliffhangerMoment(BaseModel):
    """The single most suspenseful moment in the script (bonus deliverable).

    Identifies the exact dialogue or scene beat that creates unresolved tension,
    plus an explanation of the mechanical craft behind it.
    """
    moment: str = Field(description="The exact dialogue line or scene beat that creates suspense")
    explanation: str = Field(description="Why this moment creates tension — the craft behind it")


class ScriptAnalysis(BaseModel):
    """Complete analysis output for a short-form script.

    This is the top-level schema passed to ollama.chat(format=...). Every field
    maps directly to an assignment deliverable:
    - title, summary → Story Summary
    - dominant_emotions, emotional_arc → Emotional Tone Analysis
    - engagement_score, engagement_factors → Engagement Score
    - improvement_suggestions → Improvement Suggestions
    - cliffhanger → Cliffhanger Detection (bonus)
    """
    title: str = Field(description="Title of the analyzed script")
    summary: str = Field(description="3-4 sentence narrative overview covering plot, stakes, and emotional core")
    dominant_emotions: List[str] = Field(description="Top 3 emotions felt by the audience")
    emotional_arc: List[EmotionBeat] = Field(description="3-5 chronological emotional beats")
    engagement_score: float = Field(description="Overall engagement score from 0.0 to 10.0", ge=0.0, le=10.0)
    engagement_factors: List[EngagementFactor] = Field(description="3-5 scored engagement dimensions")
    improvement_suggestions: List[str] = Field(description="4-6 actionable, script-specific suggestions")
    cliffhanger: Optional[CliffhangerMoment] = Field(
        default=None,
        description="The single most suspenseful moment and why it works (bonus)"
    )
